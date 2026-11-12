"""[2026-11-06] Arm / jaw execution latency per side, measured the UMI way (Chi et al. 2024, "latency matching"): a small chirp is
commanded on one joint per arm and on both jaws through HttpBridge.send (the only sanctioned 14-vector path, 30 Hz like the run
streamer), /observe is polled as fast as it answers, and the delay of the measured signal against the commanded one is the peak of
their cross-correlation.  It includes everything after the send: HTTP, the bridge's goal interpolation, the 100 Hz MIT loop and the
motor -- exactly what the run streamer's send-ahead (`exec_lead_ms` / `jaw_lead_ms`) has to cover.  Result -> LATENCY_FILE (arm / jaw
keys; tools.cam_latency writes the camera keys of the same file).

Refuses unless the UI is idle, both arms are within --max-from-start-deg of start_pose.json (a known table-clear pose) and every
sample of the chirp keeps the TCP above --min-z with the arm-arm capsule check ok.  Jaws stay between --jaw-lo and --jaw-hi (g, 1 =
open), away from the closed clip and the opening limit.
    python -m laundry_deploy.tools.latency_probe [--go]
"""

from __future__ import annotations
import argparse
import json
import sys
import threading
import time

import numpy as np

from ..core.collision import Collision
from ..core.contract import SIDES
from ..core.errors import DeployError, HardwareError, RunRefused
from ..core.kinematics import fk, makeArms
from ..session.paths import LATENCY_FILE, START_FILE

STREAM_HZ = 30.0
POLL_HZ = 100.0
BRIDGE_TIMEOUT_S = 0.5
UI_STATUS_URL = "http://127.0.0.1:8770/api/status"
PROBE_JOINT = 0  # shoulder_pan: a horizontal swing keeps the TCP height (the start pose is ~3 cm above the table)
CHIRP_F0_HZ, CHIRP_F1_HZ = 0.3, 1.2  # periods (0.8 .. 3.3 s) far above the lag window -> one unambiguous peak
RAMP_S = 1.0  # amplitude fade in / out
GRID_HZ = 500.0  # resampling grid of the cross-correlation
MAX_LAG_S = 0.6
CHECK_SAMPLES = 61
Z_SLACK_M = 0.003
MS_PER_S = 1e3
NS_PER_S = 1e9


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser()
    tParser.add_argument("--go", action="store_true")
    tParser.add_argument("--secs", type=float, default=12.0)
    tParser.add_argument("--amp-deg", type=float, default=4.0)
    tParser.add_argument("--joint", type=int, default=PROBE_JOINT, help="probed joint index (0 = shoulder_pan)")
    tParser.add_argument("--jaw-lo", type=float, default=0.15)
    tParser.add_argument("--jaw-hi", type=float, default=0.55)
    tParser.add_argument("--max-from-start-deg", type=float, default=8.0)
    tParser.add_argument("--min-z", type=float, default=0.04)
    tParser.add_argument("--url", default="http://127.0.0.1:8021")
    return tParser


def chirp(t: np.ndarray, secs: float) -> np.ndarray:
    """unit-amplitude linear chirp f0 -> f1 with RAMP_S fades (starts and ends at 0)"""
    tK = (CHIRP_F1_HZ - CHIRP_F0_HZ) / secs
    tEnv = np.clip(np.minimum(t, secs - t) / RAMP_S, 0.0, 1.0)
    return tEnv * np.sin(2 * np.pi * (CHIRP_F0_HZ * t + 0.5 * tK * t * t))


def estimateLag(tCmd: np.ndarray, yCmd: np.ndarray, tMeas: np.ndarray, yMeas: np.ndarray) -> tuple[float, float]:
    """delay (s) of the measured signal behind the commanded one + the normalized correlation at that delay.  The command is held
    (zero-order) between sends, the measurement linearly interpolated; both are mean-removed on a GRID_HZ grid."""
    tLo, tHi = max(tCmd[0], tMeas[0]), min(tCmd[-1], tMeas[-1])
    tGrid = np.arange(tLo, tHi, 1.0 / GRID_HZ)
    tIdx = np.clip(np.searchsorted(tCmd, tGrid, side="right") - 1, 0, len(tCmd) - 1)
    tA = yCmd[tIdx] - yCmd[tIdx].mean()
    tB = np.interp(tGrid, tMeas, yMeas)
    tB = tB - tB.mean()
    tN = int(MAX_LAG_S * GRID_HZ)
    tScores = np.array([np.dot(tA[: len(tA) - tL], tB[tL:]) / (np.linalg.norm(tA[: len(tA) - tL]) * np.linalg.norm(tB[tL:]) + 1e-12)
                        for tL in range(tN + 1)])
    tBest = int(np.argmax(tScores))
    tFrac = 0.0
    if 0 < tBest < tN:  # parabolic peak refinement
        tY0, tY1, tY2 = tScores[tBest - 1: tBest + 2]
        tDen = tY0 - 2 * tY1 + tY2
        tFrac = 0.5 * (tY0 - tY2) / tDen if tDen != 0 else 0.0
    return (tBest + tFrac) / GRID_HZ, float(tScores[tBest])


def _uiIdle() -> bool:
    import requests

    try:
        return requests.get(UI_STATUS_URL, timeout=1.0).json().get("mode") == "idle"
    except Exception:
        return True  # UI not running


def latencyProbe(args) -> dict:
    from laundry_shared.visual_teleop.robot.bridge import HttpBridge
    from laundry_shared.visual_teleop.tools.robot_moves import unpark

    if not _uiIdle():
        raise RunRefused("UI 가 실행 중(GO) -- 끝난 뒤에 측정")
    tStart = json.loads(START_FILE.read_text())
    tQs = {tSide: np.asarray(tStart["q"][tSide], float) for tSide in SIDES}
    tReader = HttpBridge(args.url, live=False, timeout_s=BRIDGE_TIMEOUT_S)
    tObs = tReader.observe()
    if tObs.estop:
        raise HardwareError("로봇 E-STOP/미연결/읽기 지연")
    tFar = max(float(np.degrees(np.abs(tObs.q[tSide] - tQs[tSide])).max()) for tSide in SIDES)
    print(f"시작 자세와 최대 관절 차이 {tFar:.1f}도 (한도 {args.max_from_start_deg})")
    if tFar > args.max_from_start_deg:
        raise RunRefused("시작 자세가 아님 -- UI '시작 자세로 이동' 후 측정")
    tQ0 = {tSide: tObs.q[tSide].copy() for tSide in SIDES}
    tAmp = np.radians(args.amp_deg)
    tArms, tRobotArgs = makeArms()
    tCol = Collision(tArms)
    # the start pose may itself sit below --min-z (start_lift 2 cm): then the swing only may not go lower than it
    tZ0 = min(fk(tArms, tSide, tQ0[tSide])[2, 3] for tSide in SIDES)
    tZMin = min(args.min_z, tZ0 - Z_SLACK_M)
    for tU in np.linspace(-1, 1, CHECK_SAMPLES):
        tQ = {tSide: tQ0[tSide].copy() for tSide in SIDES}
        for tSide in SIDES:
            tQ[tSide][args.joint] += tU * tAmp
        tZ = min(fk(tArms, tSide, tQ[tSide])[2, 3] for tSide in SIDES)
        if tZ < tZMin:
            raise RunRefused(f"흔들기 범위에서 TCP {tZ:.3f} m < {tZMin:.3f} -- 측정 안 함")
        if not tCol.check(tQ, tableZ=tRobotArgs.table_z)["ok_arm"]:
            raise RunRefused("흔들기 범위에서 두 팔 충돌 위험 -- 측정 안 함")
    print(f"검사 통과: 관절 {args.joint} ±{args.amp_deg}도, 집게 g {args.jaw_lo}..{args.jaw_hi}, {args.secs:.0f}초")
    if not args.go:
        print("계획만 출력 (--go 로 측정, 사람이 옆에서)")
        return {}
    tBridge = HttpBridge(args.url, live=True, timeout_s=BRIDGE_TIMEOUT_S)
    for tSide in SIDES:
        unpark(tBridge, tSide)
    tJawMid, tJawAmp = 0.5 * (args.jaw_lo + args.jaw_hi), 0.5 * (args.jaw_hi - args.jaw_lo)
    tMeas = []
    tDone = threading.Event()

    def poll():
        # /observe only (HttpBridge.observe adds a /status per read) at POLL_HZ: the bridge closes every connection, and an
        # unthrottled loop ran this Mac out of ephemeral ports (Errno 49, 18k sockets in TIME_WAIT) within the 12 s
        from laundry_shared.visual_teleop.robot.bridge import parse_observe

        while not tDone.is_set():
            tA = time.monotonic_ns()
            try:
                tD = tReader.s.get(tReader.base_url + "/observe", timeout=BRIDGE_TIMEOUT_S).json()
                tO = parse_observe(tD, time.monotonic_ns())
            except Exception:
                time.sleep(1.0 / POLL_HZ)
                continue
            tMeas.append((0.5 * (tA + time.monotonic_ns()) / NS_PER_S, tO))  # mid-request = the read time, +- half the round trip
            time.sleep(max(0.0, tA / NS_PER_S + 1.0 / POLL_HZ - time.monotonic()))

    # settle the jaws on the band's middle first (a closed jaw would start clipped)
    for _ in range(int(STREAM_HZ)):
        tBridge.send(tQ0, {tSide: tJawMid for tSide in SIDES})
        time.sleep(1.0 / STREAM_HZ)
    tPoller = threading.Thread(target=poll, daemon=True)
    tPoller.start()
    tCmd = []
    tT0 = time.monotonic()
    tK = 0
    try:
        while True:
            tT = tK / STREAM_HZ
            if tT > args.secs:
                break
            tC = chirp(np.array([tT]), args.secs)[0]
            tQ = {tSide: tQ0[tSide].copy() for tSide in SIDES}
            for tSide in SIDES:
                tQ[tSide][args.joint] += tAmp * tC
            tG = {tSide: tJawMid + tJawAmp * tC for tSide in SIDES}
            tReply = tBridge.send(tQ, tG)
            if not tReply.accepted:
                tBridge.estop()
                raise HardwareError(f"브리지 거부: {tReply.error} -- E-STOP")
            tCmd.append((tReply.t_ns / NS_PER_S, tC))  # the command is in effect from the bridge's reply on
            tK += 1
            time.sleep(max(0.0, tT0 + tK / STREAM_HZ - time.monotonic()))
        time.sleep(MAX_LAG_S)
    finally:
        tDone.set()
        tPoller.join(timeout=2.0)
        tBridge.send(tQ0, {tSide: tJawMid for tSide in SIDES})
        tBridge.close()
    tTc = np.array([tX[0] for tX in tCmd])
    tYc = np.array([tX[1] for tX in tCmd])
    tTm = np.array([tX[0] for tX in tMeas])
    tOut = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), n_cmd=len(tCmd), n_meas=len(tMeas),
                meas_hz=round(len(tMeas) / max(tTm[-1] - tTm[0], 1e-6), 1), arm_ms={}, jaw_ms={}, corr={})
    for tSide in SIDES:
        tArm = np.array([tO.q[tSide][args.joint] for _, tO in tMeas])
        tJaw = np.array([tO.g(tSide) for _, tO in tMeas])
        tLa, tCa = estimateLag(tTc, tYc, tTm, tArm)
        tLj, tCj = estimateLag(tTc, tYc, tTm, tJaw)
        tOut["arm_ms"][tSide] = round(tLa * MS_PER_S, 1)
        tOut["jaw_ms"][tSide] = round(tLj * MS_PER_S, 1)
        tOut["corr"][tSide] = dict(arm=round(tCa, 3), jaw=round(tCj, 3))
    print(json.dumps(tOut, indent=1, ensure_ascii=False))
    tLow = [f"{tSide} {tKey}" for tSide in SIDES for tKey, tV in tOut["corr"][tSide].items() if tV < 0.8]
    if tLow:
        print(f"주의: 상관이 낮음 ({', '.join(tLow)}) -- 저장 안 함")
        return tOut
    tAll = json.loads(LATENCY_FILE.read_text()) if LATENCY_FILE.exists() else {}
    tAll.update(probe=tOut, arm_ms=tOut["arm_ms"], jaw_ms=tOut["jaw_ms"])
    LATENCY_FILE.write_text(json.dumps(tAll, indent=1, ensure_ascii=False))
    print("저장", LATENCY_FILE)
    return tOut


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    try:
        latencyProbe(tArgs)
    except DeployError as tExc:
        print(tExc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
