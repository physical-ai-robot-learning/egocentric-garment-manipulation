"""[2026-10-28] Joint-space return to start_pose.json from a nearby pose (after a few policy steps), so the UI's GO check (READY or
start pose within 5 deg) passes again (robot helper).

Refuses if any joint is more than --max-deg away (not "nearby"), if the TCP dips below --min-z anywhere on the straight joint path,
or if the capsule arm-arm check fails. /clear_estop first (the bridge then holds the measured pose), smoothstep ramp at <= --deg-s
through HttpBridge.send (the only sanctioned 14-vector path), E-STOP on refusal.
    python -m laundry_deploy.tools.return_to_start [--go]
Was laundry_deploy/return_to_start.py.
"""

from __future__ import annotations
import argparse
import json
import sys
import time

import numpy as np

from ..core.collision import Collision
from ..core.contract import SIDES
from ..core.errors import DeployError, HardwareError, RunRefused
from ..core.kinematics import fk, makeArms
from ..session.paths import START_FILE

PATH_SAMPLES = 41
STREAM_HZ = 30.0
ESTOP_CHECK_EVERY = 15  # ticks between /observe E-STOP checks during the ramp
BRIDGE_TIMEOUT_S = 0.5
CLEAR_ESTOP_TIMEOUT_S = 2.0
CLEAR_ESTOP_SETTLE_S = 0.3
ARRIVE_SETTLE_S = 0.8


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser()
    tParser.add_argument("--go", action="store_true")
    tParser.add_argument("--deg-s", type=float, default=10.0)
    tParser.add_argument("--max-deg", type=float, default=25.0)
    tParser.add_argument("--min-z", type=float, default=0.04)
    tParser.add_argument("--url", default="http://127.0.0.1:8021")
    return tParser


def returnToStart(args) -> None:
    from laundry_shared.visual_teleop.robot.bridge import HttpBridge
    from laundry_shared.visual_teleop.tools.robot_moves import unpark

    tStartPose = json.loads(START_FILE.read_text())
    tQs = {tSide: np.asarray(tStartPose["q"][tSide], float) for tSide in SIDES}
    tArms, tRobotArgs = makeArms()
    tCol = Collision(tArms)
    tBridge = HttpBridge(args.url, live=args.go, timeout_s=BRIDGE_TIMEOUT_S)
    tObs = tBridge.observe()
    tQ0 = {tSide: tObs.q[tSide].copy() for tSide in SIDES}
    tD = {tSide: tQs[tSide] - tQ0[tSide] for tSide in SIDES}
    tMax = max(float(np.degrees(np.abs(tD[tSide])).max()) for tSide in SIDES)
    print(f"시작 자세까지 최대 관절 차이 {tMax:.1f}도 (한도 {args.max_deg})")
    if tMax > args.max_deg:
        raise RunRefused("시작 자세에서 너무 멀다 -- READY 경유로 이동하세요")
    for tU in np.linspace(0, 1, PATH_SAMPLES):
        tQ = {tSide: tQ0[tSide] + tU * tD[tSide] for tSide in SIDES}
        tZ = min(fk(tArms, tSide, tQ[tSide])[2, 3] for tSide in SIDES)
        if tZ < args.min_z:
            raise RunRefused(f"경로 중 TCP 높이 {tZ:.3f} m < {args.min_z} -- 이동 안 함")
        if not tCol.check(tQ, tableZ=tRobotArgs.table_z)["ok_arm"]:
            raise RunRefused("경로 중 두 팔 충돌 위험 -- 이동 안 함")
    tN = max(2, int(np.ceil(tMax / args.deg_s * STREAM_HZ)))
    print(f"경로 검사 통과: {tN / STREAM_HZ:.1f}초, TCP 최저 >= {args.min_z} m")
    if not args.go:
        print("계획만 출력 (--go 로 이동)")
        return
    if tBridge.status().get("estop"):
        tBridge.s.post(tBridge.base_url + "/clear_estop", timeout=CLEAR_ESTOP_TIMEOUT_S)
        time.sleep(CLEAR_ESTOP_SETTLE_S)
    for tSide in SIDES:
        unpark(tBridge, tSide)
    tObs = tBridge.observe()
    tQ0 = {tSide: tObs.q[tSide].copy() for tSide in SIDES}
    tD = {tSide: tQs[tSide] - tQ0[tSide] for tSide in SIDES}
    tG = {tSide: tObs.g(tSide) for tSide in SIDES}
    tT0 = time.monotonic()
    for tK in range(1, tN + 1):
        tU = tK / tN
        tU = tU * tU * (3 - 2 * tU)
        tReply = tBridge.send({tSide: tQ0[tSide] + tU * tD[tSide] for tSide in SIDES}, tG)
        if not tReply.accepted:
            tBridge.estop()
            raise HardwareError(f"브리지 거부: {tReply.error} -- E-STOP")
        if tK % ESTOP_CHECK_EVERY == 0 and tBridge.observe().estop:
            tBridge.estop()
            raise HardwareError("이동 중 E-STOP/읽기 지연")
        time.sleep(max(0.0, tT0 + tK / STREAM_HZ - time.monotonic()))
    time.sleep(ARRIVE_SETTLE_S)
    tObs = tBridge.observe()
    print(
        "도착: 시작 자세와 최대 차이 "
        + ", ".join(f"{tSide} {np.degrees(np.abs(tObs.q[tSide] - tQs[tSide])).max():.1f}도" for tSide in SIDES)
    )
    tBridge.close()


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    try:
        returnToStart(tArgs)
    except DeployError as tExc:
        print(tExc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
