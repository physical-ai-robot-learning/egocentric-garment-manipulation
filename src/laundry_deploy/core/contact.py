"""[2026-11-06 user: "땅밑으로 가면 torq 나 전류값으로 그것보다는 낮게 가지 않게"] Table contact from the arm's own feedback -> a
local floor that the commanded TCP never goes below again (this run).

[11-07 user: "높이는 하지 말고 토크로 해줘"] No table-height model (no table_z / z_floor / height-calibration plane).  The first
torque-only version (free-motion EMA baseline) fired 14 times in mid-air (+184..+434 mm): gravity alone moves the pitch-joint
torques 2-6 N*m between poses.  Now the WRIST FLEX torque minus a gravity model (GRAVITY_FILE, fitted on in-air samples of the
qtau.npz run logs by `python -m laundry_deploy.core.contact fit`): in-air residual p95 0.6-0.8 N*m; leave-one-run-out on the
11-07 17:25 / 17:30 / 17:33 runs, residual > 1.3 N*m + held > 3 mm above the command for 4 reads detected 45-95 % of the table
contact samples at 0-0.9 % false in the air.  Contact, per arm, at the 25 Hz robot reads (RobotPoller.mTauRing):
    blocked  the measured TCP sits more than gapM ABOVE the commanded one at that instant (streamer.anchor(t))
    pushed   wrist flex torque - gravity model(q, dq) > +tauNm (a table push; every real contact had a POSITIVE residual)
    not up   the commanded TCP is not rising (< RISE_MPS): the mid-air false hits (+132 / +600 / +691 mm) had a negative residual
             or a command going up at 19-61 mm/s (arm lagging behind a lift)
for HOLD_SAMPLES reads in a row -> a contact point (xy, measured z + CLEAR_M).  floor(side, xy) = the highest contact point
within RADIUS_M of xy, -inf elsewhere.  run_loop clamps every committed target to it, and Streamer's floor guard holds the
side's last sent q while the timeline itself is below it.
"""

from __future__ import annotations
import glob
import json
import sys
import threading
import time

import numpy as np

from laundry_common import paths as egopaths

from .contract import SIDES
from .kinematics import fk

GRAVITY_FILE = egopaths.TAGRIG_STATE_DIR / "contact_gravity.json"
HOLD_SAMPLES = 4
RADIUS_M = 0.06  # a contact point holds the floor this far around it (xy)
CLEAR_M = 0.003  # the floor sits this far above the measured contact (no scraping along it)
RISE_MPS = 0.005
TAU_JOINT = 3  # wrist_flex: the only joint whose gravity residual stays < 1 N*m in the air
FIT_AIR_Z_M = 0.06  # fit samples: TCP above this (the table is never touched there)
GUARD_TOL_M = 0.002
LIFT_STEP_M = 0.001
POLL_S = 0.02
NS_PER_S = 1e9
MM_PER_M = 1e3


def gravityFeatures(q: np.ndarray, dq: np.ndarray) -> np.ndarray:
    """(n, 6) joints + velocities (rad, rad/s) -> (n, 38): sin/cos of the pitch-chain sums and the other joints, their products,
    viscous and Coulomb terms"""
    tS1 = q[:, 1]
    tS12 = q[:, 1] + q[:, 2]
    tS123 = tS12 + q[:, 3]
    tF = [np.ones(len(q))]
    for tA in (tS1, tS12, tS123, q[:, 0], q[:, 4], q[:, 5]):
        tF += [np.sin(tA), np.cos(tA)]
    for tA in (tS1, tS12, tS123):
        for tB in (q[:, 0], q[:, 4], q[:, 5]):
            tF += [np.sin(tA) * np.cos(tB), np.cos(tA) * np.sin(tB)]
    tF += [dq[:, tJ] for tJ in range(6)] + [np.sign(dq[:, tJ]) for tJ in range(6)]
    return np.stack(tF, 1)


def backwardDq(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    """joint velocity as the detector sees it online: (q[i] - q[i-1]) / dt, 0 for the first read"""
    tDq = np.zeros_like(q)
    tDq[1:] = np.diff(q, axis=0) / np.maximum(np.diff(t), 1e-3)[:, None]
    return tDq


def fitGravity(runDirs: list, arms) -> dict:
    """least squares of the wrist flex torque on gravityFeatures over the in-air samples of every qtau.npz -> {side: weights}"""
    tOut = {}
    for tI, tSide in enumerate(SIDES):
        tX, tY = [], []
        for tDir in runDirs:
            tD = np.load(f"{tDir}/qtau.npz")
            tT, tQ, tTau = tD["t_ns"] / NS_PER_S, tD["q"][:, tI], tD["tau"][:, tI]
            tOk = np.isfinite(tTau).all(1) & np.isfinite(tQ).all(1)
            tT, tQ, tTau = tT[tOk], tQ[tOk], tTau[tOk]
            tZ = np.array([fk(arms, tSide, tQq)[2, 3] for tQq in tQ])
            tAir = tZ > FIT_AIR_Z_M
            tX.append(gravityFeatures(tQ, backwardDq(tQ, tT))[tAir])
            tY.append(tTau[tAir, TAU_JOINT])
        tW = np.linalg.lstsq(np.concatenate(tX), np.concatenate(tY), rcond=None)[0]
        tOut[tSide] = [float(tV) for tV in tW]
    return tOut


class ContactFloor(threading.Thread):
    def __init__(self, arms, poller, streamer, gapMm: float, tauNm: float, say=print) -> None:
        super().__init__(daemon=True)
        self.mArms, self.mPoller, self.mStreamer = arms, poller, streamer
        self.mGap, self.mTau, self.mSay = gapMm / MM_PER_M, float(tauNm), say
        tFile = json.loads(GRAVITY_FILE.read_text())
        self.mW = {tSide: np.asarray(tFile["w"][tSide], float) for tSide in SIDES}
        self.mGravityInfo = tFile.get("created", "?")
        self.mPoints = {tSide: [] for tSide in SIDES}  # (x, y, floor z) contact points
        self.mCount = {tSide: 0 for tSide in SIDES}
        self.mPrev = {tSide: None for tSide in SIDES}  # (t s, q, commanded z) of the previous read
        self.mEvents, self.mTrace = [], []
        self.mStop = False
        self.mLastT = None
        self.mErrorSaid = False

    def floor(self, side: str, xy) -> float:
        tZ = [tP[2] for tP in self.mPoints[side] if np.hypot(tP[0] - xy[0], tP[1] - xy[1]) < RADIUS_M]
        return max(tZ) if tZ else -np.inf

    def below(self, side: str, q) -> bool:
        """Streamer floor guard: the target q puts the TCP below this side's floor"""
        tT = fk(self.mArms, side, q)
        return tT[2, 3] < self.floor(side, tT[:2, 3]) - GUARD_TOL_M

    def step(self, tNs: int, q: dict, tau: dict) -> None:
        tQc = self.mStreamer.anchor(tNs / NS_PER_S)[1]
        tTs = tNs / NS_PER_S
        for tSide in SIDES:
            if tau.get(tSide) is None:
                continue
            tQ = np.asarray(q[tSide], float)
            tT = fk(self.mArms, tSide, tQ)
            tZ, tXy = float(tT[2, 3]), tT[:2, 3]
            tZc = float(fk(self.mArms, tSide, tQc[tSide])[2, 3])
            tPrev, self.mPrev[tSide] = self.mPrev[tSide], (tTs, tQ, tZc)
            tDt = 1e-3 if tPrev is None else max(tTs - tPrev[0], 1e-3)
            tDq = np.zeros(6) if tPrev is None else (tQ - tPrev[1]) / tDt
            tVzc = 0.0 if tPrev is None else (tZc - tPrev[2]) / tDt
            tRes = float(np.asarray(tau[tSide], float)[TAU_JOINT] - gravityFeatures(tQ[None], tDq[None])[0] @ self.mW[tSide])
            tGap = tZ - tZc
            self.mTrace.append((tNs, tSide, round(tZ, 4), round(tZc, 4), round(tRes, 3)))
            tHit = tGap > self.mGap and tRes > self.mTau and tVzc < RISE_MPS
            self.mCount[tSide] = self.mCount[tSide] + 1 if tHit else 0
            if self.mCount[tSide] >= HOLD_SAMPLES and tZ + CLEAR_M > self.floor(tSide, tXy) + LIFT_STEP_M:
                self.mPoints[tSide].append((float(tXy[0]), float(tXy[1]), tZ + CLEAR_M))
                self.mEvents.append(dict(t_ns=tNs, side=tSide, x=round(float(tXy[0]), 4), y=round(float(tXy[1]), 4), z=round(tZ, 4),
                                         z_cmd=round(tZc, 4), res_nm=round(tRes, 3)))
                self.mSay(f"[바닥 접촉] {tSide}: 명령보다 {tGap * MM_PER_M:.0f} mm 위에서 막힘, 손목 토크 {tRes:+.2f} N·m (중력 뺀 값) "
                          f"-> 이 위치 {RADIUS_M * MM_PER_M:.0f} mm 안에서 높이 {(tZ + CLEAR_M) * MM_PER_M:+.0f} mm 아래로 안 감")

    def run(self) -> None:
        while not self.mStop:
            tT, tValue = self.mPoller.mTauRing.latest()
            if tValue is not None and tT != self.mLastT:
                self.mLastT = tT
                try:
                    self.step(tT, *tValue)
                except Exception as tExc:  # never take the run down
                    if not self.mErrorSaid:
                        self.mSay(f"[바닥 접촉] 감지 오류 (이후 생략): {type(tExc).__name__}: {tExc}")
                        self.mErrorSaid = True
            time.sleep(POLL_S)


def main(argv: list[str]) -> int:
    """fit [run dir ...]: gravity model of the wrist flex torque from the qtau.npz logs (default: every run that has one)"""
    from .kinematics import makeArms

    if not argv or argv[0] != "fit":
        print("사용법: python -m laundry_deploy.core.contact fit [run 폴더 ...]")
        return 1
    tDirs = argv[1:] or sorted(tP.rsplit("/", 1)[0] for tP in glob.glob(str(egopaths.TRAIN_ROOT / "reports/act_deploy/*/qtau.npz")))
    tArms, _ = makeArms()
    tW = fitGravity(tDirs, tArms)
    GRAVITY_FILE.write_text(json.dumps(dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), runs=tDirs, joint=TAU_JOINT, w=tW), indent=1))
    print(f"중력 모델 저장 {GRAVITY_FILE} ({len(tDirs)} 개 run)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
