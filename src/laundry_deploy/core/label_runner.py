"""[10-27] Oracle 'policy' for label replay (A' vs Cg test, deploy stage): the training label of one demo (robotize_ik
ik_v1/demoNN.npz, <side>_p_rob / <side>_R_rob = the action frame, <side>_g = umi_jaw state 1 = open) played through the unchanged
deploy loop.

predict() returns rows k = 1..32 of inv(T(tau)) T(tau + k ROW_DT_S rate) at label time tau = (now - t0) * rate (rate < 1 = slow
motion), i.e. the same relative chunk a perfect policy would output; the deploy loop composes it onto the measured FK pose as for a
policy.
"""

from __future__ import annotations
import time

import numpy as np

from laundry_shared.contracts.action_contract import T_to_pose9, inv_T

from .contract import ACTION_DIM, CHUNK_ROWS, GRIP_INDEX, NS_PER_S, POSE_SLICE, ROW_DT_S, SIDES

LABEL_KIND = "label"
DEFAULT_RATE = 0.5
WORKSPACE_MARGIN_M = 0.05
END_TOL_S = 1e-6  # the cursor counts as at the label end within this


class LabelRunner:
    mKind = LABEL_KIND
    mNObs = 1

    def __init__(self, path, rate: float = DEFAULT_RATE) -> None:
        from scipy.spatial.transform import Rotation, Slerp

        tZ = np.load(path, allow_pickle=True)
        self.mPath = str(path)
        self.mRate = float(rate)
        self.mT0 = None
        self.mCursor = None  # label seconds: set by the deploy loop (--go / dry chaining)
        tT = (tZ["t_ns"].astype(np.int64) - int(tZ["t_ns"][0])) / NS_PER_S
        self.mT = tT
        self.mDurS = float(tT[-1])
        self.mP = {tSide: tZ[f"{tSide}_p_rob"].astype(float) for tSide in SIDES}
        self.mG = {tSide: tZ[f"{tSide}_g"].astype(float) for tSide in SIDES}
        self.mSlerp = {tSide: Slerp(tT, Rotation.from_matrix(tZ[f"{tSide}_R_rob"])) for tSide in SIDES}

    def poseAt(self, side: str, tau: float) -> np.ndarray:
        tau = float(np.clip(tau, 0.0, self.mDurS))
        tT = np.eye(4)
        tT[:3, 3] = [np.interp(tau, self.mT, self.mP[side][:, tJ]) for tJ in range(3)]
        tT[:3, :3] = self.mSlerp[side](tau).as_matrix()
        return tT

    def gripAt(self, side: str, tau: float) -> float:
        return float(self.mG[side][min(np.searchsorted(self.mT, np.clip(tau, 0, self.mDurS)), len(self.mT) - 1)])

    def tau(self) -> float:
        return 0.0 if self.mT0 is None else (time.monotonic() - self.mT0) * self.mRate

    def item(self, imgsRgb, state30, prev=None):
        return None

    def predict(self, item, seed: int = 0):
        tTau = self.tau() if self.mCursor is None else self.mCursor
        tChunk = np.zeros((CHUNK_ROWS, ACTION_DIM))
        # one vectorised slerp / interp per arm (216 -> ~2 ms)
        tTs = np.clip(tTau + np.arange(CHUNK_ROWS + 1) * ROW_DT_S * self.mRate, 0.0, self.mDurS)
        for tSide in SIDES:
            tRot = self.mSlerp[tSide](tTs).as_matrix()
            tPos = np.stack([np.interp(tTs, self.mT, self.mP[tSide][:, tJ]) for tJ in range(3)], 1)
            tT = np.tile(np.eye(4), (CHUNK_ROWS + 1, 1, 1))
            tT[:, :3, :3] = tRot
            tT[:, :3, 3] = tPos
            tT0Inv = inv_T(tT[0])
            for tK in range(1, CHUNK_ROWS + 1):
                tChunk[tK - 1, POSE_SLICE[tSide]] = T_to_pose9(tT0Inv @ tT[tK])
            tChunk[:, GRIP_INDEX[tSide]] = self.mG[tSide][np.minimum(np.searchsorted(self.mT, tTs[1:]), len(self.mT) - 1)]
        return tChunk, tChunk

    def done(self) -> bool:
        return self.mCursor is not None and self.mCursor >= self.mDurS - END_TOL_S

    def workspace(self, margin: float = WORKSPACE_MARGIN_M) -> dict:
        return {tSide: (self.mP[tSide].min(0) - margin, self.mP[tSide].max(0) + margin) for tSide in SIDES}
