"""Shared helpers of the Orbbec trackers (was orbbec/v2b/v32_common.py, 10-24, copied from the frozen refine_v4.py): per-arm AHRS
rotation on the Orbbec grid (rest-period bias, accel tilt, one yaw anchor per demo window at its first >= 2-tag raw R_tag frame or the
nearest other demo's, IMU clock lag per demo from a reference track) and the translation-only rigid-model fit.
Inputs: arm record (fgo.arm_data.loadArm), demo windows, a reference centre track; outputs: (n,3,3) rotations / a fitted translation.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from . import imu_motion
from .constants import NS_PER_S
from .fgo import arm_data
from .fgo import pose_smoother

DEMO_TAIL_S = 2.0                  # a demo window owns the frames up to its end + 2 s
MIN_LAG_SAMPLES = 30               # fewer finite reference samples -> lag 0
FIT_STEPS_M = (0.04, 0.025, 0.015, 0.012, 0.012)   # shrinking inlier radii of the translation-only fit
FIT_MIN_INLIERS = 40
FIT_RADIUS_M = 0.09                # points within this of the fitted centre count for the fit fraction


def armRotations(take: str, t: np.ndarray, trackC: np.ndarray, arm: str, anchorOk=None) -> np.ndarray:
    """(n,3,3) AHRS R_B_tcp on the Orbbec grid t (NaN outside demo windows); trackC = a reference centre track for the sync lag;
    anchorOk (bool on the take's tag frames) restricts the yaw-anchor frames (hidden-tag runs: visible tags only, no GT leakage)
    (was v32_common.arm_rotations)"""
    D = arm_data.loadArm(take, arm, withSlam=False)
    pose_smoother.prepGyro(D)
    tWindows = arm_data.loadTrimWindows()[take]
    D["b_rest"] = imu_motion.restBias(D, tWindows)
    tRb = D["R_tag"]
    tTag2 = (D["ntags"] >= 2) & (D["n_tagrot"] >= 2) & np.isfinite(tRb[:, 0, 0])
    tN = len(t)
    tRm = np.full((tN, 3, 3), np.nan)
    if anchorOk is not None:
        tTag2 &= anchorOk
    for tWin in tWindows:
        tK = np.flatnonzero((t >= tWin.mA0) & (t <= tWin.mW1 + DEMO_TAIL_S * NS_PER_S))
        if not len(tK):
            continue
        tAnc = np.flatnonzero((t >= tWin.mA0) & (t <= tWin.mW1 + DEMO_TAIL_S * NS_PER_S) & tTag2)
        if not len(tAnc) and tTag2.any():
            tJ = np.flatnonzero(tTag2)
            tAnc = tJ[[np.argmin(np.abs(t[tJ] - t[tK[0]]))]]
        if not len(tAnc):
            continue
        if np.isfinite(trackC[tK, 0]).sum() > MIN_LAG_SAMPLES:
            tLag, _ = imu_motion.syncLag(D, t, trackC, tK, t[tAnc[0]], R.from_matrix(tRb[tAnc[0]]))
        else:
            tLag = 0
        tRm[tK] = imu_motion.ahrsBase(D, t[tK] + tLag, t[tAnc[0]] + tLag, R.from_matrix(tRb[tAnc[0]])).as_matrix()
    return tRm


def rigidFit(pts: np.ndarray, Rm: np.ndarray, M: np.ndarray, c0: np.ndarray, steps=FIT_STEPS_M, minInl: int = FIT_MIN_INLIERS,
             fitR: float = FIT_RADIUS_M):
    """translation-only fit of model M (body frame) rotated by Rm, starting at c0 -> (T, n_inl, fit, rmse) or None
    (was v32_common.rigid_fit)"""
    tT = c0.copy()
    tMr = M @ Rm.T
    for tRadius in steps:
        tD, tJ = cKDTree(tMr + tT).query(pts, distance_upper_bound=tRadius)
        tInl = np.isfinite(tD)
        if tInl.sum() < minInl // 2:
            return None
        tT = tT + np.median(pts[tInl] - (tMr[tJ[tInl]] + tT), axis=0)
    tD, _ = cKDTree(tMr + tT).query(pts)
    tNear = np.linalg.norm(pts - tT, axis=1) < fitR
    tInl = tD < steps[-1]
    if tNear.sum() == 0:
        return None
    return tT, int(tInl.sum()), float((tInl & tNear).sum() / tNear.sum()), float(np.sqrt(np.mean(tD[tInl] ** 2))) if tInl.any() else np.nan
