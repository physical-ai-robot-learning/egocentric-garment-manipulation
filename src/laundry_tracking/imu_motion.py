"""IMU motion consistency for Orbbec 3D tracks, the AHRS rotation and the Orbbec<->IMU clock lag (was orbbec/v2b/imu_motion.py,
10-23; tracker v2 building block; syncLag was build_candidate_v0.sync_lag).

Not a hard acceleration equality: both signals are low-passed (gaussian SMOOTH_SIGMA_S) and compared over NCC_WINDOW_S windows by
  ncc   = sum(a_o . a_i) / sqrt(sum|a_o|^2 sum|a_i|^2)      direction + timing agreement of the 3D linear acceleration
  energy gate: windows where neither moves (rms < ENERGY_MIN) are 'still' -> no evidence (score NaN), never a reject.
IMU linear acceleration in the base frame = R_B_tcp(t) f_tcp(t) - g, R_B_tcp from the gyro propagated from an anchor rotation; lever
arm ignored (low-pass keeps the omega^2 r / alpha x r terms small; the score is a correlation, not a difference).
Inputs: the arm record D (fgo.arm_data.loadArm, IMU on the orbbec clock in the TCP frame) and an Orbbec centre track.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from .constants import NS_PER_MS, NS_PER_S
from .intervals import trueRuns
from .slam_consistency import gyroRelative

SMOOTH_SIGMA_S = 0.08              # was SIG_S
NCC_WINDOW_S = 0.6                 # was WIN_S
ENERGY_MIN = 0.35                  # m/s^2 rms; was E_MIN
GRID_HZ = 30.0                     # was HZ
MAX_GAP_S = 0.25                   # track samples further apart are not interpolated across
MIN_STRETCH_SAMPLES = 8            # shorter interpolated stretches give no acceleration
MIN_TRACK_SAMPLES = 5
MIN_WINDOW_FINITE = 0.7            # a window needs this fraction of finite Orbbec samples
MIN_DENOMINATOR = 1e-12
# scripts/handumi_canonical_track.py CANON_V1 Mahony gains (dict: passed through as one parameter set)
AHRS_GAINS = dict(KP=1.0, KI=0.005, ACC_TOL=0.8, GYRO_MAX=3.0)
AHRS_INIT_SAMPLES = 50             # initial tilt from the mean of the first 50 accelerometer samples
AHRS_MAX_DT_S = 0.05
AHRS_REORTHO_EVERY = 200
REST_BIAS_BAND_DEG_S = 1.5         # rest samples within 1.5 deg/s of the period median
REST_MIN_SAMPLES = 50
REST_MIN_STILL = 200
LAG_RANGE_MS = (-500, 301, 20)     # searched Orbbec<->IMU lags (ms): range(-500, 301, 20)
LAG_MIN_NCC = 0.2                  # a best lag below this correlation is not trusted (lag 0 + flag)
LAG_MIN_VALID = 30                 # too few finite track accelerations -> lag 0
LAG_NO_NCC = -9.0


def motionGrid(t0: int, t1: int) -> np.ndarray:
    """30 Hz grid [t0, t1] (was imu_motion.grid)"""
    return np.arange(t0, t1 + 1, int(NS_PER_S / GRID_HZ), dtype=np.int64)


def imuLinearAcc(D: dict, g: np.ndarray, anchorT: int, anchorR: R) -> np.ndarray:
    """specific force -> base frame (gyro-propagated rotation) -> minus gravity (median over the window) -> low-pass
    (was imu_motion.imu_lin_acc(D, C, g, anchor_t, anchor_R); the core-module argument is gone)"""
    tTi = D["imu_t"]
    tF = np.stack([np.interp(g, tTi, D["acc"][:, tK]) for tK in range(3)], 1)
    tRt = anchorR * gyroRelative(D, np.full(len(g), anchorT), g)
    tFb = tRt.apply(tF)
    tGb = np.median(tFb, 0)
    tA = tFb - tGb
    return gaussian_filter1d(tA, SMOOTH_SIGMA_S * GRID_HZ, axis=0, mode="nearest")


def trackAcc(t: np.ndarray, c: np.ndarray, g: np.ndarray, maxGapS: float = MAX_GAP_S) -> np.ndarray:
    """Orbbec centre (valid samples) -> linear on the grid inside stretches (gaps <= maxGapS) -> low-pass -> 2nd derivative; NaN
    elsewhere (was imu_motion.track_acc)"""
    tValid = np.isfinite(c[:, 0])
    tTv, tCv = t[tValid], c[tValid]
    tOut = np.full((len(g), 3), np.nan)
    if len(tTv) < MIN_TRACK_SAMPLES:
        return tOut
    tJ = np.clip(np.searchsorted(tTv, g, side="right"), 1, len(tTv) - 1)
    tOk = ((tTv[tJ] - tTv[tJ - 1]) <= maxGapS * NS_PER_S) & (g >= tTv[0]) & (g <= tTv[-1])
    tF = ((g - tTv[tJ - 1]) / np.maximum(tTv[tJ] - tTv[tJ - 1], 1))[:, None]
    tP = tCv[tJ - 1] * (1 - tF) + tCv[tJ] * tF
    for tA, tB in trueRuns(tOk):
        if tB - tA < MIN_STRETCH_SAMPLES:
            continue
        tPs = gaussian_filter1d(tP[tA:tB], SMOOTH_SIGMA_S * GRID_HZ, axis=0, mode="nearest")
        tOut[tA:tB] = np.gradient(np.gradient(tPs, 1 / GRID_HZ, axis=0), 1 / GRID_HZ, axis=0)
    return tOut


def windowNcc(ao: np.ndarray, ai: np.ndarray, winS: float = NCC_WINDOW_S, eMin: float = ENERGY_MIN) -> np.ndarray:
    """per grid sample: ncc over the centred window; NaN if the window has < 70 % finite Orbbec samples or both sides are still
    (was imu_motion.window_ncc)"""
    tN = len(ao)
    tHalf = int(winS * GRID_HZ / 2)
    tOut = np.full(tN, np.nan)
    tFinite = np.isfinite(ao[:, 0])
    for tK in range(tHalf, tN - tHalf):
        tSl = slice(tK - tHalf, tK + tHalf + 1)
        tM = tFinite[tSl]
        if tM.mean() < MIN_WINDOW_FINITE:
            continue
        tA, tB = ao[tSl][tM], ai[tSl][tM]
        tEa, tEb = np.sqrt((tA * tA).sum(1).mean()), np.sqrt((tB * tB).sum(1).mean())
        if max(tEa, tEb) < eMin:
            continue
        tOut[tK] = float((tA * tB).sum() / np.sqrt((tA * tA).sum() * (tB * tB).sum() + MIN_DENOMINATOR))
    return tOut


def mahony(ti: np.ndarray, gyro: np.ndarray, acc: np.ndarray, bias: np.ndarray, gains: dict = AHRS_GAINS) -> np.ndarray:
    """R_g_body (body -> gravity world, +Z up, yaw arbitrary) at every IMU sample; the accelerometer corrects roll / pitch only
    (same filter as scripts/handumi_canonical_track.mahony, bias given)"""
    tT = (ti - ti[0]) / NS_PER_S
    tN = len(tT)
    tGRest = float(np.median(np.linalg.norm(acc, axis=1)))
    tA0 = acc[:AHRS_INIT_SAMPLES].mean(0)
    tA0 /= np.linalg.norm(tA0)
    tRm = R.align_vectors([[0, 0, 1.0]], [tA0])[0].as_matrix()
    tRs = np.empty((tN, 3, 3))
    tInteg = np.zeros(3)
    for tK in range(tN):
        tRs[tK] = tRm
        if tK == tN - 1:
            break
        tDt = min(max(float(tT[tK + 1] - tT[tK]), 0.0), AHRS_MAX_DT_S)
        tW = gyro[tK] - bias
        tAn = np.linalg.norm(acc[tK])
        if abs(tAn - tGRest) < gains["ACC_TOL"] and np.linalg.norm(tW) < gains["GYRO_MAX"]:
            tE = np.cross(acc[tK] / tAn, tRm[2])
            tInteg += gains["KI"] * tE * tDt
            tW = tW + gains["KP"] * tE + tInteg
        tRm = tRm @ R.from_rotvec(tW * tDt).as_matrix()
        if tK % AHRS_REORTHO_EVERY == 0:
            tU, _, tVt = np.linalg.svd(tRm)
            tRm = tU @ tVt
    return tRs


def ahrsBase(D: dict, g, anchorT, anchorR: R) -> R:
    """R_B_tcp on grid g: Mahony tilt + one yaw aligned to anchorR at anchorT (rig frame = TCP frame in arm_data) (was
    imu_motion.ahrs_base); caches the Mahony run in D["Rg"]"""
    if "Rg" not in D:
        D["Rg"] = mahony(D["imu_t"], D["gyro"], D["acc"], D.get("b_rest", D["b0"]))
    tSlerp = Slerp(D["imu_t"] / NS_PER_S, R.from_matrix(D["Rg"]))
    tTi = D["imu_t"]
    tRa = tSlerp(np.clip(anchorT, tTi[0], tTi[-1]) / NS_PER_S)
    tM = anchorR * tRa.inv()                                  # world_B <- gravity world
    tYaw = np.arctan2(tM.as_matrix()[1, 0], tM.as_matrix()[0, 0])
    tRz = R.from_euler("z", tYaw)
    return tRz * tSlerp(np.clip(g, tTi[0], tTi[-1]) / NS_PER_S)


def restBias(D: dict, windows: list) -> np.ndarray:
    """gyro bias from the rest periods between the take's demo windows (samples within 1.5 deg/s of the period median)
    (was imu_motion.rest_bias(D, take, FD); windows = arm_data.loadTrimWindows()[take]).  pose_smoother.prepGyro's
    |omega| < 3 deg/s stillness test misses the true bias when |bias| itself is ~3-4 deg/s (123747: 1.5 deg/s error)."""
    tTi, tGyro = D["imu_t"], D["gyro"]
    tStill = []
    for tI in range(len(windows) - 1):
        tM = (tTi > windows[tI].mW1) & (tTi < windows[tI + 1].mW0)
        tW = tGyro[tM]
        if tM.sum() < REST_MIN_SAMPLES:
            continue
        tSt = np.linalg.norm(tW - np.median(tW, 0), axis=1) < np.radians(REST_BIAS_BAND_DEG_S)
        if tSt.sum() > REST_MIN_STILL:
            tStill.append(tW[tSt])
    return np.median(np.concatenate(tStill), 0) if tStill else D["b0"]


def syncLag(D: dict, t: np.ndarray, c: np.ndarray, k: np.ndarray, anchorT, anchorR: R):
    """per-demo Orbbec<->IMU clock offset (ns): the IMU sampled at t + lag best matches the Orbbec body acceleration (low-passed
    ncc) -> (lag ns, ncc); lag 0 when the best ncc < LAG_MIN_NCC (was build_candidate_v0.sync_lag)"""
    tG = motionGrid(t[k][0], t[k][-1])
    tAo = trackAcc(t[k], c[k], tG)
    tM = np.isfinite(tAo[:, 0])
    if tM.sum() < LAG_MIN_VALID:
        return 0, 0.0
    tBest = (0, LAG_NO_NCC)
    for tLag in range(*LAG_RANGE_MS):
        tGg = tG + int(tLag * NS_PER_MS)
        tRt = ahrsBase(D, tGg, anchorT + int(tLag * NS_PER_MS), anchorR)
        tF = np.stack([np.interp(tGg, D["imu_t"], D["acc"][:, tJ]) for tJ in range(3)], 1)
        tAi = tRt.apply(tF)
        tAi[:, 2] -= np.median(np.linalg.norm(tF, axis=1))
        tAi = gaussian_filter1d(tAi, SMOOTH_SIGMA_S * GRID_HZ, axis=0, mode="nearest")
        tA, tB = tAo[tM], tAi[tM]
        tCc = float((tA * tB).sum() / np.sqrt((tA * tA).sum() * (tB * tB).sum() + MIN_DENOMINATOR))
        if tCc > tBest[1]:
            tBest = (tLag, tCc)
    return (int(tBest[0] * NS_PER_MS), tBest[1]) if tBest[1] >= LAG_MIN_NCC else (0, tBest[1])
