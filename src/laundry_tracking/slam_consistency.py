"""Marker-free SLAM<->IMU self-consistency for UMI wrist MASt3R trajectories (was _imu_consistency/core.py, 10-23).

Reads (never writes) other stages' data: raw MASt3R per-window segments (works while windows are still landing), the arm record of
fgo.arm_data.loadArm (IMU, tags), the demo windows, and the pooled hand-eye FUSION_WORK/out/RX_pooled_<cam>.npy when present.
Output of evaluateArm: one row per 50 ms of every processed demo window (rotation residuals vs the gyro over several horizons,
continuity, scale jumps, overlap with re-tracked segments, IMU acceleration plausibility, held-out tag validation) + the per-arm
calibration (extra SLAM time offset by |omega| cross-correlation, rotation-only hand-eye jointly with a constant gyro bias).
All times on the orbbec clock (arm_data convention: SLAM t = wrist t - 20 ms, IMU = host + calibrated offset).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from . import workdirs
from .constants import MM_PER_M, NS_PER_MS, NS_PER_S
from .fgo import arm_data
from .fgo import pose_smoother

HORIZONS_S = (0.1, 0.4, 0.8, 1.6)            # rotation-residual windows around every evaluated time
HARNESS_HORIZONS_S = (0.4, 1.6)              # ... also evaluated with the harness (tag) hand-eye
SLAM_OK_STATES = ("INIT", "TRACKING")
MIN_SEGMENT_FRAMES = 10
MIN_QUAT_NORM = 1e-3
OMEGA_LAGS_MS = np.arange(-200, 201, 2)      # searched extra SLAM time offsets
OMEGA_SMOOTH_S = 0.02                        # gyro |omega| smoothing before the correlation
OMEGA_MAX_STEP_S = 0.1                       # SLAM steps longer than this give no angular rate
OMEGA_MIN_DT_S = 1e-3
OMEGA_CLIP_PERCENTILE = 99.5                 # SLAM angular-rate outliers above this percentile are ignored
OMEGA_NO_CORR = -9                           # start value of the best correlation
SLAM_FPS = 30                                # MASt3R frame rate (frames per horizon)
HANDEYE_HORIZON_S = 0.4
HANDEYE_STRIDE = 3
HANDEYE_PAIR_SLACK_S = 0.1                   # a pair may span at most horizon + this (no frame drop inside)
HANDEYE_MIN_ROT_DEG = 5.0                    # pairs rotating less carry no hand-eye information
HANDEYE_ROUNDS = 6
CALIB_ROUNDS = 8
CALIB_MIN_ROT_DEG = 2.0
INLIER_FLOOR_DEG = 1.0                       # inlier gate = max(3 x median residual, this)
INLIER_MEDIAN_FACTOR = 3
BIAS_INLIER_PERCENTILE = 50
LOST_END_TRIM_S = 0.2                        # a segment that ended in tracking loss: its last 0.2 s are not trusted
JUMP_STEP_FACTOR = 8                         # pose jump = a step > 8 x the segment median step ...
JUMP_NEIGHBOUR_FACTOR = 3                    # ... and > 3 x both neighbouring steps
FRAME_GAP_S = 0.1
MIN_SIM3 = 1e-4
ACC_MIN_SEGMENT_S = 3.0                      # IMU accelerometer plausibility only on segments longer than this
ACC_GRID_HZ = 30
ACC_SMOOTH_S = 0.1
ACC_EDGE_S = 0.3                             # smoothing edge cut at both ends
ACC_MIN_SAMPLES = 30
OVERLAP_MIN_S = 1.2                          # overlap scale needs this much common span
OVERLAP_WINDOW_S = 0.8
OVERLAP_STEP_S = 0.1
EVAL_STEP_S = 0.05
MIN_WINDOW_COVERAGE = 0.95                   # demo windows whose MASt3R windows are not landed yet are skipped
JUMP_NEAR_S = 0.1
CHUNK_S = 1.6                                # forward action-chunk window [t, t + 1.6 s] and its halves
CONTINUITY_NEAR_S = 0.25
OVERLAP_FLOOR_SPEED_FRACTION = 0.3           # relative overlap error floor = 0.3 x typical speed x 0.8 s
ACC_NEAR_S = 0.8
ACC_MIN_NEAR = 10
TAG_FRAME_TOL_NS = 25e6
VAL_PAIR_TOL_NS = 40e6
VAL_EXCLUDE_S = 3.0                          # held-out scale: tag pairs at least 3 s away from the evaluated chunk
VAL_EXCLUDE_NEAR_S = 1.0
VAL_MIN_PAIRS = 8
VAL_MIN_PAIRS_NEAR = 5
MIN_DENOMINATOR = 1e-12
CHUNK_NS = int(CHUNK_S * NS_PER_S)           # was K32


# ---------------------------------------------------------------- MASt3R raw segments (per window dirs, works while windows are landing)
slamTag = arm_data.slamTag


def loadRawSegments(take: str, cam: str):
    """raw MASt3R segments of every landed window -> (segs, available windows [(t0, t1, w)]) (was core.load_segs_raw)"""
    tTag = slamTag(take, cam)
    tManifest = json.loads((arm_data.M3_RUNS / tTag / "segs/manifest.json").read_text())
    tFrameTime = {tF["i"]: tF["t_ns"] for tF in json.loads((arm_data.M3_FRAMES / tTag / "frames.json").read_text())["frames"]}
    tSegs, tAvail = [], []
    for tWin in tManifest["windows"]:
        tDir = arm_data.M3_RUNS / tTag / "windows" / tWin["sub"] / "segs"
        tCsvs = sorted(tDir.glob("*.csv")) if tDir.exists() else []
        if not tCsvs:
            continue
        tAvail.append((tWin["t0_ns"] - int(arm_data.SLAM_DT_NS), tWin["t1_ns"] - int(arm_data.SLAM_DT_NS), tWin["w"]))
        for tCsv in tCsvs:
            tS0 = int(tCsv.stem.split("_")[0]) + int(tWin["i0"])
            tDf = pd.read_csv(tCsv)
            if len(tDf) == 0:
                continue
            tBad = tDf["is_lost"].astype(str).str.lower().eq("true").to_numpy() | ~tDf["state"].isin(list(SLAM_OK_STATES)).to_numpy()
            tCut = int(np.argmax(tBad)) if tBad.any() else len(tDf)
            tLostEnd = bool(tBad.any())
            tDf = tDf.iloc[:tCut]
            tQ = tDf[["q_x", "q_y", "q_z", "q_w"]].to_numpy(float)
            tP = tDf[["x", "y", "z"]].to_numpy(float)
            tGi = tS0 + tDf["frame_idx"].to_numpy(int)
            tOk = np.isfinite(tQ).all(1) & np.isfinite(tP).all(1) & (np.linalg.norm(tQ, axis=1) > MIN_QUAT_NORM) & np.isin(tGi,
                    list(tFrameTime.keys()))
            if tOk.sum() < MIN_SEGMENT_FRAMES:
                continue
            tT = np.array([tFrameTime[tI] for tI in tGi[tOk]], np.int64) - int(arm_data.SLAM_DT_NS)
            tOrder = np.argsort(tT, kind="stable")
            tT, tU = np.unique(tT[tOrder], return_index=True)
            tOrder = tOrder[tU]
            tSegs.append(dict(t=tT, p=tP[tOk][tOrder], R=R.from_quat(tQ[tOk][tOrder]),
                              sim3=tDf["sim3_scale"].to_numpy(float)[tOk][tOrder],
                              kf=tDf["is_keyframe"].astype(str).str.lower().eq("true").to_numpy()[tOk][tOrder], lost_end=tLostEnd,
                              gi0=tS0, win=tWin["w"]))
    return tSegs, tAvail


# ---------------------------------------------------------------- IMU helpers
def gyroRelative(D: dict, ta, tb) -> R:
    """bias-corrected gyro rotation from ta to tb (TCP frame), arrays (was core.gyro_rel)"""
    return pose_smoother.gyroAt(D, ta).inv() * pose_smoother.gyroAt(D, tb)


def omegaMagCorr(D: dict, segs: list, lagsMs=OMEGA_LAGS_MS):
    """extra SLAM time offset (ns, added to SLAM t) maximising corr(|omega_slam|, |omega_gyro|) -> (offset ns, corr, corr per lag)
    (was core.omega_mag_corr)"""
    tTi = D["imu_t"]
    tGm = np.linalg.norm(D["gyro"] - D.get("b0", 0), axis=1)
    tDtImu = np.median(np.diff(tTi)) / NS_PER_S
    tGms = gaussian_filter1d(tGm, max(1, OMEGA_SMOOTH_S / tDtImu))
    tTimes, tRates = [], []
    for tSeg in segs:
        if len(tSeg["t"]) < 5:
            continue
        tDt = np.diff(tSeg["t"]) / NS_PER_S
        tOk = tDt < OMEGA_MAX_STEP_S
        tW = (tSeg["R"][:-1].inv() * tSeg["R"][1:]).magnitude() / np.maximum(tDt, OMEGA_MIN_DT_S)
        tTimes.append(((tSeg["t"][:-1] + tSeg["t"][1:]) // 2)[tOk])
        tRates.append(tW[tOk])
    tTimes = np.concatenate(tTimes)
    tRates = np.concatenate(tRates)
    tM = tRates < np.percentile(tRates, OMEGA_CLIP_PERCENTILE)
    tBest = (OMEGA_NO_CORR, 0, None)
    tCorrs = []
    for tLag in lagsMs:
        tG = np.interp(tTimes[tM] + tLag * NS_PER_MS, tTi, tGms)
        tC = np.corrcoef(tG, tRates[tM])[0, 1]
        tCorrs.append(tC)
        if tC > tBest[0]:
            tBest = (tC, tLag)
    return int(tBest[1] * NS_PER_MS), float(tBest[0]), np.array(tCorrs)


def kabschRotation(a: np.ndarray, b: np.ndarray, w=None) -> R:
    """RX minimising sum w |a - RX b|^2 (was core.kabsch_rot)"""
    w = np.ones(len(a)) if w is None else w
    tH = (b * w[:, None]).T @ a
    tU, _, tVt = np.linalg.svd(tH)
    tD = np.sign(np.linalg.det(tVt.T @ tU.T))
    return R.from_matrix(tVt.T @ np.diag([1, 1, tD]) @ tU.T)


def handEyeRotation(D: dict, segs: list, horizon: float = HANDEYE_HORIZON_S, stride: int = HANDEYE_STRIDE):
    """rotation-only hand-eye from SLAM-vs-gyro: rotvec(cam rel) = RX . rotvec(gyro rel)  (fgo convention M = RX^-1 A RX = B)
    -> (RX, n inliers, median residual deg, (A, B, TA, TB)) (was core.handeye_rot)"""
    tA, tTa, tTb = [], [], []
    for tSeg in segs:
        tT = tSeg["t"]
        tK = int(round(horizon * SLAM_FPS))
        if len(tT) <= tK:
            continue
        tI = np.arange(0, len(tT) - tK, stride)
        tJ = tI + tK
        tOk = (tT[tJ] - tT[tI]) < (horizon + HANDEYE_PAIR_SLACK_S) * NS_PER_S
        tI, tJ = tI[tOk], tJ[tOk]
        if not len(tI):
            continue
        tA.append((tSeg["R"][tI].inv() * tSeg["R"][tJ]).as_rotvec())
        tTa.append(tT[tI])
        tTb.append(tT[tJ])
    tA = np.concatenate(tA)
    tTa = np.concatenate(tTa)
    tTb = np.concatenate(tTb)
    tB = gyroRelative(D, tTa, tTb).as_rotvec()
    tM = (np.linalg.norm(tB, axis=1) > np.radians(HANDEYE_MIN_ROT_DEG))
    tKeep = tM.copy()
    for _ in range(HANDEYE_ROUNDS):
        tRX = kabschRotation(tA[tKeep], tB[tKeep])
        tRes = np.linalg.norm(tA - tRX.apply(tB), axis=1)
        tThr = max(INLIER_MEDIAN_FACTOR * np.median(tRes[tM]), np.radians(INLIER_FLOOR_DEG))
        tKeep = tM & (tRes < tThr)
    return tRX, int(tKeep.sum()), float(np.degrees(np.median(tRes[tKeep]))), (tA, tB, tTa, tTb)


def refineBias(D: dict, RX: R, pairs) -> np.ndarray:
    """constant gyro bias correction (TCP frame) so that the gyro matches the SLAM rotation rate; robust median (was core.refine_bias)"""
    tA, tB, tTa, tTb = pairs
    tDt = (tTb - tTa) / NS_PER_S
    tBm = RX.inv().apply(tA)                 # SLAM rotation expressed in the TCP frame
    tDb = (tB - tBm) / tDt[:, None]
    tRes = np.linalg.norm(tB - tBm, axis=1)
    tM = tRes < np.percentile(tRes, BIAS_INLIER_PERCENTILE)
    return np.median(tDb[tM], 0)


def integrateGyro(D: dict, bias: np.ndarray) -> None:
    """re-integrate Q / Qs with a new total bias (was core.integrate_q)"""
    tTi = D["imu_t"]
    tDt = np.diff(tTi) / NS_PER_S
    tInc = R.from_rotvec((D["gyro"][:-1] - bias) * tDt[:, None])
    tQ = np.empty((len(tTi), 4))
    tCur = R.identity()
    tQ[0] = tCur.as_quat()
    for tK in range(len(tInc)):
        tCur = tCur * tInc[tK]
        tQ[tK + 1] = tCur.as_quat()
    D["Q"] = R.from_quat(tQ)
    D["Qs"] = Slerp(tTi / NS_PER_S, D["Q"])
    D["bias_total"] = bias


def calibrate(D: dict, segs: list, RXFixed: R | None = None) -> dict:
    """per arm: extra SLAM time offset, rotation-only hand-eye (joint with a constant gyro bias).  Mutates segs t and D gyro / Q."""
    pose_smoother.prepGyro(D)
    tOff, tCorr, tCorrs = omegaMagCorr(D, segs)
    for tSeg in segs:
        tSeg["t"] = tSeg["t"] + tOff
    tRXe, _, _, tPairs = handEyeRotation(D, segs)
    tA, tB, tTa, tTb = tPairs
    tDt = (tTb - tTa) / NS_PER_S
    tDb = np.zeros(3)
    tMb = np.linalg.norm(tB, axis=1) > np.radians(CALIB_MIN_ROT_DEG)
    for _ in range(CALIB_ROUNDS):
        tBc = tB - tDb * tDt[:, None]
        tRes = np.linalg.norm(tA - tRXe.apply(tBc), axis=1)
        tThr = max(INLIER_MEDIAN_FACTOR * np.median(tRes[tMb]), np.radians(INLIER_FLOOR_DEG))
        tKeep = tMb & (tRes < tThr)
        tRXe = kabschRotation(tA[tKeep], tBc[tKeep]) if RXFixed is None else RXFixed
        tDb = np.median(((tB - tRXe.inv().apply(tA)) / tDt[:, None])[tKeep], 0)
    tResDeg = {}
    for tName, tX in (("est", tRXe), ("harness", D["RX"])):
        tResDeg[tName] = float(np.degrees(np.median(np.linalg.norm(tA - tX.apply(tB - tDb * tDt[:, None]), axis=1))))
    integrateGyro(D, D["b0"] + tDb)
    tRXa = handEyeRotation(D, segs)[0]
    return dict(RX_arm_only=tRXa, he_arm_vs_used_deg=float(np.degrees((tRXa.inv() * tRXe).magnitude())),
                offset_ms=tOff / NS_PER_MS, corr=tCorr, corr0=float(tCorrs[len(tCorrs) // 2]), RX_est=tRXe, RX_harness=D["RX"],
                he_diff_deg=float(np.degrees((tRXe.inv() * D["RX"]).magnitude())), dbias_dps=np.degrees(tDb).tolist(),
                b0_dps=np.degrees(D["b0"]).tolist(), res04_est_deg=tResDeg["est"], res04_harness_deg=tResDeg["harness"],
                n_pairs=int(tKeep.sum()))


# ---------------------------------------------------------------- per-segment precompute
def prepSegment(D: dict, s: dict, RX: R) -> dict:
    """adds to segment s: Slerp, trusted span, pose jumps, gaps, continuity z-scores, sim3 scale steps, IMU acceleration
    plausibility (was core.prep_seg)"""
    tT = s["t"]
    s["Rs"] = Slerp(tT / NS_PER_S, s["R"])
    s["t0"] = int(tT[0])
    s["t1"] = int(tT[-1]) - (int(LOST_END_TRIM_S * NS_PER_S) if s["lost_end"] else 0)
    tSteps = np.linalg.norm(np.diff(s["p"], axis=0), axis=1)
    tDtt = np.diff(tT) / NS_PER_S
    tMedStep = np.median(tSteps) + 1e-9
    s["v_med"] = float(np.median(tSteps / np.maximum(tDtt, OMEGA_MIN_DT_S)))
    # pose jump: a single-frame step far above the segment's typical step and its neighbours
    tNeighbour = np.maximum(np.r_[tSteps[1:], tSteps[-1]], np.r_[tSteps[0], tSteps[:-1]])
    tJump = (tSteps > JUMP_STEP_FACTOR * tMedStep) & (tSteps > JUMP_NEIGHBOUR_FACTOR * tNeighbour)
    s["jump_t"] = tT[1:][tJump]
    s["gap_t"] = tT[1:][tDtt > FRAME_GAP_S]
    # continuity: |second difference| normalised by the segment median
    if len(tT) > 3:
        tA2 = np.linalg.norm(s["p"][2:] - 2 * s["p"][1:-1] + s["p"][:-2], axis=1)
        s["acc_z"] = tA2 / (np.median(tA2) + 1e-9)
        s["acc_zt"] = tT[1:-1]
    else:
        s["acc_z"] = np.zeros(0)
        s["acc_zt"] = np.zeros(0, np.int64)
    tLogSim3 = np.log(np.clip(s["sim3"], MIN_SIM3, None))
    s["dls"] = np.abs(np.diff(tLogSim3))
    s["dls_t"] = tT[1:]
    # IMU accelerometer plausibility (soft): SLAM camera acceleration vs IMU linear acceleration in the SLAM world
    s["acc_ok"] = False
    if (tT[-1] - tT[0]) > ACC_MIN_SEGMENT_S * NS_PER_S:
        tG = np.arange(tT[0], tT[-1], int(NS_PER_S / ACC_GRID_HZ))
        tP = np.stack([np.interp(tG, tT, s["p"][:, tAx]) for tAx in range(3)], 1)
        tSig = ACC_SMOOTH_S * ACC_GRID_HZ
        tPs = gaussian_filter1d(tP, tSig, axis=0, mode="nearest")
        tAccSlam = np.gradient(np.gradient(tPs, 1 / ACC_GRID_HZ, axis=0), 1 / ACC_GRID_HZ, axis=0)
        tTi = D["imu_t"]
        tDti = np.median(np.diff(tTi)) / NS_PER_S
        tFa = gaussian_filter1d(D["acc"], ACC_SMOOTH_S / tDti, axis=0, mode="nearest")
        tF = np.stack([np.interp(tG, tTi, tFa[:, tAx]) for tAx in range(3)], 1)
        tRw = s["Rs"](tG / NS_PER_S) * RX
        tFw = tRw.apply(tF)
        tAccImu = tFw - tFw.mean(0)
        tEdge = int(ACC_EDGE_S * ACC_GRID_HZ)                        # gravity in the segment world = mean specific force
        tG, tAccSlam, tAccImu = tG[tEdge:-tEdge], tAccSlam[tEdge:-tEdge], tAccImu[tEdge:-tEdge]
        if len(tG) > ACC_MIN_SAMPLES:
            s["acc_g"] = tG
            s["aS"] = tAccSlam
            s["aI"] = tAccImu
            s["s_seg"] = float((tAccSlam * tAccImu).sum() / max((tAccSlam * tAccSlam).sum(), MIN_DENOMINATOR))
            s["acc_ok"] = True
    return s


def overlapScale(sp: dict, so: dict, RX: R, cache: dict, key):
    """relative scale of segment so vs sp over their common span (NaN if < 1.2 s), cached by key (was core.overlap_k)"""
    if key in cache:
        return cache[key]
    tA, tB = max(sp["t0"], so["t0"]), min(sp["t1"], so["t1"])
    tK = np.nan
    if tB - tA > OVERLAP_MIN_S * NS_PER_S:
        tTa = np.arange(tA, tB - int(OVERLAP_WINDOW_S * NS_PER_S), int(OVERLAP_STEP_S * NS_PER_S))
        tTb = tTa + int(OVERLAP_WINDOW_S * NS_PER_S)
        tDp = cameraDisplacement(sp, tTa, tTb)
        tDo = cameraDisplacement(so, tTa, tTb)
        tDen = (tDo * tDo).sum()
        if tDen > 0:
            tK = float((tDp * tDo).sum() / tDen)
    cache[key] = tK
    return tK


def segmentPosition(s: dict, t) -> np.ndarray:
    """segment camera centre at times t (was core.pos)"""
    return np.stack([np.interp(t, s["t"], s["p"][:, tAx]) for tAx in range(3)], 1)


def cameraDisplacement(s: dict, ta, tb) -> np.ndarray:
    """camera displacement ta->tb in the camera frame at ta (segment units) (was core.disp_cam)"""
    return s["Rs"](ta / NS_PER_S).inv().apply(segmentPosition(s, tb) - segmentPosition(s, ta))


def windowInSegment(s: dict, tc, h: float):
    """an h-second window centred at tc, shifted to lie inside the segment's trusted span; None if the segment is shorter
    (was core.win_in_seg)"""
    tA, tB = tc - h / 2 * NS_PER_S, tc + h / 2 * NS_PER_S
    if s["t1"] - s["t0"] < h * NS_PER_S:
        return None
    if tA < s["t0"]:
        tA, tB = s["t0"], s["t0"] + h * NS_PER_S
    if tB > s["t1"]:
        tA, tB = s["t1"] - h * NS_PER_S, s["t1"]
    return int(tA), int(tB)


def rotationResidual(D: dict, s: dict, ta, tb, RX: R, withMag: bool = False):
    """|SLAM relative rotation (TCP frame) vs gyro| over [ta, tb] in degrees (+ gyro rotation magnitude) (was core.rot_res)"""
    tA = s["Rs"](np.array([ta, tb]) / NS_PER_S)
    tM = RX.inv() * (tA[0].inv() * tA[1]) * RX
    tB = gyroRelative(D, np.array([ta]), np.array([tb]))[0]
    tE = float(np.degrees((tM.inv() * tB).magnitude()))
    return (tE, float(np.degrees(tB.magnitude()))) if withMag else tE


def evaluateArm(take: str, arm: str, demos=None):
    """-> (DataFrame of per-50 ms rows, calibration, D, segments) (was core.evaluate_arm)"""
    D = arm_data.loadArm(take, arm, withSlam=False)
    tSegs, tAvail = loadRawSegments(take, D["cam"])
    tPooled = workdirs.pooledHandEyePath(D["cam"])
    tCal = calibrate(D, tSegs, R.from_quat(np.load(tPooled)) if tPooled.exists() else None)
    tRX = tCal["RX_est"]
    for tSeg in tSegs:
        prepSegment(D, tSeg, tRX)
    D["tX_val"] = arm_data.meanHandEye1022(D["cam"])[1]
    validationPrep(D, tSegs, tRX)
    tRows = []
    tCache = {}
    for tWin in arm_data.loadTrimWindows()[take]:
        if demos is not None and tWin.mDemo not in demos:
            continue
        tTimes = np.arange(tWin.mW0, tWin.mW1, int(EVAL_STEP_S * NS_PER_S))
        tCov = np.zeros(len(tTimes), bool)
        for tA0, tA1, _ in tAvail:
            tCov |= (tTimes >= tA0) & (tTimes <= tA1)
        if tCov.mean() < MIN_WINDOW_COVERAGE:
            continue                                         # window not processed yet
        for tT in tTimes:
            tRow = dict(take=take, arm=arm, demo=tWin.mDemo, t=int(tT))
            tCand = [tI for tI, tSeg in enumerate(tSegs) if tSeg["t0"] <= tT <= tSeg["t1"]]
            tRow["n_cover"] = len(tCand)
            if not tCand:
                tRow["reason_track"] = "no_segment"
                tRows.append(tRow)
                continue
            tPi = max(tCand, key=lambda i: min(tT - tSegs[i]["t0"], tSegs[i]["t1"] - tT))
            tSeg = tSegs[tPi]
            tRow["seg"] = tPi
            tRow["jump"] = bool(np.any(np.abs(tSeg["jump_t"] - tT) < JUMP_NEAR_S * NS_PER_S)) or \
                bool(np.any(np.abs(tSeg["gap_t"] - tT) < JUMP_NEAR_S * NS_PER_S))
            tRow["seg_len_s"] = (tSeg["t1"] - tSeg["t0"]) / NS_PER_S
            tRow["edge_s"] = min(tT - tSeg["t0"], tSeg["t1"] - tT) / NS_PER_S
            for tH in HORIZONS_S:
                tWinSeg = windowInSegment(tSeg, tT, tH)
                tRow[f"e{tH}"], tRow[f"g{tH}"] = rotationResidual(D, tSeg, tWinSeg[0], tWinSeg[1], tRX,
                                                                  True) if tWinSeg else (np.nan, np.nan)
                if tH in HARNESS_HORIZONS_S:
                    tRow[f"eH{tH}"] = rotationResidual(D, tSeg, tWinSeg[0], tWinSeg[1], tCal["RX_harness"]) if tWinSeg else np.nan
            if tT + CHUNK_S * NS_PER_S <= tSeg["t1"]:                 # forward action-chunk window [t, t + 1.6 s] and its halves
                tHalf = int(CHUNK_S / 2 * NS_PER_S)
                tRow["f1.6"], tRow["fg1.6"] = rotationResidual(D, tSeg, tT, tT + int(CHUNK_S * NS_PER_S), tRX, True)
                tRow["f0.8a"], tRow["fg0.8a"] = rotationResidual(D, tSeg, tT, tT + tHalf, tRX, True)
                tRow["f0.8b"], tRow["fg0.8b"] = rotationResidual(D, tSeg, tT + tHalf, tT + int(CHUNK_S * NS_PER_S), tRX, True)
            tM = np.abs(tSeg["acc_zt"] - tT) <= CONTINUITY_NEAR_S * NS_PER_S
            tRow["cont_z"] = float(tSeg["acc_z"][tM].max()) if tM.any() else np.nan
            tM = np.abs(tSeg["dls_t"] - tT) <= CONTINUITY_NEAR_S * NS_PER_S
            tRow["dlogsim3"] = float(tSeg["dls"][tM].max()) if tM.any() else np.nan
            # overlap (independent re-tracking of the same frames by another segment): relative displacement discrepancy over 0.8 s
            tOv = []
            for tJ in tCand:
                if tJ == tPi:
                    continue
                tOther = tSegs[tJ]
                tA, tB = max(tSeg["t0"], tOther["t0"]), min(tSeg["t1"], tOther["t1"])
                if tB - tA < OVERLAP_WINDOW_S * NS_PER_S:
                    continue
                tK = overlapScale(tSeg, tOther, tRX, tCache, (tPi, tJ))
                if not np.isfinite(tK):
                    continue
                tTa = int(np.clip(tT - OVERLAP_WINDOW_S / 2 * NS_PER_S, tA, tB - OVERLAP_WINDOW_S * NS_PER_S))
                tTb = tTa + int(OVERLAP_WINDOW_S * NS_PER_S)
                tDp = cameraDisplacement(tSeg, np.array([tTa]), np.array([tTb]))[0]
                tDo = cameraDisplacement(tOther, np.array([tTa]), np.array([tTb]))[0]
                tFloor = OVERLAP_FLOOR_SPEED_FRACTION * tSeg["v_med"] * OVERLAP_WINDOW_S
                tOv.append(np.linalg.norm(tDp - tK * tDo) / max(np.linalg.norm(tDp), tFloor, 1e-9))
            tRow["ov_rel"] = float(np.min(tOv)) if tOv else np.nan
            if tSeg["acc_ok"]:
                tM = np.abs(tSeg["acc_g"] - tT) <= ACC_NEAR_S * NS_PER_S
                if tM.sum() > ACC_MIN_NEAR:
                    tAccSlam, tAccImu = tSeg["aS"][tM], tSeg["aI"][tM]
                    tRow["acc_exc"] = float(np.sqrt((tAccImu ** 2).sum(1).mean()))
                    tRow["acc_cos"] = float((tAccSlam * tAccImu).sum() / np.sqrt((tAccSlam ** 2).sum() * (tAccImu ** 2).sum() + MIN_DENOMINATOR))
                    tSlope = (tAccSlam * tAccImu).sum() / max((tAccSlam * tAccSlam).sum(), MIN_DENOMINATOR)
                    tRow["acc_sratio"] = float(tSlope / tSeg["s_seg"]) if tSeg["s_seg"] > 0 else np.nan
            validationChunk(D, tSegs, tPi, tT, tRX, tRow)
            tRows.append(tRow)
    return pd.DataFrame(tRows), tCal, D, tSegs


# ---------------------------------------------------------------- marker side check ONLY (held-out validation; never used by the confidence)
def tagFrame(D: dict, t, tol: float = TAG_FRAME_TOL_NS) -> int:
    """index of the orbbec frame nearest t if it is within tol and has >= 2 tags (position and rotation), else -1 (was core.tag_frame)"""
    tI = int(np.clip(np.searchsorted(D["t"], t), 1, len(D["t"]) - 1))
    tI = tI if abs(D["t"][tI] - t) < abs(D["t"][tI - 1] - t) else tI - 1
    return tI if abs(D["t"][tI] - t) <= tol and D["ntags"][tI] >= 2 and D["n_tagrot"][tI] >= 2 else -1


def chunkXly(D: dict, s: dict, RX: R, ta, tb):
    """SLAM chunk in the TCP frame at ta: x = R_tcp^T dC (segment units), l = metric lever term, + the TCP relative rotation
    (was core.chunk_xly)"""
    ta, tb = np.clip([ta, tb], s["t"][0], s["t"][-1])
    tRa, tRb = s["Rs"](np.array([ta, tb]) / NS_PER_S)
    tRt = tRa * RX
    tX = tRt.inv().apply(segmentPosition(s, np.array([tb]))[0] - segmentPosition(s, np.array([ta]))[0])
    tL = tRt.inv().apply(tRb.apply(D["tX_val"]) - tRa.apply(D["tX_val"]))
    return tX, tL, RX.inv() * (tRa.inv() * tRb) * RX


def validationPrep(D: dict, segs: list, RX: R) -> None:
    """per segment: tag-referenced 0.8 s pairs for the held-out scale -> s["vp"] (was core.val_prep)"""
    tOk = (D["ntags"] >= 2) & (D["n_tagrot"] >= 2)
    tT = D["t"]
    for tSeg in segs:
        tIdx = np.where(tOk & (tT >= tSeg["t0"]) & (tT <= tSeg["t1"] - OVERLAP_WINDOW_S * NS_PER_S))[0]
        tPairs = []
        for tI in tIdx[::2]:
            tJ = tagFrame(D, tT[tI] + OVERLAP_WINDOW_S * NS_PER_S, VAL_PAIR_TOL_NS)
            if tJ < 0 or tT[tJ] > tSeg["t1"]:
                continue
            tX, tL, _ = chunkXly(D, tSeg, RX, tT[tI], tT[tJ])
            tY = R.from_matrix(D["R_tag"][tI]).inv().apply(D["p_tag"][tJ] - D["p_tag"][tI])
            tPairs.append((tT[tI], tT[tJ], tX, tL, tY))
        tSeg["vp"] = tPairs


def _heldOutScale(pairs: list) -> float:
    tX = np.array([tP[2] for tP in pairs])
    tY = np.array([tP[4] - tP[3] for tP in pairs])
    return float((tX * tY).sum() / max((tX * tX).sum(), MIN_DENOMINATOR))


def validationChunk(D: dict, segs: list, pi: int, t, RX: R, r: dict) -> None:
    """adds the held-out tag validation of the 1.6 s chunk at t (segment pi) to row r (was core.val_chunk)"""
    tSeg = segs[pi]
    tTb = t + CHUNK_NS
    if tTb > tSeg["t1"]:
        return
    tI, tJ = tagFrame(D, t), tagFrame(D, tTb)
    if tI < 0 or tJ < 0:
        return
    tPairs = [tP for tP in tSeg["vp"] if tP[1] < t - VAL_EXCLUDE_S * NS_PER_S or tP[0] > tTb + VAL_EXCLUDE_S * NS_PER_S]
    tX, tL, tM = chunkXly(D, tSeg, RX, D["t"][tI], D["t"][tJ])
    tRi = R.from_matrix(D["R_tag"][tI])
    tY = tRi.inv().apply(D["p_tag"][tJ] - D["p_tag"][tI])
    tRref = tRi.inv() * R.from_matrix(D["R_tag"][tJ])
    r["v_disp_mm"] = float(np.linalg.norm(tY) * MM_PER_M)
    r["v_rot_deg"] = float(np.degrees((tM.inv() * tRref).magnitude()))
    r["v_rot_gyro_deg"] = float(np.degrees((gyroRelative(D, np.array([D["t"][tI]]),
                                                         np.array([D["t"][tJ]]))[0].inv() * tRref).magnitude()))
    tXx = float(tX @ tX)
    if tSeg.get("acc_ok") and tSeg["s_seg"] > 0:
        r["v_trans_imuscale_mm"] = float(np.linalg.norm(tSeg["s_seg"] * tX + tL - tY) * MM_PER_M)   # diagnostic only
    if tXx > 0:
        tSo = float(tX @ (tY - tL)) / tXx
        r["v_trans_oracle_mm"] = float(np.linalg.norm(tSo * tX + tL - tY) * MM_PER_M)
    tPairsNear = [tP for tP in tSeg["vp"] if tP[1] < t - VAL_EXCLUDE_NEAR_S * NS_PER_S or tP[0] > tTb + VAL_EXCLUDE_NEAR_S * NS_PER_S]
    if len(tPairsNear) >= VAL_MIN_PAIRS_NEAR:
        tSh = _heldOutScale(tPairsNear)
        if tSh > 0:
            r["v_trans_m1_mm"] = float(np.linalg.norm(tSh * tX + tL - tY) * MM_PER_M)
    if len(tPairs) >= VAL_MIN_PAIRS:
        tSh = _heldOutScale(tPairs)
        if tSh > 0:
            r["v_scale_n"] = len(tPairs)
            r["v_trans_mm"] = float(np.linalg.norm(tSh * tX + tL - tY) * MM_PER_M)
