"""UMI pose smoother (method c) + baselines (a) current fused, (b) endpoint-anchored SLAM bridge (was _fgo_1023/tools/fgo_core.py,
10-23).  Input: the per-arm record D of arm_data.loadArm; output: rotation / position on a 20 Hz node grid per trimmed demo window.

Two-stage factor graph:
  stage 1 ROTATION  nodes R_n = R0_n Exp(d_n), constant gyro bias db.  Factors: gyro relative rotation (strong, L2-ish), raw AprilTag
                    PnP rotation (soft, 1 tag weaker, Cauchy), MASt3R relative rotation over ~0.5 s (soft, Cauchy).
  stage 2 POSITION  nodes P_n, MASt3R scale per segment chunk (random walk between chunks).  Factors: tag position (soft, Cauchy,
                    1 tag weaker), MASt3R TCP displacement between consecutive wrist frames = s_chunk R_B_cam(t_i) dC_i + lever term
                    (Cauchy) -> all segments act on ONE trajectory (no segment jumps), weak acceleration prior (sigma large so fast
                    grasps are not blunted).  Linear in (P, s) given R -> damped Gauss-Newton / IRLS with sparse normal equations;
                    posterior position std from the inverse information matrix.
All times on the orbbec clock (SLAM -20 ms, IMU host + calibrated offset as tag_rig_fuse).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import least_squares
from scipy.sparse.linalg import splu
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from ..constants import MM_PER_M, NS_PER_MS, NS_PER_S

NODE_DT_S = 0.05                   # 20 Hz node grid
# Smoother options (string keys: callers override single entries, e.g. smooth(..., o=dict(slam_rot=False))).
SMOOTHER_OPTIONS = dict(sig_gyro_deg=0.15, sig_tagrot2_deg=2.0, sig_tagrot1_deg=6.0, sig_slamrot_deg=1.5, slam_rot=True,
                        sig_tag2=0.004, sig_tag1=0.008, sig_slam=0.003, slam_pos=True, sig_acc_prior=30.0, chunk_s=2.0,
                        sig_scale_rw=0.05, sig_rot_rw_deg=2.0, imu_acc=False, sig_acc=2.0, cauchy=3.0, tag_lag_ms=0.0,
                        max_nfev=200, gn_iters=15)
STILL_GYRO_DEG_S = 3.0             # |omega| below this = a still IMU sample (initial bias)
MIN_STILL_SAMPLES = 200            # fewer still samples -> zero initial bias
MIN_SLAM_IN_WINDOW = 3             # a SLAM segment needs this many frames inside the window to be used
SLAM_ROT_PAIR_FRAMES = 15          # relative-rotation factor between frames 15 apart (~0.5 s at 30 fps)
SLAM_ROT_PAIR_STRIDE = 5
BIAS_PRIOR_DEG = 0.5               # prior sigma of the bias correction db (per axis, as rotation over 1 s)
ROT_MAX_NFEV = 30                  # 10-23: the rotation stage is capped here; SMOOTHER_OPTIONS["max_nfev"] is not read by it
SLAM_STEP_MAX_S = 0.1              # consecutive SLAM frames further apart are not a displacement factor
MIN_SLAM_STEPS = 3
SCALE_INIT_MIN, SCALE_INIT_MAX = 0.03, 2.0      # tag-path / SLAM-path ratio clip for the initial chunk scale
SCALE_INIT_MIN_TAG_PATH_M = 0.02   # shorter tag path: initial scale = SCALE_INIT_FALLBACK
SCALE_INIT_FALLBACK = 0.2
MIN_TAG_FRAMES_SCALE_INIT = 3
LOG_SCALE_PRIOR_SIGMA = 1.5        # weak prior of log s around its initial value
GN_DAMPING = (1e-5, 1e-3, 1e-1)    # Levenberg damping ladder of one Gauss-Newton step
GN_STEP_CLIP_LOG_SCALE = 0.5
GN_STEP_CLIP_ROT = 0.3
GN_STOP_STEP_M = 1e-5
LOG_SCALE_RANGE = (np.log(0.01), np.log(3.0))
COV_REGULARIZER = 1e-9
SLAM_COVERED_WEIGHT = 0.5          # a SLAM step counts as covering its nodes when its robust weight is above this
METHOD_B_MIN_ANCHORS = 4
METHOD_B_MAX_FRAME_GAP_S = 0.3
METHOD_B_EDGE_FRAMES = 3           # anchor residual averaged over the 3 anchors next to the gap on each side
CLEAN_MIN_ANCHORS = 8
CLEAN_SCALE_RANGE = (0.05, 3.0)
CLEAN_MAX_DEV_M = 0.15             # chain_v1 gate: max deviation from the tag baseline
MIN_ORACLE_FRAMES = 4


def prepGyro(D: dict) -> dict:
    """cumulative gyro rotation Q(t) on the IMU samples (TCP frame), bias = median over still samples (was fgo_core.prep)"""
    if "Q" in D:
        return D
    tTi = D["imu_t"]
    tGyro = D["gyro"]
    tStill = np.linalg.norm(tGyro, axis=1) < np.radians(STILL_GYRO_DEG_S)
    D["b0"] = np.median(tGyro[tStill], 0) if tStill.sum() > MIN_STILL_SAMPLES else np.zeros(3)
    tDt = np.diff(tTi) / NS_PER_S
    tInc = R.from_rotvec((tGyro[:-1] - D["b0"]) * tDt[:, None])
    tQ = np.empty((len(tTi), 4))
    tCur = R.identity()
    tQ[0] = tCur.as_quat()
    for tK in range(len(tInc)):
        tCur = tCur * tInc[tK]
        tQ[tK + 1] = tCur.as_quat()
    D["Q"] = R.from_quat(tQ)
    D["Qs"] = Slerp(tTi / NS_PER_S, D["Q"])
    return D


def gyroAt(D: dict, t) -> R:
    """integrated gyro rotation at times t (clipped to the IMU span) (was fgo_core.Qat)"""
    tTi = D["imu_t"]
    return D["Qs"](np.clip(t, tTi[0], tTi[-1]) / NS_PER_S)


def nodeOf(tn: np.ndarray, t) -> np.ndarray:
    """nearest node index of times t"""
    return np.clip(np.round((t - tn[0]) / (NODE_DT_S * NS_PER_S)).astype(int), 0, len(tn) - 1)


def rotationAt(D: dict, tn: np.ndarray, Rn: R, t) -> R:
    """node rotation propagated to t by the gyro (was fgo_core.rot_at)"""
    tNode = nodeOf(tn, t)
    return Rn[tNode] * (gyroAt(D, tn[tNode]).inv() * gyroAt(D, t))


def slamInWindow(D: dict, w0: int, w1: int) -> list[dict]:
    """the part of every SLAM segment inside [w0, w1] (was fgo_core.slam_in)"""
    tOut = []
    for tK, tSeg in enumerate(D["segs"]):
        tM = (tSeg["t"] >= w0) & (tSeg["t"] <= w1)
        if tM.sum() >= MIN_SLAM_IN_WINDOW:
            tOut.append(dict(k=tK, t=tSeg["t"][tM], p=tSeg["p"][tM], R=tSeg["R"][tM], t0=tSeg["t"][0]))
    return tOut


def hideMask(t, hide) -> np.ndarray:
    """True inside the hidden window hide = (h0, h1) (None = nothing hidden)"""
    if hide is None:
        return np.zeros(len(t), bool)
    return (t >= hide[0]) & (t <= hide[1])


# ------------------------------------------------------------------ stage 1: rotation
def solveRotation(D: dict, tn: np.ndarray, hide, o: dict):
    """-> (Rn on tn, info dict) (was fgo_core.solve_rot)"""
    tN = len(tn)
    tR0 = Slerp(D["t"] / NS_PER_S, R.from_matrix(D["R_fused"]))(np.clip(tn, D["t"][0], D["t"][-1]) / NS_PER_S)
    tG = gyroAt(D, tn[:-1]).inv() * gyroAt(D, tn[1:])
    tDtn = np.diff(tn) / NS_PER_S
    # tag rotation frames in the window, not hidden
    tT = D["t"]
    tM = (tT >= tn[0]) & (tT <= tn[-1]) & (D["n_tagrot"] > 0) & ~hideMask(tT, hide)
    tTf = tT[tM]
    tRt = R.from_matrix(D["R_tag"][tM])
    tNf = nodeOf(tn, tTf)
    tCf = gyroAt(D, tn[tNf]).inv() * gyroAt(D, tTf)
    tSf = np.radians(np.where(D["n_tagrot"][tM] >= 2, o["sig_tagrot2_deg"], o["sig_tagrot1_deg"]))
    # SLAM relative rotation pairs (~0.5 s apart)
    tPi, tPj, tMrel = [], [], []
    if o["slam_rot"] and "RX" in D:
        for tSeg in slamInWindow(D, tn[0], tn[-1]):
            tK = SLAM_ROT_PAIR_FRAMES
            if len(tSeg["t"]) <= tK:
                continue
            tA = np.arange(0, len(tSeg["t"]) - tK, SLAM_ROT_PAIR_STRIDE)
            tB = tA + tK
            tPi.append(tSeg["t"][tA])
            tPj.append(tSeg["t"][tB])
            tMrel.append(D["RX"].inv() * (tSeg["R"][tA].inv() * tSeg["R"][tB]) * D["RX"])
    if tPi:
        tPi = np.concatenate(tPi)
        tPj = np.concatenate(tPj)
        tMrel = R.concatenate(tMrel)
        tNi = nodeOf(tn, tPi)
        tNj = nodeOf(tn, tPj)
        tCi = gyroAt(D, tn[tNi]).inv() * gyroAt(D, tPi)
        tCj = gyroAt(D, tn[tNj]).inv() * gyroAt(D, tPj)
    else:
        tNi = tNj = np.zeros(0, int)
    tSigGyro = np.radians(o["sig_gyro_deg"])
    tSigSlam = np.radians(o["sig_slamrot_deg"])

    def _residual(x):
        tD = x[:3 * tN].reshape(tN, 3)
        tDb = x[3 * tN:]
        tRn = tR0 * R.from_rotvec(tD)
        tE1 = ((tG.inv() * (tRn[:-1].inv() * tRn[1:])).as_rotvec() + tDb[None] * tDtn[:, None]) / tSigGyro
        tE2 = (tRt.inv() * (tRn[tNf] * tCf)).as_rotvec() / tSf[:, None] if len(tNf) else np.zeros((0, 3))
        tE3 = (tMrel.inv() * ((tRn[tNi] * tCi).inv() * (tRn[tNj] * tCj))).as_rotvec() / tSigSlam if len(tNi) else np.zeros((0, 3))
        return np.concatenate([tE1.ravel(), tE2.ravel(), tE3.ravel(), x[3 * tN:] / np.radians(BIAS_PRIOR_DEG)])
    tRows, tCols = [], []

    def _addBlock(r0, nodes):
        for tNode in nodes:
            for tAx in range(3):
                tRows.extend([r0 + tAx] * 3)
                tCols.extend([3 * tNode + 0, 3 * tNode + 1, 3 * tNode + 2])
    tR = 0
    for tNode in range(tN - 1):
        _addBlock(tR, [tNode, tNode + 1])
        for tAx in range(3):
            tRows.extend([tR + tAx] * 3)
            tCols.extend([3 * tN, 3 * tN + 1, 3 * tN + 2])
        tR += 3
    for tNode in tNf:
        _addBlock(tR, [tNode])
        tR += 3
    for tA, tB in zip(tNi, tNj):
        _addBlock(tR, [tA, tB] if tA != tB else [tA])
        tR += 3
    for tAx in range(3):
        tRows.append(tR + tAx)
        tCols.append(3 * tN + tAx)
    tR += 3
    tSparsity = sp.coo_matrix((np.ones(len(tRows)), (tRows, tCols)), shape=(tR, 3 * tN + 3)).tocsr()
    tSparsity.data[:] = 1
    tSol = least_squares(_residual, np.zeros(3 * tN + 3), jac_sparsity=tSparsity, loss="cauchy", f_scale=o["cauchy"],
                         method="trf", x_scale=1.0, max_nfev=ROT_MAX_NFEV, tr_solver="lsmr")
    tRn = tR0 * R.from_rotvec(tSol.x[:3 * tN].reshape(tN, 3))
    return tRn, dict(rot_cost=float(tSol.cost), n_tagrot=int(len(tNf)), n_slamrot=int(len(tNi)))


# ------------------------------------------------------------------ stage 2: position (Gauss-Newton, per-chunk Sim3 of each SLAM segment)
def _skew(v: np.ndarray) -> np.ndarray:
    tZ = np.zeros(len(v))
    return np.stack([np.stack([tZ, -v[:, 2], v[:, 1]], 1), np.stack([v[:, 2], tZ, -v[:, 0]], 1), np.stack([-v[:, 1], v[:, 0],
            tZ], 1)], 1)


def solvePosition(D: dict, tn: np.ndarray, Rn: R, hide, o: dict, wantCov: bool = True):
    """x = [P_n (3N) | per SLAM chunk c: R_c (left increment), log s_c].  SLAM factor: P(t_j) - P(t_i) - lever = s_c R_c (pc_j - pc_i)
    (R_c, s_c free per chunk_s chunk, random walk between chunks of one segment -> SLAM translation SHAPE, one global trajectory,
    no jumps).  Damped Gauss-Newton with sparse direct solves; first third plain L2, then Cauchy IRLS (tags + SLAM).
    -> (P (N,3), posterior std (N,) or None, info) (was fgo_core.solve_pos)"""
    tN = len(tn)
    tDtn = NODE_DT_S * NS_PER_S

    def _nodeWeights(t):
        tF = np.clip((t - tn[0]) / tDtn, 0, tN - 1 - 1e-9)
        tI = np.floor(tF).astype(int)
        return tI, tF - tI
    tT = D["t"] + int(o["tag_lag_ms"] * NS_PER_MS)
    tMask = (tT >= tn[0]) & (tT <= tn[-1]) & (D["ntags"] > 0) & ~hideMask(D["t"], hide)
    tTagI, tTagA = _nodeWeights(tT[tMask])
    tPt = D["p_tag"][tMask]
    tSigT = np.where(D["ntags"][tMask] >= 2, o["sig_tag2"], o["sig_tag1"])
    tTt = tT[tMask]
    if len(tTt) >= 2:
        tP0 = np.stack([np.interp(tn, tTt, tPt[:, tAx]) for tAx in range(3)], 1)
    else:
        tP0 = np.repeat(D["p_tag"][np.argmin(np.abs(D["t"] - tn[0]))][None], tN, 0)
    tI0, tA0, tI1, tA1, tU, tWl, tCh, tRw = [], [], [], [], [], [], [], []
    tC = 0
    tRinit, tLs0 = [], []
    if o["slam_pos"] and "RX" in D:
        tRXi = D["RX"].inv()
        tX = D["tX"]
        for tSeg in slamInWindow(D, tn[0], tn[-1]):
            tTs = tSeg["t"]
            tOk = np.where(np.diff(tTs) < SLAM_STEP_MAX_S * NS_PER_S)[0]
            if len(tOk) < MIN_SLAM_STEPS:
                continue
            tRbc = rotationAt(D, tn, Rn, tTs) * tRXi
            tStep = tSeg["p"][1:] - tSeg["p"][:-1]
            tLever = tRbc[1:].apply(tX) - tRbc[:-1].apply(tX)
            tChunk = ((tTs[:-1] - tSeg["t0"]) // int(o["chunk_s"] * NS_PER_S)).astype(int)
            tUch = np.unique(tChunk[tOk])
            tCmap = {tCc: tC + tJ for tJ, tCc in enumerate(tUch)}
            for tCc in tUch:
                tKk = tOk[tChunk[tOk] == tCc]
                tRinit.append((tRbc[tKk] * tSeg["R"][tKk].inv()).mean().as_matrix())
                tM2 = tMask & (tT >= tTs[tKk[0]]) & (tT <= tTs[tKk[-1] + 1])
                tSl = np.linalg.norm(tStep[tKk], axis=1).sum()
                tTp = np.linalg.norm(np.diff(D["p_tag"][tM2], axis=0),
                                     axis=1).sum() if tM2.sum() > MIN_TAG_FRAMES_SCALE_INIT else 0
                tLs0.append(np.log(np.clip(tTp / tSl, SCALE_INIT_MIN, SCALE_INIT_MAX)) if (tSl > 0 and tTp > SCALE_INIT_MIN_TAG_PATH_M)
                            else np.log(SCALE_INIT_FALLBACK))
            tRw += [(tCmap[tA], tCmap[tB]) for tA, tB in zip(tUch[:-1], tUch[1:])]
            tSi0, tSa0 = _nodeWeights(tTs[:-1][tOk])
            tSi1, tSa1 = _nodeWeights(tTs[1:][tOk])
            tI0.append(tSi0)
            tA0.append(tSa0)
            tI1.append(tSi1)
            tA1.append(tSa1)
            tU.append(tStep[tOk])
            tWl.append(tLever[tOk])
            tCh.append(np.array([tCmap[tCc] for tCc in tChunk[tOk]]))
            tC += len(tUch)
    if tI0:
        tI0, tA0, tI1, tA1, tU, tWl, tCh = map(np.concatenate, (tI0, tA0, tI1, tA1, tU, tWl, tCh))
    else:
        tI0 = tI1 = tCh = np.zeros(0, int)
        tA0 = tA1 = np.zeros(0)
        tU = tWl = np.zeros((0, 3))
    tRw = np.array(tRw, int).reshape(-1, 2)
    tLs0 = np.array(tLs0)
    tRc = np.array(tRinit).reshape(-1, 3, 3)
    tLs = tLs0.copy()
    tP = tP0.copy()
    tNv = 3 * tN + 4 * tC
    tSq = o["sig_slam"]
    tSa = o["sig_acc_prior"] * NODE_DT_S * NODE_DT_S
    tSr = np.radians(o["sig_rot_rw_deg"])
    tSs = o["sig_scale_rw"]
    tOffR, tOffL = 3 * tN, 3 * tN + 3 * tC
    tAx3 = np.arange(3)
    # constant parts of the Jacobian (P columns)
    tNT, tNS, tNA = len(tTagI), len(tI0), max(tN - 2, 0)
    tRowT = 3 * np.arange(tNT)[:, None] + tAx3
    tRowS = 3 * tNT + 3 * np.arange(tNS)[:, None] + tAx3
    tRowA = 3 * (tNT + tNS) + 3 * np.arange(tNA)[:, None] + tAx3
    tRowRW = 3 * (tNT + tNS + tNA) + 3 * np.arange(len(tRw))[:, None] + tAx3
    tRowLW = 3 * (tNT + tNS + tNA + len(tRw)) + np.arange(len(tRw))
    tRowLP = tRowLW[-1] + 1 + np.arange(tC) if len(tRw) else 3 * (tNT + tNS + tNA) + np.arange(tC)
    tNr = 3 * (tNT + tNS + tNA + len(tRw)) + len(tRw) + tC

    def _jacobianResidual(P, Rc, ls):
        tJr, tJc, tJv = [], [], []

        def _add(r, c, v):
            tJr.append(np.broadcast_to(r, np.shape(v)).ravel())
            tJc.append(np.broadcast_to(c, np.shape(v)).ravel())
            tJv.append(np.ravel(v))
        tE1 = ((1 - tTagA)[:, None] * P[tTagI] + tTagA[:, None] * P[tTagI + 1] - tPt) / tSigT[:, None]
        _add(tRowT, 3 * tTagI[:, None] + tAx3, ((1 - tTagA) / tSigT)[:, None] * np.ones(3))
        _add(tRowT, 3 * (tTagI + 1)[:, None] + tAx3, (tTagA / tSigT)[:, None] * np.ones(3))
        if tNS:
            tRu = np.einsum("nij,nj->ni", Rc[tCh], tU)
            tSc = np.exp(ls[tCh])[:, None]
            tPj = (1 - tA1)[:, None] * P[tI1] + tA1[:, None] * P[tI1 + 1]
            tPi = (1 - tA0)[:, None] * P[tI0] + tA0[:, None] * P[tI0 + 1]
            tE2 = (tPj - tPi - tWl - tSc * tRu) / tSq
            for tIdx, tCoef in ((tI1, (1 - tA1)), (tI1 + 1, tA1), (tI0, -(1 - tA0)), (tI0 + 1, -tA0)):
                _add(tRowS, 3 * tIdx[:, None] + tAx3, (tCoef / tSq)[:, None] * np.ones(3))
            tJd = tSc[:, :, None] * _skew(tRu) / tSq                          # d e2 / d delta_c (left increment)
            _add(tRowS[:, :, None], (tOffR + 3 * tCh)[:, None, None] + tAx3[None, None, :], tJd)
            _add(tRowS, (tOffL + tCh)[:, None] * np.ones(3, int), -tSc * tRu / tSq)
        else:
            tE2 = np.zeros((0, 3))
        tE3 = (P[2:] - 2 * P[1:-1] + P[:-2]) / tSa
        tInner = np.arange(1, tN - 1)
        for tOff, tCoef in ((-1, 1.0), (0, -2.0), (1, 1.0)):
            _add(tRowA, 3 * (tInner + tOff)[:, None] + tAx3, np.full((tNA, 3), tCoef / tSa))
        if len(tRw):
            tE4 = R.from_matrix(np.einsum("nij,nkj->nik", Rc[tRw[:, 1]], Rc[tRw[:, 0]])).as_rotvec() / tSr
            _add(tRowRW, (tOffR + 3 * tRw[:, 1])[:, None] + tAx3, np.full((len(tRw), 3), 1 / tSr))
            _add(tRowRW, (tOffR + 3 * tRw[:, 0])[:, None] + tAx3, np.full((len(tRw), 3), -1 / tSr))
            tE5 = (ls[tRw[:, 1]] - ls[tRw[:, 0]]) / tSs
            _add(tRowLW, tOffL + tRw[:, 1], np.full(len(tRw), 1 / tSs))
            _add(tRowLW, tOffL + tRw[:, 0], np.full(len(tRw), -1 / tSs))
        else:
            tE4 = np.zeros((0, 3))
            tE5 = np.zeros(0)
        tE6 = (ls - tLs0) / LOG_SCALE_PRIOR_SIGMA
        _add(tRowLP, tOffL + np.arange(tC), np.full(tC, 1 / LOG_SCALE_PRIOR_SIGMA))
        tRes = np.concatenate([tE1.ravel(), tE2.ravel(), tE3.ravel(), tE4.ravel(), tE5, tE6])
        tJ = sp.csr_matrix((np.concatenate(tJv), (np.concatenate(tJr), np.concatenate(tJc))), shape=(tNr, tNv))
        return tRes, tJ
    tNRobust = 3 * (tNT + tNS)

    def _weights(r):
        tW = np.ones(tNr)
        tBlock = r[:tNRobust].reshape(-1, 3)
        tE = np.linalg.norm(tBlock, axis=1)
        tW[:tNRobust] = np.repeat(1.0 / (1.0 + (tE / o["cauchy"]) ** 2), 3)
        return tW
    tW = np.ones(tNr)
    for tIt in range(o["gn_iters"]):
        tRes, tJ = _jacobianResidual(tP, tRc, tLs)
        if tIt >= o["gn_iters"] // 3:
            tW = _weights(tRes)
        tH = (tJ.T @ sp.diags(tW) @ tJ).tocsc()
        tGrad = tJ.T @ (tW * tRes)
        tDx = None
        for tLm in GN_DAMPING:
            try:
                tDx = -splu((tH + sp.identity(tNv) * tLm).tocsc()).solve(tGrad)
                if np.all(np.isfinite(tDx)):
                    break
            except RuntimeError:
                tDx = None
        if tDx is None or not np.all(np.isfinite(tDx)):
            break
        tDx[tOffL:] = np.clip(tDx[tOffL:], -GN_STEP_CLIP_LOG_SCALE, GN_STEP_CLIP_LOG_SCALE)
        tDx[tOffR:tOffL] = np.clip(tDx[tOffR:tOffL], -GN_STEP_CLIP_ROT, GN_STEP_CLIP_ROT)
        tP = tP + tDx[:3 * tN].reshape(tN, 3)
        if tC:
            tRc = R.from_matrix(np.einsum("nij,njk->nik", R.from_rotvec(tDx[tOffR:tOffL].reshape(tC, 3)).as_matrix(),
                                          tRc)).as_matrix()
            tLs = np.clip(tLs + tDx[tOffL:], LOG_SCALE_RANGE[0], LOG_SCALE_RANGE[1])
        if np.abs(tDx[:3 * tN]).max() < GN_STOP_STEP_M:
            break
    tRes, tJ = _jacobianResidual(tP, tRc, tLs)
    tW = _weights(tRes)
    tStd = None
    if wantCov:
        tH = (tJ.T @ sp.diags(tW) @ tJ).toarray() + np.eye(tNv) * COV_REGULARIZER
        tCov = np.linalg.inv(tH)
        tStd = np.sqrt(np.maximum(np.diag(tCov)[:3 * tN].reshape(tN, 3).sum(1), 0))
    tNt3, tNs3 = 3 * tNT, 3 * tNS
    tInfo = dict(n_tag=int(tNT), n_slam=int(tNS), nchunk=int(tC), scales=np.exp(tLs),
                 tag_w_med=float(np.median(tW[:tNt3])) if tNt3 else 1.0,
                 slam_w_med=float(np.median(tW[tNt3:tNt3 + tNs3])) if tNs3 else 1.0, nfev=tIt + 1)
    tCovered = np.zeros(tN, bool)
    if tNs3:
        tStrong = tW[tNt3:tNt3 + tNs3].reshape(-1, 3).min(1) > SLAM_COVERED_WEIGHT
        tCovered[tI0[tStrong]] = True
        tCovered[np.minimum(tI1[tStrong] + 1, tN - 1)] = True
    tInfo["slam_cov"] = tCovered
    return tP, tStd, tInfo


def smooth(D: dict, w0: int, w1: int, hide=None, o: dict | None = None, wantCov: bool = True) -> dict:
    """both stages on [w0, w1] -> dict(tn, Rn, Pn, std, info)"""
    tOpt = dict(SMOOTHER_OPTIONS, **(o or {}))
    prepGyro(D)
    tn = np.arange(w0, w1 + 1, int(NODE_DT_S * NS_PER_S), dtype=np.int64)
    tRn, tInfo1 = solveRotation(D, tn, hide, tOpt)
    tPn, tStd, tInfo2 = solvePosition(D, tn, tRn, hide, tOpt, wantCov)
    return dict(tn=tn, Rn=tRn, Pn=tPn, std=tStd, info={**tInfo1, **tInfo2})


def evalAt(D: dict, S: dict, t):
    """(P, R, std) of a smooth() result at times t (was fgo_core.eval_at)"""
    tn = S["tn"]
    tP = np.stack([np.interp(t, tn, S["Pn"][:, tAx]) for tAx in range(3)], 1)
    tR = rotationAt(D, tn, S["Rn"], t)
    tSd = None if S["std"] is None else np.interp(t, tn, S["std"])
    return tP, tR, tSd


# ------------------------------------------------------------------ (a) current fused (no new info)
def methodA(D: dict, tEval, hide):
    """the current file: gated SLAM where slam_ok, else linear interpolation between the remaining fused_valid samples; rotation =
    stored R_B_tcp (was fgo_core.method_a)"""
    tT = D["t"]
    tHid = hideMask(tT, hide)
    tValid = D["fused_valid"] & ~(tHid & ~D["slam_ok"])
    tP = np.stack([np.interp(tEval, tT[tValid], D["p_fused"][tValid, tAx]) for tAx in range(3)], 1)
    tK = np.searchsorted(tT, tEval)
    tK = np.clip(tK, 0, len(tT) - 1)
    tSlam = D["slam_ok"][tK] & (tT[tK] == tEval)
    tP[tSlam] = D["p_fused"][tK[tSlam]]
    return tP, R.from_matrix(D["R_fused"][tK])


# ------------------------------------------------------------------ (b) endpoint-anchored SLAM bridge
def gyroBridge(D: dict, ta: int, tb: int, Ra: R, Rb: R, t) -> R:
    """rotation in a gap from the gyro, endpoint error distributed by SLERP (was fgo_core.gyro_bridge)"""
    prepGyro(D)
    tQa, tQb = gyroAt(D, np.array([ta]))[0], gyroAt(D, np.array([tb]))[0]
    tRg = Ra * (tQa.inv() * gyroAt(D, t))
    tRgb = Ra * (tQa.inv() * tQb)
    tErr = (tRgb.inv() * Rb).as_rotvec()
    tWgt = np.clip((t - ta) / max(tb - ta, 1), 0, 1)
    return tRg * R.from_rotvec(tWgt[:, None] * tErr[None])


def _segmentPositions(seg: dict, t) -> np.ndarray:
    return np.stack([np.interp(t, seg["t"], seg["p"][:, tAx]) for tAx in range(3)], 1)


def methodB(D: dict, tEval, hide, ctxS: float = 1.0, maxRes: float = 0.02):
    """hidden window [h0, h1]: anchors = tag frames outside.  Every raw SLAM segment spanning [h0 - ctx, h1 + ctx] with >= 4 anchors
    each side: rotation RA from the (fused) anchor rotations, scale + translation by LS, residual at both ends removed linearly.
    Best context residual wins; none (or residual > maxRes) -> linear fill.  Rotation = gyro bridge between the anchor rotations.
    -> (P, R, best residual or None) (was fgo_core.method_b)"""
    tT = D["t"]
    tH0, tH1 = hide
    tAnchor = (D["ntags"] > 0) & ~hideMask(tT, hide)
    tIa = np.where(tAnchor & (tT < tH0))[0]
    tIb = np.where(tAnchor & (tT > tH1))[0]
    tTa, tTb = tT[tIa[-1]], tT[tIb[0]]
    tPa, tPb = D["p_tag"][tIa[-1]], D["p_tag"][tIb[0]]
    tRf = R.from_matrix(D["R_fused"])
    tRr = gyroBridge(D, tTa, tTb, tRf[tIa[-1]], tRf[tIb[0]], tEval)
    tP = tPa + (tPb - tPa) * np.clip((tEval - tTa) / (tTb - tTa), 0, 1)[:, None]
    tBest = None
    if "RX" in D:
        tCa = tIa[tT[tIa] >= tTa - ctxS * NS_PER_S]
        tCb = tIb[tT[tIb] <= tTb + ctxS * NS_PER_S]
        for tSeg in D["segs"]:
            if tSeg["t"][0] > tT[tCa[0]] or tSeg["t"][-1] < tT[tCb[-1]] or len(tCa) < METHOD_B_MIN_ANCHORS or \
                    len(tCb) < METHOD_B_MIN_ANCHORS:
                continue
            tCtx = np.r_[tCa, tCb]
            tTc = tT[tCtx]
            if np.diff(tSeg["t"][(tSeg["t"] >= tTc[0]) & (tSeg["t"] <= tTc[-1])]).max(initial=0) > METHOD_B_MAX_FRAME_GAP_S * NS_PER_S:
                continue
            tPc = _segmentPositions(tSeg, tTc)
            tRc = Slerp(tSeg["t"] / NS_PER_S, tSeg["R"])(tTc / NS_PER_S)
            tRA = (tRf[tCtx] * D["RX"].inv() * tRc.inv()).mean()
            tY = D["p_tag"][tCtx] - (tRA * tRc).apply(D["tX"])
            tXx = tRA.apply(tPc)
            tAm = np.zeros((3 * len(tCtx), 4))
            tAm[:, 0] = tXx.ravel()
            tAm[0::3, 1] = 1
            tAm[1::3, 2] = 1
            tAm[2::3, 3] = 1
            tSol = np.linalg.lstsq(tAm, tY.ravel(), rcond=None)[0]
            tS, tTt = tSol[0], tSol[1:]
            if tS <= 0:
                continue
            tPred = tS * tXx + (tRA * tRc).apply(D["tX"]) + tTt
            tRes = tPred - D["p_tag"][tCtx]
            tRms = float(np.sqrt(np.mean(np.sum(tRes ** 2, 1))))
            if tBest is None or tRms < tBest[0]:
                tBest = (tRms, tSeg, tRA, tS, tTt, tRes, tCtx)
        if tBest is not None and tBest[0] < maxRes:
            tRms, tSeg, tRA, tS, tTt, tRes, tCtx = tBest
            tM = (tEval >= tSeg["t"][0]) & (tEval <= tSeg["t"][-1])
            tPe = _segmentPositions(tSeg, tEval)
            tRe = Slerp(tSeg["t"] / NS_PER_S, tSeg["R"])(np.clip(tEval, tSeg["t"][0], tSeg["t"][-1]) / NS_PER_S)
            tPred = tS * tRA.apply(tPe) + (tRA * tRe).apply(D["tX"]) + tTt
            tEa = tRes[np.isin(tCtx, tIa)][-METHOD_B_EDGE_FRAMES:].mean(0)
            tEb = tRes[np.isin(tCtx, tIb)][:METHOD_B_EDGE_FRAMES].mean(0)
            tWgt = np.clip((tEval - tTa) / (tTb - tTa), 0, 1)[:, None]
            tPb2 = tPred - ((1 - tWgt) * tEa + tWgt * tEb)
            tP[tM] = tPb2[tM]
    return tP, tRr, (tBest[0] if tBest is not None else None)


# ------------------------------------------------------------------ (a') current pipeline, leak-free: whole-segment Sim3 (hand-eye fixed)
# on the non-hidden tags, gated like chain_v1 (own-fit p90 <= 30 mm, max deviation from the tag baseline <= 0.15 m); else linear fill
def methodAClean(D: dict, tEval, hide, p90Mm: float = 30.0):
    """-> (P, R, whether a segment was used) (was fgo_core.method_a_clean)"""
    tT = D["t"]
    tAnchor = (D["ntags"] > 0) & ~hideMask(tT, hide)
    tP = np.stack([np.interp(tEval, tT[tAnchor], D["p_tag"][tAnchor, tAx]) for tAx in range(3)], 1)
    tRf = R.from_matrix(D["R_fused"])
    tK = np.clip(np.searchsorted(tT, tEval), 0, len(tT) - 1)
    tBest = None
    if "RX" in D:
        for tSeg in D["segs"]:
            tEv = (tEval >= tSeg["t"][0]) & (tEval <= tSeg["t"][-1])
            if not tEv.any():
                continue
            tCtx = np.where(tAnchor & (tT >= tSeg["t"][0]) & (tT <= tSeg["t"][-1]))[0]
            if len(tCtx) < CLEAN_MIN_ANCHORS:
                continue
            tPc = _segmentPositions(tSeg, tT[tCtx])
            tRc = Slerp(tSeg["t"] / NS_PER_S, tSeg["R"])(tT[tCtx] / NS_PER_S)
            tRA = (tRf[tCtx] * D["RX"].inv() * tRc.inv()).mean()
            tY = D["p_tag"][tCtx] - (tRA * tRc).apply(D["tX"])
            tXx = tRA.apply(tPc)
            tAm = np.zeros((3 * len(tCtx), 4))
            tAm[:, 0] = tXx.ravel()
            tAm[0::3, 1] = 1
            tAm[1::3, 2] = 1
            tAm[2::3, 3] = 1
            tSol = np.linalg.lstsq(tAm, tY.ravel(), rcond=None)[0]
            tS, tTt = tSol[0], tSol[1:]
            if not (CLEAN_SCALE_RANGE[0] < tS < CLEAN_SCALE_RANGE[1]):
                continue
            tRes = np.linalg.norm((tAm @ tSol - tY.ravel()).reshape(-1, 3), axis=1)
            if np.percentile(tRes, 90) * MM_PER_M > p90Mm:
                continue
            tTe = tEval[tEv]
            tPe = _segmentPositions(tSeg, tTe)
            tRe = Slerp(tSeg["t"] / NS_PER_S, tSeg["R"])(tTe / NS_PER_S)
            tPred = tS * tRA.apply(tPe) + (tRA * tRe).apply(D["tX"]) + tTt
            if np.linalg.norm(tPred - tP[tEv], axis=1).max() > CLEAN_MAX_DEV_M:
                continue
            if tBest is None or len(tCtx) > tBest[0]:
                tBest = (len(tCtx), tEv, tPred)
    if tBest is not None:
        tP[tBest[1]] = tBest[2]
    return tP, tRf[tK], tBest is not None


def oracleSlam(D: dict, tEval):
    """CHEATING upper bound: best single raw segment, Sim3 fitted directly on the evaluated (hidden) tag frames (was fgo_core.oracle_slam)"""
    tBest = None
    tY = D["p_tag"][np.searchsorted(D["t"], tEval)]
    for tSeg in D["segs"]:
        if tSeg["t"][0] > tEval[0] or tSeg["t"][-1] < tEval[-1] or len(tEval) < MIN_ORACLE_FRAMES:
            continue
        tPc = _segmentPositions(tSeg, tEval)
        tA = tPc - tPc.mean(0)
        tB = tY - tY.mean(0)
        tU, tSv, tVt = np.linalg.svd(tB.T @ tA / len(tA))
        tDg = np.diag([1, 1, np.sign(np.linalg.det(tU @ tVt))])
        tRm = tU @ tDg @ tVt
        tS = (tSv * np.diag(tDg)).sum() / tA.var(0).sum()
        tP = tS * tPc @ tRm.T + (tY.mean(0) - tS * tRm @ tPc.mean(0))
        tE = np.linalg.norm(tP - tY, axis=1)
        if tBest is None or tE.mean() < tBest[0]:
            tBest = (tE.mean(), tP)
    return None if tBest is None else tBest[1]
