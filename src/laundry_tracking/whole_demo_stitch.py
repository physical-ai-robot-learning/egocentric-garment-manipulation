"""Whole-demo stitch optimizer (v2 / v3) on EXISTING raw visual segments (was _imu_consistency/stitch/wdo.py, 10-23).

Inputs: the arm record (fgo.arm_data.loadArm), MASt3R raw segments (slam_consistency.loadRawSegments) or any generic visual
front-end npz (--visual), the pooled hand-eye FUSION_WORK/out/RX_pooled_<cam>.npy, the demo windows.
Output: FUSION_WORK/stitch/qa/<variant>/<take>_<arm>_dNN_h<off>.npz (t_ns, p_B, R_B_tcp on the orbbec frames + node solution).
Builds on fgo.pose_smoother.solveRotation for the gyro-led rotation stage.  Differences vs fgo_v0:
  * camera<->IMU: marker-free pooled hand-eye RX + per-arm camera time offset + gyro bias from SLAM<->gyro (fgo_v0 used the per-arm tag
    hand-eye, 150 deg / 34-39 deg wrong on 124754 R / 144321 L,R, which also rotated every SLAM displacement into the wrong direction)
  * position: per visual segment ONE Sim(3) (R_i fixed from the gyro-led rotation, s_i free; t_i eliminated by displacement factors), so
    each segment keeps its relative shape (fgo_v0: free rotation + scale random walk per 2 s chunk)          [v2]
    v3: scale knots every KNOT_S inside a segment, strong random-walk smoothness between knots (+ optional weak Cauchy accel factor)
  * segments cut at pose jumps / tracking loss; SLAM steps whose 0.4 s rotation disagrees with the gyro are down-weighted (x0.1)
  * smoothness = second difference (velocity-continuous, Hermite-like across gaps; no straight-line fill), weak where a visual or tag
    measurement touches the node, strong only on no-information nodes
  * tags: soft robust (Cauchy) absolute anchors only; hidden-tag QA removes them entirely (fit + init) in hidden blocks
Linear in (P, s) given the rotations -> sparse IRLS.
    python -m laundry_tracking.whole_demo_stitch TAKE ARM VARIANT(v2|v3on|v3off) [--offsets 0,1,2,-1] [--demos 0,4]
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from . import workdirs
from .constants import NS_PER_S
from .fgo import arm_data
from .fgo import pose_smoother
from .slam_consistency import calibrate, loadRawSegments, prepSegment

NODE_DT_S = 0.05
KNOT_S = 1.5
HIDE_BLOCK_S = 4.0                 # hidden-tag protocol: tags dropped in blocks of 4 s, block % 3 == offset
HIDE_MOD = 3
MIN_SEGMENT_FRAMES = 10
MIN_QUAT_NORM = 1e-3
JUMP_EXCLUDE_S = 0.05              # steps within 50 ms of a pose jump get no factor
SCALE_OUTLIER_FACTOR = 4           # a segment whose scale leaves [s0 / 4, 4 s0] is dropped and the demo re-solved
ROT_CHECK_HALF_FRAMES = 6          # rotation consistency over +-6 frames (0.4 s at 30 fps)
ROT_GOOD_BASE_DEG = 2.5            # step is 'good' when |SLAM - gyro| rotation <= 2.5 deg + 0.1 x gyro rotation
ROT_GOOD_SLOPE = 0.1
MIN_GOOD_FOR_ALIGN = 5
STEP_MAX_S = 0.1
MIN_TAGS_SCALE_INIT = 10
SCALE_INIT_LAG_FRAMES = 5
SCALE_INIT_FALLBACK = 0.1
ABS_PRIOR_OFF = 1e5                # abs_prior sigmas >= this disable the drift guard
IMU_SMOOTH_TAPS = 5
IRLS_ITERS = 8
IRLS_REGULARIZER = 1e-9
MIN_DENOMINATOR = 1e-12
ABS_PRIOR_ABLATION_M = 0.15        # --abs-prior: 0.15 m L2 pull toward the tag-only linear fill (REJECTED 10-23)
WINDOW_MARGIN_S = 2.0              # solve [w0 - 2 s, w1 + 2 s]
QUANTILES = (10, 50, 90)


@dataclass(frozen=True)
class StitchSigmas:
    """factor sigmas of the stitch (was the SIG dict); mAbsPrior >= ABS_PRIOR_OFF disables the drift guard"""
    mAbsPrior: float = 1e6
    mSlam: float = 0.003
    mTag2: float = 0.004
    mTag1: float = 0.008
    mAccInfo: float = 30.0
    mAccGap: float = 3.0
    mScalePrior: float = 0.5
    mKnotRw: float = 0.05
    mImuAcc: float = 2.0
    mCauchy: float = 3.0
    mRotBadW: float = 0.1


STITCH_SIGMAS = StitchSigmas()


def hiddenMask(t: np.ndarray, off: int, blockS: float = HIDE_BLOCK_S, mod: int = HIDE_MOD) -> np.ndarray:
    """hidden-tag protocol: True on the frames of the 4 s blocks with block % 3 == off (off < 0 = production, nothing hidden)
    (was wdo.hidden_mask)"""
    return np.zeros(len(t), bool) if off < 0 else ((((t - t[0]) // int(blockS * NS_PER_S)) % mod) == off)


def loadVisual(npz) -> list[dict]:
    """generic front-end trajectory (one camera) -> raw segments: t_ns (n,) wrist-camera clock (= m3_frames/<tag>/frames.json t_ns),
    p (n,3) camera centre in the segment world (any unit), q_xyzw (n,4) camera->world (OpenCV camera axes), segment_id (n,)
    (restart / new map = new id), optional conf (n,) (was wdo.load_visual)"""
    tZ = np.load(npz)
    tT = tZ["t_ns"].astype(np.int64) - int(arm_data.SLAM_DT_NS)
    tIds = tZ["segment_id"]
    tSegs = []
    tKeep = np.isfinite(tZ["p"]).all(1) & np.isfinite(tZ["q_xyzw"]).all(1) & (np.linalg.norm(tZ["q_xyzw"],
                                                                                             axis=1) > MIN_QUAT_NORM)
    for tK in np.unique(tIds):
        tM = tKeep & (tIds == tK)
        if tM.sum() < MIN_SEGMENT_FRAMES:
            continue
        tOrder = np.argsort(tT[tM], kind="stable")
        tTimes, tU = np.unique(tT[tM][tOrder], return_index=True)
        tOrder = tOrder[tU]
        # lost_end=True: the last 0.2 s of a segment are dropped, as for MASt3R
        tSegs.append(dict(t=tTimes, p=tZ["p"][tM][tOrder], R=R.from_quat(tZ["q_xyzw"][tM][tOrder]), sim3=np.ones(len(tTimes)),
                          kf=np.zeros(len(tTimes), bool), lost_end=True, gi0=int(tK), win=-1))
    return tSegs


def prepare(take: str, arm: str, visual=None) -> dict:
    """arm record with calibrated hand-eye and the usable visual segments (bad = steps touching a pose jump) (was wdo.prepare)"""
    D = arm_data.loadArm(take, arm, withSlam=False)
    if visual:
        tSegs = loadVisual(str(visual).format(take=take, arm=arm, cam=D["cam"]))
    else:
        tSegs, _ = loadRawSegments(take, D["cam"])
    tCal = calibrate(D, tSegs, R.from_quat(np.load(workdirs.pooledHandEyePath(D["cam"]))))
    D["RX"] = tCal["RX_est"]
    D["tX"] = arm_data.meanHandEye1022(D["cam"])[1]
    tOut = []
    for tSeg in tSegs:                                       # no splitting (one map = one Sim3); steps touching a pose jump get no factor
        prepSegment(D, tSeg, D["RX"])
        tBad = np.zeros(len(tSeg["t"]), bool)
        for tJump in tSeg["jump_t"]:
            tBad |= np.abs(tSeg["t"] - tJump) <= JUMP_EXCLUDE_S * NS_PER_S
        tM = tSeg["t"] <= tSeg["t1"]
        if tM.sum() >= MIN_SEGMENT_FRAMES:
            tOut.append(dict(t=tSeg["t"][tM], p=tSeg["p"][tM], R=tSeg["R"][tM], bad=tBad[tM], src=(tSeg["gi0"],
                                                                                                   int(tSeg["t"][0]))))
    D["segs"] = tOut
    D["cal"] = tCal
    return D


def maskTags(D: dict, hid: np.ndarray) -> dict:
    """copy of D without the tags of hidden frames; their initial rotation comes from the gyro, not the tag (was wdo.mask_tags)"""
    D2 = dict(D)
    D2["ntags"] = np.where(hid, 0, D["ntags"])
    D2["n_tagrot"] = np.where(hid, 0, D["n_tagrot"])
    tRf = R.from_matrix(D["R_fused"])
    tOk = ~hid
    if hid.any() and tOk.any():
        tIdx = np.where(tOk)[0]
        tK = tIdx[np.clip(np.searchsorted(tIdx, np.arange(len(hid))) - 1, 0, len(tIdx) - 1)]
        tRg = tRf[tK] * (pose_smoother.gyroAt(D, D["t"][tK]).inv() * pose_smoother.gyroAt(D, D["t"]))
        tM = tRf.as_matrix().copy()
        tM[hid] = tRg.as_matrix()[hid]
        D2["R_fused"] = tM
    return D2


def solveDemo(D: dict, w0: int, w1: int, variant: str, hid: np.ndarray, sigmas: StitchSigmas = STITCH_SIGMAS) -> dict:
    """one demo window; segments whose scale leaves [s0 / 4, 4 s0] are dropped and the window solved again (was wdo.solve_demo)"""
    tSol = _solve(D, w0, w1, variant, hid, set(), sigmas)
    tDrop = set()
    for (tSrc, _, _, tNk), tOff in zip(tSol["seginfo"], np.cumsum([0] + [tX[3] for tX in tSol["seginfo"]])[:-1]):
        tSc = tSol["scales"][tOff:tOff + tNk]
        if np.any(tSc < tSol["s0"] / SCALE_OUTLIER_FACTOR) or np.any(tSc > SCALE_OUTLIER_FACTOR * tSol["s0"]):
            tDrop.add(tSrc)
    if tDrop:
        tSol = _solve(D, w0, w1, variant, hid, tDrop, sigmas)
        tSol["dropped"] = len(tDrop)
    return tSol


def _solve(D: dict, w0: int, w1: int, variant: str, hid: np.ndarray, drop: set, sigmas: StitchSigmas) -> dict:
    D2 = maskTags(D, hid)
    tOpt = dict(pose_smoother.SMOOTHER_OPTIONS)
    tn = np.arange(w0, w1 + 1, int(NODE_DT_S * NS_PER_S), dtype=np.int64)
    tN = len(tn)
    tRn, _ = pose_smoother.solveRotation(D2, tn, None,
                                         tOpt)   # gyro-led rotation (TCP), soft tag + SLAM relative rotation factors
    tRXi = D["RX"].inv()
    tX = D["tX"]

    def _nodeWeights(t):
        tF = np.clip((t - tn[0]) / (NODE_DT_S * NS_PER_S), 0, tN - 1 - 1e-9)
        tI = np.floor(tF).astype(int)
        return tI, tF - tI
    # unknowns: P (3N) then scales
    tNs = 0
    tInfo = np.zeros(tN, bool)
    tSegInfo = []
    # tags
    tT = D2["t"]
    tM = (tT >= tn[0]) & (tT <= tn[-1]) & (D2["ntags"] > 0)
    tTagsUsed = tM.sum()
    # visual segments
    for tSeg in D["segs"]:
        tK = (tSeg["t"] >= tn[0]) & (tSeg["t"] <= tn[-1])
        if tK.sum() < MIN_SEGMENT_FRAMES or tSeg["src"] in drop:
            continue
        tTs, tPs, tRs = tSeg["t"][tK], tSeg["p"][tK], tSeg["R"][tK]
        tRbc = pose_smoother.rotationAt(D, tn, tRn, tTs) * tRXi
        # rotation consistency per step (0.4 s window), SLAM vs gyro-led rotation
        tH = ROT_CHECK_HALF_FRAMES
        tA = np.clip(np.arange(len(tTs)) - tH, 0, len(tTs) - 1)
        tB = np.clip(np.arange(len(tTs)) + tH, 0, len(tTs) - 1)
        tE = np.degrees(((tRs[tA].inv() * tRs[tB]).inv() * (tRbc[tA].inv() * tRbc[tB])).magnitude())
        tG = np.degrees((tRbc[tA].inv() * tRbc[tB]).magnitude())
        tGood = tE <= ROT_GOOD_BASE_DEG + ROT_GOOD_SLOPE * tG
        tRi = (tRbc[tGood] * tRs[tGood].inv()).mean() if tGood.sum() >= MIN_GOOD_FOR_ALIGN else (tRbc * tRs.inv()).mean()
        tU = tRi.apply(tPs[1:] - tPs[:-1])
        tLever = tRbc[1:].apply(tX) - tRbc[:-1].apply(tX)
        tW = np.where(tGood[1:] & tGood[:-1], 1.0, sigmas.mRotBadW)
        tBad = tSeg["bad"][tK]
        tOk = (np.diff(tTs) < STEP_MAX_S * NS_PER_S) & ~tBad[1:] & ~tBad[:-1]
        if variant == "v2":
            tKnots = np.array([tTs[0]])
            tKid = np.zeros(len(tTs) - 1, int)
            tKw = np.zeros(len(tTs) - 1)
        else:
            tNk = max(1, int(np.ceil((tTs[-1] - tTs[0]) / (KNOT_S * NS_PER_S)))) + 1
            tKnots = np.linspace(tTs[0], tTs[-1], tNk)
            tF = np.clip((tTs[:-1] - tTs[0]) / max(tTs[-1] - tTs[0], 1) * (tNk - 1), 0, tNk - 1 - 1e-9)
            tKid = np.floor(tF).astype(int)
            tKw = tF - tKid
        tSegInfo.append(dict(src=tSeg["src"], t0=int(tTs[0]), t1=int(tTs[-1]), s_off=tNs, nk=len(tKnots), ts=tTs, u=tU,
                             lev=tLever, w=tW, ok=tOk, kid=tKid, kw=tKw))
        tNs += len(tKnots)
    tNv = 3 * tN + tNs
    # initial scale prior s0: per segment tag LS (non-hidden tags), median over segments
    tScaleEst = []
    for tSi in tSegInfo:
        tMm = (tT >= tSi["t0"]) & (tT <= tSi["t1"]) & (D2["ntags"] > 0)
        if tMm.sum() >= MIN_TAGS_SCALE_INIT:
            tIdx = np.where(tMm)[0]
            tIa, tIb = tIdx[:-SCALE_INIT_LAG_FRAMES], tIdx[SCALE_INIT_LAG_FRAMES:]
            tDpt = D["p_tag"][tIb] - D["p_tag"][tIa]
            tSeg = next(tS for tS in D["segs"] if tS["src"] == tSi["src"])

            def _segPos(tt, seg=tSeg):
                return np.stack([np.interp(tt, seg["t"], seg["p"][:, tQ]) for tQ in range(3)], 1)
            tRbc = pose_smoother.rotationAt(D, tn, tRn, tT[tIa]) * tRXi
            tRi = (tRbc * Slerp(tSeg["t"] / NS_PER_S, tSeg["R"])(np.clip(tT[tIa], tSeg["t"][0],
                                                                         tSeg["t"][-1]) / NS_PER_S).inv()).mean()
            tXs = tRi.apply(_segPos(tT[tIb]) - _segPos(tT[tIa]))
            tSs = (tXs * tDpt).sum() / max((tXs * tXs).sum(), MIN_DENOMINATOR)
            if tSs > 0:
                tScaleEst.append(tSs)
    tS0 = float(np.median(tScaleEst)) if tScaleEst else SCALE_INIT_FALLBACK
    # assemble (as triplets with explicit column indices)
    tRowsR, tColsC, tVals, tTargets, tSigmas, tRobust = [], [], [], [], [], []
    tNr = 0

    def _addFactor(colmat, b, s, robust):
        """one 3-vector factor: colmat = list of (kind, column base / scalar column, coefficient or per-axis vector)"""
        nonlocal tNr
        for tAx in range(3):
            for tKind, tCol, tV in colmat:
                tRowsR.append(tNr + tAx)
                tColsC.append(3 * tCol + tAx if tKind == "P" else tCol)
                tVals.append(tV if np.ndim(tV) == 0 else tV[tAx])
        tTargets.extend(b)
        tSigmas.extend([s] * 3)
        tRobust.extend([robust] * 3)
        tNr += 3
    tTagI, tTagA = _nodeWeights(tT[tM])
    for tI, tA, tP, tNt in zip(tTagI, tTagA, D["p_tag"][tM], D2["ntags"][tM]):
        _addFactor([("P", tI, 1 - tA), ("P", tI + 1, tA)], tP, sigmas.mTag2 if tNt >= 2 else sigmas.mTag1, True)
        tInfo[tI] = tInfo[min(tI + 1, tN - 1)] = True
    for tSi in tSegInfo:
        tTs = tSi["ts"]
        tI0, tA0 = _nodeWeights(tTs[:-1])
        tI1, tA1 = _nodeWeights(tTs[1:])
        for tK in np.where(tSi["ok"])[0]:
            tCs = [("P", tI1[tK], 1 - tA1[tK]), ("P", tI1[tK] + 1, tA1[tK]), ("P", tI0[tK], -(1 - tA0[tK])), ("P", tI0[tK] + 1,
                    -tA0[tK])]
            tSc = 3 * tN + tSi["s_off"] + tSi["kid"][tK]
            if tSi["nk"] == 1:
                tCs.append(("S", tSc, -tSi["u"][tK]))
            else:
                tCs += [("S", tSc, -(1 - tSi["kw"][tK]) * tSi["u"][tK]), ("S", tSc + 1, -tSi["kw"][tK] * tSi["u"][tK])]
            _addFactor(tCs, tSi["lev"][tK], sigmas.mSlam / np.sqrt(tSi["w"][tK]), True)
            tInfo[[tI0[tK], tI1[tK]]] = True
    # weak absolute drift guard: every node toward the tag-only linear fill of the NON-hidden tags (held flat beyond the last tag)
    tTagIdx = np.where((D2["ntags"] > 0))[0]
    if len(tTagIdx) > 1 and sigmas.mAbsPrior < ABS_PRIOR_OFF:
        tBase = np.stack([np.interp(tn, tT[tTagIdx], D["p_tag"][tTagIdx, tQ]) for tQ in range(3)], 1)
        for tNode in range(tN):
            _addFactor([("P", tNode, 1.0)], tBase[tNode], sigmas.mAbsPrior, False)
    # scalar rows: scale prior + knot random walk
    tScalarRows = []
    for tSi in tSegInfo:
        for tQ in range(tSi["nk"]):
            tScalarRows.append(([(3 * tN + tSi["s_off"] + tQ, 1.0)], tS0, sigmas.mScalePrior * tS0))
        for tQ in range(tSi["nk"] - 1):
            tScalarRows.append(([(3 * tN + tSi["s_off"] + tQ + 1, 1.0), (3 * tN + tSi["s_off"] + tQ, -1.0)], 0.0,
                                sigmas.mKnotRw * tS0))
    # smoothness (second difference), weak on info nodes, strong on no-information nodes
    tAccTarget = np.zeros((tN, 3))
    if variant == "v3on":
        tTi = D["imu_t"]
        tF = np.stack([np.interp(tn, tTi, D["acc"][:, tAx]) for tAx in range(3)],
                      1)      # TCP frame specific force (20 Hz avg-ish)
        tK5 = np.ones(IMU_SMOOTH_TAPS) / IMU_SMOOTH_TAPS
        tF = np.stack([np.convolve(tF[:, tAx], tK5, mode="same") for tAx in range(3)], 1)
        tFb = tRn.apply(tF)
        tAccTarget = tFb - tFb.mean(0)
    for tNode in range(1, tN - 1):
        tSa = (sigmas.mAccInfo if (tInfo[tNode - 1] or tInfo[tNode] or tInfo[tNode + 1]) else sigmas.mAccGap) * NODE_DT_S * NODE_DT_S
        _addFactor([("P", tNode - 1, 1.0), ("P", tNode, -2.0), ("P", tNode + 1, 1.0)], np.zeros(3), tSa, False)
        if variant == "v3on":
            _addFactor([("P", tNode - 1, 1.0), ("P", tNode, -2.0), ("P", tNode + 1, 1.0)],
                       tAccTarget[tNode] * NODE_DT_S * NODE_DT_S, sigmas.mImuAcc * NODE_DT_S * NODE_DT_S, True)
    for tCols, tB, tS in tScalarRows:
        for tCol, tV in tCols:
            tRowsR.append(tNr)
            tColsC.append(tCol)
            tVals.append(tV)
        tTargets.append(tB)
        tSigmas.append(tS)
        tRobust.append(False)
        tNr += 1
    tAm = sp.csr_matrix((tVals, (tRowsR, tColsC)), shape=(tNr, tNv))
    tBv = np.array(tTargets)
    tSv = np.array(tSigmas)
    tRb = np.array(tRobust)
    tWv = 1.0 / tSv ** 2
    tXsol = None
    for _ in range(IRLS_ITERS):
        tH = (tAm.T @ sp.diags(tWv) @ tAm).tocsc() + sp.identity(tNv) * IRLS_REGULARIZER
        tXsol = splu(tH).solve(tAm.T @ (tWv * tBv))
        tE = (tAm @ tXsol - tBv) / tSv
        tCw = np.where(tRb, 1.0 / (1.0 + (np.abs(tE) / sigmas.mCauchy) ** 2), 1.0)
        tWv = tCw / tSv ** 2
    tP = tXsol[:3 * tN].reshape(tN, 3)
    tScales = tXsol[3 * tN:]
    return dict(tn=tn, Rn=tRn, P=tP, scales=tScales, s0=tS0,
                seginfo=[(tSi["src"], tSi["t0"], tSi["t1"], tSi["nk"]) for tSi in tSegInfo], n_tags=int(tTagsUsed))


def _buildParser() -> argparse.ArgumentParser:
    tAp = argparse.ArgumentParser()
    tAp.add_argument("take")
    tAp.add_argument("arm")
    tAp.add_argument("variant", choices=["v2", "v3on", "v3off"])
    tAp.add_argument("--offsets", default="0,1,2,-1")
    tAp.add_argument("--demos", default="")
    tAp.add_argument("--visual", default="", help="generic frontend npz (pattern with {take} {arm} {cam}); default = raw MASt3R")
    tAp.add_argument("--mode", default="B", choices=["A", "B", "auto"], help="B = fragmented input (one Sim3 per segment, default); A = continuous/global "
                     "input: all samples treated as ONE world (one Sim3 for the demo, shape preserved, tags soft + light smoothing only); auto = A if <= 2 segments")
    tAp.add_argument("--no-abs-prior", action="store_true", help="(default now) no drift guard")
    tAp.add_argument("--abs-prior", action="store_true",
                     help="ablation: 0.15 m L2 pull toward the tag-only linear fill (REJECTED 10-23, kept for reproduction)")
    tAp.add_argument("--name", default="", help="output variant name (default = VARIANT); e.g. v2_megasam")
    return tAp


def main(argv: list[str] | None = None) -> int:
    tArgs = _buildParser().parse_args(argv)
    tSigmas = STITCH_SIGMAS
    if tArgs.abs_prior and not tArgs.no_abs_prior:
        tSigmas = dataclasses.replace(tSigmas, mAbsPrior=ABS_PRIOR_ABLATION_M)
    D = prepare(tArgs.take, tArgs.arm, tArgs.visual or None)
    if tArgs.mode == "A" or (tArgs.mode == "auto" and len(D["segs"]) <= 2):
        tSegs = sorted(D["segs"], key=lambda g: g["t"][0])
        tT = np.concatenate([tG["t"] for tG in tSegs])
        tOrder = np.argsort(tT, kind="stable")
        tT, tU = np.unique(tT[tOrder], return_index=True)
        tOrder = tOrder[tU]
        D["segs"] = [dict(t=tT, p=np.concatenate([tG["p"] for tG in tSegs])[tOrder],
                          R=R.concatenate([tG["R"] for tG in tSegs])[tOrder],
                          bad=np.concatenate([tG["bad"] for tG in tSegs])[tOrder], src=(0, 0))]
        print("mode A (one world for the whole input)", flush=True)
    tVariant = tArgs.variant
    for tWin in arm_data.loadTrimWindows()[tArgs.take]:
        if tArgs.demos and tWin.mDemo not in [int(tX) for tX in tArgs.demos.split(",")]:
            continue
        tW0, tW1 = tWin.mW0 - int(WINDOW_MARGIN_S * NS_PER_S), tWin.mW1 + int(WINDOW_MARGIN_S * NS_PER_S)
        for tOff in [int(tX) for tX in tArgs.offsets.split(",")]:
            tHid = hiddenMask(D["t"], tOff)
            tSol = solveDemo(D, tW0, tW1, tVariant, tHid, tSigmas)
            tM = (D["t"] >= tSol["tn"][0]) & (D["t"] <= tSol["tn"][-1])
            tTt = D["t"][tM]
            tP = np.stack([np.interp(tTt, tSol["tn"], tSol["P"][:, tQ]) for tQ in range(3)], 1)
            tRr = pose_smoother.rotationAt(D, tSol["tn"], tSol["Rn"], tTt)
            tOut = workdirs.qaRunPath(tArgs.name or tVariant, tArgs.take, tArgs.arm, tWin.mDemo, tOff)
            tOut.parent.mkdir(parents=True, exist_ok=True)
            np.savez(tOut, t_ns=tTt, p_B=tP, R_B_tcp=tRr.as_matrix(), qa_hidden=tHid[tM], tn=tSol["tn"], Pn=tSol["P"],
                     scales=tSol["scales"], s0=tSol["s0"], seginfo=np.array(tSol["seginfo"], dtype=object), variant=tVariant,
                     cal=str({tKey: tVal for tKey, tVal in D["cal"].items() if not hasattr(tVal, "as_quat")}))
            print(tOut.name, "s0 %.3f scales" % tSol["s0"],
                  np.round(np.percentile(tSol["scales"], QUANTILES), 3) if len(tSol["scales"]) else "-", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
