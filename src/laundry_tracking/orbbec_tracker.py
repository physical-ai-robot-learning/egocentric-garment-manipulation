"""Orbbec UMI tracker v3.1 (was orbbec/v2b/track_v31.py, 10-23): per take, the two UMI body centres in the robot base frame from the
fixed Orbbec colour + depth frames, tag-free.

Inputs: datasets/tag_rig/<take> (meta.json, t_ns.npy, color_/depth_ frames), the per-take camera calibration, the start-marker seeds
FUSION_WORK/orbbec/v2b/start_seed_v1.json (per-take override start_seed_<take>.json; 10-26: fixed tags / start spots moved), the
arm IMUs (static-lock check) and, when present, out/trackv2b_<take>.npz (table height).
Output: FUSION_WORK/orbbec/v2b/out/<V31_OUT or trackv31>_<take>.npz: t_ns, z_table, roi, params, global_cands, c_<arm>, npts_<arm>,
valid_<arm> (same keys as v1 / v2b) + per-arm logs mode_<arm> (0 none, 1 local, 2 global), dpred_<arm> [m], cost_<arm>, ext_<arm>,
static_unlock_<arm>, ncand (global candidates).
Environment: V31_P = JSON overrides of TRACKER_PARAMS (e.g. '{"JUMP":0.10,"V_GATE":1.0}' = the v31speed production setting),
V31_OUT = output name (default trackv31).  Parameters set on 123747 only, then frozen.

v3.1 = v3.0 (frozen official) + dev-take-only fixes:
  (1) seed lock: an unseeded arm only accepts a candidate within SEED_R of its start marker (no side-prior Hungarian on the first
      frame); after SEED_TIMEOUT_S without a seed it falls back to the v3.0 lost-arm rule
  (2) EXT_MAX re-checked on 123747 (correct-UMI vs false global candidates)
  (3) W_FLAT swept on 123747 (0.05 / 0.1 / 0.2 / 0.3)
  (4) velocity: updated only for 0 < dt <= VEL_DT_MAX (else reset to 0); dt in the update floored at VEL_DT_MIN (frame drops)
  (5) table ROI (123747 R d1: the global search locked onto dark equipment beside the table): points must lie inside the table-plane
      xy bounds (1st..99th percentile of plane inliers) + ROI_PAD
  (6) [tried, reverted: side prior only while both arms are lost -- no gain]
  (7) static-lock check (123747 R d1: locked onto desk equipment / hair): over the last STATIC_WIN_S the tracked centre stays within
      STATIC_D while the arm's IMU |omega| (rest-period bias removed, low-passed) has median > STATIC_W_DPS -> inconsistency; after
      STATIC_N consecutive inconsistent frames the arm is unlocked (lost) and that position is blacklisted (BLACK_R) for BLACK_S
  (8) workspace ROI from the protocol start markers (tag-free): x, y within the markers' range +- WS_R (intersected with (5))
v3.0 (targets the v1 failure analysis: search_miss, cloth_overlap, swap / false object):
  points per frame (stride 2, base frame): dark (gray < DARK) AND H_MIN..H_MAX above the table AND not in an image component touching
  the bottom edge (operator); NO size filter.  Weights: 1 where the height top-hat (height minus its K_OPEN opening) > TH (locally
  raised: the rigid body), W_FLAT elsewhere (flat dark cloth), so mean-shift prefers the raised body inside a big dark blob.
  candidates: local = weighted mean-shift (bandwidth BW) from the constant-velocity prediction of a tracked arm; global (an arm lost
  > LOST_S) = modes of the whole frame (mean-shift from seeds on every component, merged within MERGE), npts >= MIN_PTS, extent <= EXT_MAX.
  joint assignment (Hungarian) of {left, right} to candidates, cost = distance to the prediction / gate (R_GATE + V_GATE * time since
  the last accepted centre, cap GATE_LOST) + SIDE_W * side-prior violation (lost arm only: base-y sign of its start marker); pairs
  above cost 1 are not assigned; two arms never share a candidate; a lost arm's candidate must be >= MIN_SEP from the other arm.
  identity: seeded once per demo window at the start markers, then only this assignment.
    python -m laundry_tracking.orbbec_tracker <take>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment

from laundry_common import paths

from . import imu_motion
from . import workdirs
from .camera_calibration import baseFromCamera
from .constants import MM_PER_M, NS_PER_S
from .fgo import arm_data

PARAMS_ENV = "V31_P"
OUT_NAME_ENV = "V31_OUT"
DEFAULT_OUT_NAME = "trackv31"
# Tracker parameters (dict: stored as JSON in the output and overridable through V31_P).
TRACKER_PARAMS = dict(DARK=40, H_MIN=0.020, H_MAX=0.65, STRIDE=2, K_OPEN=15, TH=0.025, W_FLAT=0.3, BW=0.07, MIN_PTS=60,
                      EXT_MAX=0.09, MERGE=0.06, R_GATE=0.07, V_GATE=0.5, GATE_LOST=0.6, LOST_S=0.3, SIDE_W=1.0, MIN_SEP=0.10,
                      JUMP=0.06, N_SEEDS=4, SEED_R=0.15, SEED_TIMEOUT_S=3.0, VEL_DT_MIN=0.05, VEL_DT_MAX=0.25, ROI_PAD=0.02,
                      STATIC_WIN_S=0.8, STATIC_D=0.015, STATIC_W_DPS=30.0, STATIC_N=5, BLACK_R=0.08, BLACK_S=5.0, WS_R=0.45)
ARMS = ("left", "right")
FREE_CUE_PREFIX = arm_data.FREE_CUE_PREFIX
DEPTH_MIN_M, DEPTH_MAX_M = 0.2, 2.0           # valid Orbbec depth range
TABLE_SAMPLE_FRAMES = 12                      # frames sampled for the table height (mode of bright-pixel base z)
TABLE_BRIGHT_GRAY = 150
TABLE_HIST_BINS = 400
TABLE_HIST_RANGE_M = (-0.3, 0.3)
TABLE_HIST_HALF_BIN_M = 0.00075               # bin centre = left edge + half a bin (0.6 m / 400 / 2)
ROI_SAMPLE_FRAMES = 6
ROI_PLANE_TOL_M = 0.01
ROI_PERCENTILES = (1, 99)
OMEGA_SMOOTH_SAMPLES = 0.05 * 420             # 50 ms at the nominal 420 Hz IMU rate
WINDOW_MARGIN_S = 2.0                         # tracked span = protocol window +- 2 s
LOST_START_S = 10.0                           # every arm starts 'lost' (last accepted centre 10 s ago) at its marker
MESHIFT_ITERS = 6
MESHIFT_MIN_WEIGHT = 1e-6
PREDICT_MAX_S = 0.5                           # constant-velocity prediction horizon cap
SEED_POINTS_PER = 400                         # one extra global seed per 400 component points
NO_ASSIGN_COST = 9.0
MAX_ASSIGN_COST = 1.0
SIDE_MIN_ABS_Y_M = 0.05                       # side prior only for candidates more than 5 cm off the base x axis
FRAME_S = 0.067                               # jump allowance grows with the time beyond one frame
STATIC_SPAN_FRACTION = 0.8                    # static check needs history over 80 % of STATIC_WIN_S
VEL_KEEP = 0.5                                # velocity update: 0.5 old + 0.5 new


@dataclass
class _ArmState:
    """tracking state of one arm inside a demo window (mutable)"""
    mC: np.ndarray
    mV: np.ndarray
    mTl: int
    mSeeded: bool = False
    mNstat: int = 0


def trackerParams() -> dict:
    """TRACKER_PARAMS with the V31_P overrides"""
    tParams = dict(TRACKER_PARAMS)
    tParams.update(json.loads(os.environ.get(PARAMS_ENV, "{}")))
    return tParams


def meanShift(pts: np.ndarray, w: np.ndarray, c: np.ndarray, bw: float, it: int = MESHIFT_ITERS):
    """weighted mean-shift from c -> (centre or None, final selection mask) (was track_v31.wshift)"""
    for _ in range(it):
        tSel = np.linalg.norm(pts - c, axis=1) < bw
        if w[tSel].sum() < MESHIFT_MIN_WEIGHT:
            return None, tSel
        c = (pts[tSel] * w[tSel, None]).sum(0) / w[tSel].sum()
    return c, np.linalg.norm(pts - c, axis=1) < bw


def loadSeeds(take: str) -> dict:
    """start-marker seeds {arm: [x, y, z]} (start_seed_v1.json, or the per-take start_seed_<take>.json when present)"""
    tPerTake = workdirs.startSeedPath(take)
    tFile = tPerTake if tPerTake.exists() else workdirs.startSeedPath()
    return json.loads(tFile.read_text())


def _baseGrid(tRays, depthMm, tbc):
    tZ = depthMm.astype(np.float32) / MM_PER_M
    tOk = (tZ > DEPTH_MIN_M) & (tZ < DEPTH_MAX_M)
    tXn, tYn = tRays
    return np.stack([tXn * tZ, tYn * tZ, tZ], -1) @ tbc[:3, :3].T + tbc[:3, 3], tOk


def track(take: str, params: dict, outName: str) -> None:
    tP = params
    tTakeDir = paths.TAG_RIG / take
    tIntr = json.loads((tTakeDir / "meta.json").read_text())["intrinsics"]
    tTbc = baseFromCamera(take)                                     # 10-26: per-take camera calibration
    tSeeds = loadSeeds(take)
    tT = np.load(tTakeDir / "t_ns.npy").astype(np.int64)
    tN = len(tT)
    tStride = tP["STRIDE"]
    tH, tW = cv2.imread(str(tTakeDir / "color_00000.jpg")).shape[:2]
    tVv, tUu = np.mgrid[0:tH:tStride, 0:tW:tStride]
    tRays = ((tUu - tIntr["cx"]) / tIntr["fx"], (tVv - tIntr["cy"]) / tIntr["fy"])
    tV2b = workdirs.trackPath("trackv2b", take)
    if tV2b.exists():
        tZTable = float(np.load(tV2b)["z_table"])
    else:                                                           # same estimate as orbbec_track.py: mode of bright-pixel base z
        tZs = []
        for tK in np.linspace(0, tN - 1, TABLE_SAMPLE_FRAMES).astype(int):
            tC0 = cv2.imread(str(tTakeDir / f"color_{tK:05d}.jpg"), cv2.IMREAD_GRAYSCALE)
            tD0 = cv2.imread(str(tTakeDir / f"depth_{tK:05d}.png"), -1)
            if tC0 is None or tD0 is None:
                continue
            tX0, tOk0 = _baseGrid(tRays, tD0[::tStride, ::tStride], tTbc)
            tZs.append(tX0[tOk0 & (tC0[::tStride, ::tStride] > TABLE_BRIGHT_GRAY)][:, 2])
        tHist, tEdges = np.histogram(np.concatenate(tZs), bins=TABLE_HIST_BINS, range=TABLE_HIST_RANGE_M)
        tZTable = float(tEdges[np.argmax(tHist)] + TABLE_HIST_HALF_BIN_M)
    tKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tP["K_OPEN"], tP["K_OPEN"]))
    tTablePts = []
    for tK in np.linspace(0, tN - 1, ROI_SAMPLE_FRAMES).astype(int):
        tD0 = cv2.imread(str(tTakeDir / f"depth_{tK:05d}.png"), -1)
        if tD0 is None:
            continue
        tX0, tOk0 = _baseGrid(tRays, tD0[::tStride, ::tStride], tTbc)
        tX0 = tX0[tOk0]
        tTablePts.append(tX0[np.abs(tX0[:, 2] - tZTable) < ROI_PLANE_TOL_M])
    tTablePts = np.concatenate(tTablePts)
    tLo, tHi = ROI_PERCENTILES
    tRoi = (np.percentile(tTablePts[:, 0], tLo) - tP["ROI_PAD"], np.percentile(tTablePts[:, 0], tHi) + tP["ROI_PAD"],
            np.percentile(tTablePts[:, 1], tLo) - tP["ROI_PAD"], np.percentile(tTablePts[:, 1], tHi) + tP["ROI_PAD"])
    tSx = [tSeeds[tA][0] for tA in ARMS]
    tSy = [tSeeds[tA][1] for tA in ARMS]
    tRoi = (max(tRoi[0], min(tSx) - tP["WS_R"]), min(tRoi[1], max(tSx) + tP["WS_R"]), max(tRoi[2], min(tSy) - tP["WS_R"]),
            min(tRoi[3], max(tSy) + tP["WS_R"]))                     # (8)
    tWindows = arm_data.loadTrimWindows()
    tOmega = {}
    for tA in ARMS:
        D = arm_data.loadArm(take, tA, withSlam=False)
        D["b0"] = np.zeros(3)
        tB = imu_motion.restBias(D, tWindows[take])
        tW = np.degrees(np.linalg.norm(D["gyro"] - tB, axis=1))
        tOmega[tA] = (D["imu_t"], gaussian_filter1d(tW, OMEGA_SMOOTH_SAMPLES))
    tSide = {tA: np.sign(tSeeds[tA][1]) for tA in ARMS}
    tHistory = {tA: [] for tA in ARMS}               # (t, centre) of the last STATIC_WIN_S per arm; kept across demo windows
    tBlack = []
    tResStatic = {tA: np.zeros(tN, bool) for tA in ARMS}
    tRes = {f"c_{tA}": np.full((tN, 3), np.nan) for tA in ARMS}
    for tA in ARMS:
        tRes[f"npts_{tA}"] = np.zeros(tN, int)
        tRes[f"mode_{tA}"] = np.zeros(tN, int)
        tRes[f"dpred_{tA}"] = np.full(tN, np.nan)
        tRes[f"cost_{tA}"] = np.full(tN, np.nan)
        tRes[f"ext_{tA}"] = np.full(tN, np.nan)
    tRes["ncand"] = np.zeros(tN, int)
    tGlobalLog = []
    tMeta = json.loads((tTakeDir / "meta.json").read_text())
    tT0 = int(tMeta["t_start_monotonic_ns"])
    tWins = [(tT0 + int(tC["t0"] * NS_PER_S), tT0 + int((tC["t0"] + tC["dur"]) * NS_PER_S)) for tC in tMeta["protocol"]
             if tC["text"].startswith(FREE_CUE_PREFIX)]
    for tW0, tW1 in tWins:
        tKs = np.flatnonzero((tT >= tW0 - WINDOW_MARGIN_S * NS_PER_S) & (tT <= tW1 + WINDOW_MARGIN_S * NS_PER_S))
        if not len(tKs):
            continue                                                # protocol window without Orbbec frames
        tState = {tA: _ArmState(mC=np.array(tSeeds[tA]), mV=np.zeros(3),
                                mTl=tT[tKs[0]] - int(LOST_START_S * NS_PER_S)) for tA in ARMS}
        for tK in tKs:
            _trackFrame(tK, tT, tTakeDir, tStride, tRays, tTbc, tZTable, tRoi, tKernel, tP, tSeeds, tSide, tState, tKs, tHistory,
                        tBlack, tOmega, tRes, tResStatic, tGlobalLog)
    for tA in ARMS:
        tRes[f"valid_{tA}"] = np.isfinite(tRes[f"c_{tA}"][:, 0])
    tRes.update({f"static_unlock_{tA}": tResStatic[tA] for tA in ARMS})
    workdirs.TRACK_OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(workdirs.trackPath(outName, take), t_ns=tT, z_table=tZTable, roi=np.array(tRoi), params=json.dumps(tP),
             global_cands=np.array(tGlobalLog).reshape(-1, 6), **tRes)
    tInWin = np.zeros(tN, bool)
    for tW0, tW1 in tWins:
        tInWin |= (tT >= tW0) & (tT <= tW1)
    print(take, "static unlocks", {tA: int(tResStatic[tA].sum()) for tA in ARMS}, "roi", np.round(tRoi, 2),
          {tA: round(float(tRes[f"valid_{tA}"][tInWin].mean()), 3) for tA in ARMS}, "global-mode share",
          {tA: round(float((tRes[f"mode_{tA}"][tInWin] == 2).mean()), 3) for tA in ARMS})


def _trackFrame(tK, tT, tTakeDir, tStride, tRays, tTbc, tZTable, tRoi, tKernel, tP, tSeeds, tSide, tState, tKs, tHistory, tBlack,
                tOmega, tRes, tResStatic, tGlobalLog) -> None:
    """one Orbbec frame: candidates, joint assignment, state / static-lock update (results written into tRes)"""
    tCol = cv2.imread(str(tTakeDir / f"color_{tK:05d}.jpg"), cv2.IMREAD_GRAYSCALE)
    tDepth = cv2.imread(str(tTakeDir / f"depth_{tK:05d}.png"), -1)
    if tCol is None or tDepth is None:
        return
    tXg, tOk = _baseGrid(tRays, tDepth[::tStride, ::tStride], tTbc)
    tHeight = tXg[..., 2] - tZTable
    tGray = tCol[::tStride, ::tStride]
    tInRoi = (tXg[..., 0] > tRoi[0]) & (tXg[..., 0] < tRoi[1]) & (tXg[..., 1] > tRoi[2]) & (tXg[..., 1] < tRoi[3])
    tMask = (tOk & (tGray < tP["DARK"]) & (tHeight > tP["H_MIN"]) & (tHeight < tP["H_MAX"]) & tInRoi).astype(np.uint8)
    tNc, tLab, tStats, _ = cv2.connectedComponentsWithStats(tMask, connectivity=8)
    tDrop = [tJ for tJ in range(1, tNc) if tStats[tJ, cv2.CC_STAT_TOP] + tStats[tJ, cv2.CC_STAT_HEIGHT] >= tMask.shape[0] - 1]
    if tDrop:
        tMask[np.isin(tLab, tDrop)] = 0
    tHz = np.where(tOk, tHeight, 0).astype(np.float32)
    tTop = tHz - cv2.morphologyEx(tHz, cv2.MORPH_OPEN, tKernel)
    tSelM = tMask > 0
    tPts = tXg[tSelM]
    tWts = np.where(tTop[tSelM] > tP["TH"], 1.0, tP["W_FLAT"])
    tLabs = tLab[tSelM]
    if len(tPts) < tP["MIN_PTS"]:
        return
    # ---- candidates: (centre, npts, extent, mode)
    tCands = []
    tPred = {}
    tLost = {}
    for tA in ARMS:
        tS = tState[tA]
        tDt = (tT[tK] - tS.mTl) / NS_PER_S
        tLost[tA] = (not tS.mSeeded) or tDt > tP["LOST_S"]
        tPred[tA] = tS.mC + (tS.mV * min(tDt, PREDICT_MAX_S) if not tLost[tA] else 0)
        if not tLost[tA]:
            tC, tSel = meanShift(tPts, tWts, tPred[tA], tP["BW"])
            if tC is not None and tSel.sum() >= tP["MIN_PTS"]:
                tCands.append((tC, int(tSel.sum()), float(np.sqrt(tPts[tSel].var(0).sum())), 1))
    if any(tLost.values()):
        tRng = np.random.default_rng(tK)
        for tJ in np.unique(tLabs):
            tIdx = np.flatnonzero(tLabs == tJ)
            if len(tIdx) < tP["MIN_PTS"]:
                continue
            for tSd in tPts[tRng.choice(tIdx, min(tP["N_SEEDS"] + len(tIdx) // SEED_POINTS_PER, len(tIdx)), replace=False)]:
                tC, tSel = meanShift(tPts, tWts, tSd, tP["BW"])
                if tC is None or tSel.sum() < tP["MIN_PTS"]:
                    continue
                tExt = float(np.sqrt(tPts[tSel].var(0).sum()))
                if any(np.linalg.norm(tC - tQ[0]) < tP["MERGE"] for tQ in tCands):
                    continue
                tGlobalLog.append((tK, *tC, tExt, int(tSel.sum())))
                if tExt > tP["EXT_MAX"]:
                    continue
                tCands.append((tC, int(tSel.sum()), tExt, 2))
        tRes["ncand"][tK] = sum(1 for tQ in tCands if tQ[3] == 2)
    if not tCands:
        return
    # ---- joint assignment
    tCost = np.full((2, len(tCands)), NO_ASSIGN_COST)
    for tI, tA in enumerate(ARMS):
        tS = tState[tA]
        tDt = (tT[tK] - tS.mTl) / NS_PER_S
        tGate = tP["GATE_LOST"] if tLost[tA] else min(tP["R_GATE"] + tP["V_GATE"] * tDt, tP["GATE_LOST"])
        for tJ, (tC, _, _, tMode) in enumerate(tCands):
            tCj = np.linalg.norm(tC - tPred[tA]) / tGate
            if any(np.linalg.norm(tC - tBp) < tP["BLACK_R"] and tT[tK] < tBt for tBp, tBt in tBlack):
                tCost[tI, tJ] = NO_ASSIGN_COST                                                    # (7)
                continue
            if not tS.mSeeded and (tT[tK] - tT[tKs[0]]) / NS_PER_S <= tP["SEED_TIMEOUT_S"]:
                tCost[tI, tJ] = np.linalg.norm(tC - np.array(tSeeds[tA])) / tP["SEED_R"]          # (1) seed lock: start marker only
                continue
            if tLost[tA]:
                tCj += tP["SIDE_W"] * float(np.sign(tC[1]) != tSide[tA] and abs(tC[1]) > SIDE_MIN_ABS_Y_M)
                tOther = "right" if tA == "left" else "left"
                if not tLost[tOther] and np.linalg.norm(tC - tState[tOther].mC) < tP["MIN_SEP"]:
                    tCj = NO_ASSIGN_COST
            elif tMode == 2 and np.linalg.norm(tC - tPred[tA]) > tP["JUMP"]:
                tCj = NO_ASSIGN_COST                                                              # tracked arms do not jump to a global mode
            tCost[tI, tJ] = tCj
    tRows, tColsIdx = linear_sum_assignment(tCost)
    for tI, tJ in zip(tRows, tColsIdx):
        if tCost[tI, tJ] > MAX_ASSIGN_COST:
            continue
        tA = ARMS[tI]
        tC, tNpt, tExt, tMode = tCands[tJ]
        tS = tState[tA]
        tDt = (tT[tK] - tS.mTl) / NS_PER_S
        if not tLost[tA] and np.linalg.norm(tC - tPred[tA]) > tP["JUMP"] + tP["V_GATE"] * max(tDt - FRAME_S, 0):
            continue
        if not tLost[tA] and 0 < tDt <= tP["VEL_DT_MAX"]:                                         # (4)
            tS.mV = VEL_KEEP * tS.mV + VEL_KEEP * (tC - tS.mC) / max(tDt, tP["VEL_DT_MIN"])
        else:
            tS.mV = np.zeros(3)
        tRes[f"dpred_{tA}"][tK] = np.linalg.norm(tC - tPred[tA])
        tRes[f"cost_{tA}"][tK] = tCost[tI, tJ]
        tS.mC, tS.mTl, tS.mSeeded = tC, tT[tK], True
        tHistory[tA] = [(tTh, tCh) for tTh, tCh in tHistory[tA] if tT[tK] - tTh <= tP["STATIC_WIN_S"] * NS_PER_S] + [(tT[tK], tC)]
        if (tT[tK] - tHistory[tA][0][0]) >= STATIC_SPAN_FRACTION * tP["STATIC_WIN_S"] * NS_PER_S:
            tHistC = np.array([tCh for _, tCh in tHistory[tA]])
            tTi, tWv = tOmega[tA]
            tMw = (tTi >= tHistory[tA][0][0]) & (tTi <= tT[tK])
            if np.linalg.norm(tHistC - tHistC.mean(0),
                              axis=1).max() < tP["STATIC_D"] and tMw.any() and np.median(tWv[tMw]) > tP["STATIC_W_DPS"]:
                tS.mNstat = tS.mNstat + 1
                if tS.mNstat >= tP["STATIC_N"]:
                    tBlack.append((tC.copy(), tT[tK] + int(tP["BLACK_S"] * NS_PER_S)))
                    tS.mSeeded = True
                    tS.mTl = tT[tK] - int(LOST_START_S * NS_PER_S)
                    tHistory[tA] = []
                    tS.mNstat = 0
                    tResStatic[tA][tK] = True
                    continue                                                                      # unlocked: this frame's centre is not written
            else:
                tS.mNstat = 0
        tRes[f"c_{tA}"][tK] = tC
        tRes[f"npts_{tA}"][tK] = tNpt
        tRes[f"ext_{tA}"][tK] = tExt
        tRes[f"mode_{tA}"][tK] = tMode


def main(argv: list[str] | None = None) -> int:
    tAp = argparse.ArgumentParser(description="Orbbec UMI tracker v3.1 (env V31_P = JSON parameter overrides, V31_OUT = output name)")
    tAp.add_argument("take")
    tArgs = tAp.parse_args(argv)
    track(tArgs.take, trackerParams(), os.environ.get(OUT_NAME_ENV, DEFAULT_OUT_NAME))
    return 0


if __name__ == "__main__":
    sys.exit(main())
