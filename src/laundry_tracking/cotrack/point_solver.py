"""CoTracker3 points -> UMI body origin with a point LIFECYCLE (was orbbec/v2b/cotrack/solve2.py, 10-24; pack v2: initial seeds +
reserve candidates every 1 s).  States: RESERVE (tracked by CoTracker, not used) -> PROBATION -> TRUSTED -> (OUTLIER streak) -> DROPPED.
  pose      AHRS rotation fixed; votes of visible TRUSTED points c_i = P_i - R q_i, weight 0.3 + 0.7 min(age / AGE_FULL, 1); weighted
            median, inliers within INL (2 rounds); pose valid if >= MIN_INL inliers and inlier fraction >= MIN_FRAC.  conf HIGH if
            >= HIGH_INL inliers, fraction >= 0.6 and 3D spread >= SPREAD_MIN.
  outliers  a TRUSTED point outside INL for OUT_N consecutive visible frames -> DROPPED (also PROBATION after 2 misses)
  promotion a PROBATION point within INL for PROB_N consecutive visible frames -> TRUSTED (q fixed at activation)
  refill    only while the pose is HIGH and (alive TRUSTED < REFILL_FRAC x initial, or spread < SPREAD_MIN, or residual > RES_HI), at
            most every REFILL_GAP_S: up to N_ADD visible RESERVE points whose 3D point lies within BODY_R of the origin -> PROBATION
  lost      no valid pose: candidate = RESERVE points started within the last 1 s, rigid-model fit around their centroid
            (fit >= FIT_MIN), within min(GATE0 + GATE_V dt, GATE_MAX) of the last pose -> those points become PROBATION with q from
            that fit; the pose is output again once >= MIN_INL of them are TRUSTED (self-consistent for PROB_N frames)
Metrics vs tag GT (fixed per-rig offset) + vs the v31speed (+v4) raw baseline: coverage, longest gap, abs error, far10, e32, longest
wrong burst (> 10 cm, unknown frames joined only between wrong frames), trusted points min / median, refill + reacquisition count,
longest no-trusted gap.
Inputs: <pack>/<take>_dNN/{meta.json, tracks.npz}, the take's depth / colour frames, FUSION_WORK/orbbec/v2b/out/{trackv31speed,
trackv4speed}_<take>.npz and rig_model_<rig>.npz, the arm record and demo windows.
Output: <pack>/<take>_dNN/solve2_<arm>[_h<off>].npz {c, conf, ntr, t, gt}.
    python -m laundry_tracking.cotrack.point_solver <pack_dir> [hide_off]   (hide_off 0/1/2: yaw anchor from visible tags only)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from laundry_common import paths

from .. import workdirs
from ..arm_rotation import armRotations, rigidFit
from ..camera_calibration import baseFromCamera
from ..constants import MM_PER_M, NS_PER_S, ORBBEC_FRAME_S
from ..fgo import arm_data
from ..intervals import runDurationsS
from ..whole_demo_stitch import hiddenMask

ARMS = ("left", "right")
REFERENCE_TRACK = "trackv31speed"
BASELINE_TRACKS = ("trackv4speed", "trackv31speed")   # first one present is the printed raw baseline
VISIBLE_MIN = 0.5                  # CoTracker visibility above this = visible
FAR_MM = 100                       # wrong = more than 10 cm from the tag GT
E32_HORIZON_NS = int(1.6e9)        # e32: 1.6 s displacement error
E32_TOL_NS = 70e6
WEIGHT_BASE, WEIGHT_AGE = 0.3, 0.7 # vote weight 0.3 + 0.7 min(age / AGE_FULL, 1)
INLIER_ROUNDS = 2
HIGH_FRACTION = 0.6
PROBATION_MAX_MISSES = 2
RECENT_RESERVE_FRAMES = 15         # re-acquisition only from reserve points started within the last 15 frames (1 s)
FIT_RADIUS_M = 0.13                # rigid-fit points within 13 cm of the candidate centroid
FIT_DARK = 40
FIT_H_MIN = 0.02
FIT_MIN_POINTS = 40
MIN_DEPTH_M = 0.2
NEG_INF_NS = -1e18
STATE_DROPPED, STATE_RESERVE, STATE_PROBATION, STATE_TRUSTED = -1, 0, 1, 2
KIND_SEED, KIND_RESERVE = 0, 1
CONF_MID, CONF_HIGH = 1, 2


@dataclass(frozen=True)
class SolverParams:
    """lifecycle parameters (was the P dict of solve2.py)"""
    mInl: float = 0.025
    mMinInl: int = 6
    mMinFrac: float = 0.4
    mHighInl: int = 10
    mSpreadMin: float = 0.012
    mOutN: int = 3
    mProbN: int = 4
    mAgeFull: int = 30
    mRefillFrac: float = 0.5
    mResHi: float = 0.012
    mRefillGapS: float = 0.5
    mNAdd: int = 15
    mBodyR: float = 0.08
    mFitMin: float = 0.3
    mGate0: float = 0.05
    mGateV: float = 1.0
    mGateMax: float = 0.4


SOLVER_PARAMS = SolverParams()


def liftTracks(depth: list, tracks: np.ndarray, intr: dict, tbc: np.ndarray) -> np.ndarray:
    """(T, N, 3) base-frame points from a 3x3 median of depth at every track pixel (NaN where invalid) (was solve2.lift_all)"""
    tT, tN, _ = tracks.shape
    tOut = np.full((tT, tN, 3), np.nan)
    for tJ in range(tT):
        tD = depth[tJ]
        tH, tW = tD.shape
        tU = tracks[tJ, :, 0]
        tV = tracks[tJ, :, 1]
        tOk = np.isfinite(tU) & (tU >= 1) & (tU < tW - 1) & (tV >= 1) & (tV < tH - 1)
        tUi, tVi = tU[tOk].astype(int), tV[tOk].astype(int)
        tStack = np.stack([tD[tVi + tDy, tUi + tDx] for tDy in (-1, 0, 1) for tDx in (-1, 0, 1)], 1).astype(float)
        tStack[tStack <= 0] = np.nan
        tZ = np.nanmedian(tStack, 1) / MM_PER_M if tStack.size else np.zeros(0)
        tIdx = np.flatnonzero(tOk)
        tGood = np.isfinite(tZ)
        tIdx, tZ, tUu, tVv = tIdx[tGood], tZ[tGood], tU[tOk][tGood], tV[tOk][tGood]
        tOut[tJ, tIdx] = np.stack([(tUu - intr["cx"]) / intr["fx"] * tZ, (tVv - intr["cy"]) / intr["fy"] * tZ, tZ],
                                  1) @ tbc[:3, :3].T + tbc[:3, 3]
    return tOut


def weightedMedian(x: np.ndarray, w: np.ndarray):
    """was solve2.wmed"""
    tOrder = np.argsort(x)
    tCw = np.cumsum(w[tOrder])
    return x[tOrder[np.searchsorted(tCw, tCw[-1] / 2)]]


def trackMetrics(c: np.ndarray, t: np.ndarray, gt: np.ndarray, okGt: np.ndarray, okGt2: np.ndarray) -> dict:
    """coverage / gap / abs / far10 / wrong burst / e32 of a centre track vs the tag GT (was solve2.metrics)"""
    tValid = np.isfinite(c[:, 0])
    tM = tValid & okGt
    tAbsErr = np.linalg.norm(c - gt, axis=1) * MM_PER_M
    tLab = np.full(len(t), np.nan)
    tLab[tM] = (tAbsErr[tM] > FAR_MM).astype(float)
    tWrong = np.zeros(len(t), bool)
    tKnown = np.flatnonzero(np.isfinite(tLab))
    for tA, tB in zip(tKnown[:-1], tKnown[1:]):
        if tLab[tA] == 1 and tLab[tB] == 1 and tValid[tA:tB + 1].all():
            tWrong[tA:tB + 1] = True
    tWrong[tKnown[tLab[tKnown] == 1]] = True
    tE = []
    for tI in range(len(t)):
        tJ = np.searchsorted(t, t[tI] + E32_HORIZON_NS)
        if tJ < len(t) and abs(t[tJ] - t[tI] - 1.6e9) < E32_TOL_NS and tValid[tI] and tValid[tJ] and okGt2[tI] and okGt2[tJ]:
            tE.append(np.linalg.norm((c[tJ] - c[tI]) - (gt[tJ] - gt[tI])) * MM_PER_M)
    return dict(cov=round(float(tValid.mean()), 3), gap_max_s=round(max(runDurationsS(~tValid, t, ORBBEC_FRAME_S) or [0.0]), 2),
                abs_med=round(float(np.nanmedian(tAbsErr[tM])), 1) if tM.any() else np.nan,
                far10=round(float(np.nanmean(tAbsErr[tM] > FAR_MM)), 3) if tM.any() else np.nan,
                wrong_burst_s=round(max(runDurationsS(tWrong, t, ORBBEC_FRAME_S) or [0.0]), 2),
                e32_med=round(float(np.median(tE)), 1) if tE else np.nan, e32_p90=round(float(np.percentile(tE, 90)),
                                                                                        1) if tE else np.nan,
                n_e32=len(tE))


def solveArm(PT, VIS, qStart, kind, Rm, t, j0, T0, M, fitFn, params: SolverParams = SOLVER_PARAMS):
    """point lifecycle for one arm -> (centre (T,3), conf (T,), trusted count (T,), refill + reacquisition events) (was solve2.solve_arm)"""
    tP = params
    tT, tN = VIS.shape
    tState = np.zeros(tN, int)
    tQ = np.full((tN, 3), np.nan)
    tAge = np.zeros(tN)
    tOutStreak = np.zeros(tN, int)
    tGood = np.zeros(tN, int)
    tSeed = np.flatnonzero((kind == KIND_SEED) & np.isfinite(PT[j0, :, 0]))
    tQ[tSeed] = (PT[j0, tSeed] - T0) @ Rm[j0]
    tState[tSeed] = STATE_TRUSTED
    tNInit = len(tSeed)
    tC = np.full((tT, 3), np.nan)
    tConf = np.zeros(tT, int)
    tNtr = np.zeros(tT, int)
    tEvents = 0
    tLastRefill = NEG_INF_NS
    tLastC = T0.copy()
    tLastT = t[j0]
    for tJ in range(j0, tT):
        tR = Rm[tJ]
        if not np.isfinite(tR[0, 0]):
            continue
        tVis = (VIS[tJ] > VISIBLE_MIN) & np.isfinite(PT[tJ, :, 0])
        tTr = np.flatnonzero((tState == STATE_TRUSTED) & tVis)
        tNtr[tJ] = int((tState == STATE_TRUSTED).sum())
        tPose = None
        if len(tTr) >= tP.mMinInl:
            tVotes = PT[tJ, tTr] - tQ[tTr] @ tR.T
            tWt = WEIGHT_BASE + WEIGHT_AGE * np.minimum(tAge[tTr] / tP.mAgeFull, 1)
            tMed = np.array([weightedMedian(tVotes[:, tK], tWt) for tK in range(3)])
            for _ in range(INLIER_ROUNDS):
                tInl = np.linalg.norm(tVotes - tMed, axis=1) < tP.mInl
                if not tInl.any():
                    break
                tMed = (tVotes[tInl] * tWt[tInl, None]).sum(0) / tWt[tInl].sum()
            if tInl.sum() >= tP.mMinInl and tInl.mean() >= tP.mMinFrac:
                tPose = tMed
                tRes = float(np.median(np.linalg.norm(tVotes[tInl] - tMed, axis=1)))
                tSpread = float(np.sqrt(PT[tJ, tTr[tInl]].var(0).sum()))
                tHigh = tInl.sum() >= tP.mHighInl and tInl.mean() >= HIGH_FRACTION and tSpread >= tP.mSpreadMin
                tConf[tJ] = CONF_HIGH if tHigh else CONF_MID
                tAge[tTr[tInl]] += 1
                tOutStreak[tTr[tInl]] = 0
                tOutStreak[tTr[~tInl]] += 1
                tState[tTr[~tInl][tOutStreak[tTr[~tInl]] >= tP.mOutN]] = STATE_DROPPED
        if tPose is not None:
            tC[tJ] = tPose
            tLastC, tLastT = tPose, t[tJ]
            tPr = np.flatnonzero((tState == STATE_PROBATION) & tVis)
            if len(tPr):
                tOk = np.linalg.norm(PT[tJ, tPr] - tQ[tPr] @ tR.T - tPose, axis=1) < tP.mInl
                tGood[tPr[tOk]] += 1
                tOutStreak[tPr[~tOk]] += 1
                tState[tPr[tOk & (tGood[tPr] >= tP.mProbN)]] = STATE_TRUSTED
                tState[tPr[~tOk & (tOutStreak[tPr] >= PROBATION_MAX_MISSES)]] = STATE_DROPPED
            tAlive = int((tState == STATE_TRUSTED).sum())
            if tConf[tJ] == CONF_HIGH and (tAlive < tP.mRefillFrac * tNInit or tSpread < tP.mSpreadMin or tRes > tP.mResHi) and \
                    t[tJ] - tLastRefill >= tP.mRefillGapS * NS_PER_S:
                tRv = np.flatnonzero((tState == STATE_RESERVE) & tVis & (qStart <= tJ) & (kind == KIND_RESERVE))
                tRv = tRv[np.linalg.norm(PT[tJ, tRv] - tPose, axis=1) < tP.mBodyR][:tP.mNAdd]
                if len(tRv):
                    tQ[tRv] = (PT[tJ, tRv] - tPose) @ tR
                    tState[tRv] = STATE_PROBATION
                    tGood[tRv] = 0
                    tOutStreak[tRv] = 0
                    tLastRefill = t[tJ]
                    tEvents += 1
        else:                                                                # lost: re-acquire from recent reserve points
            tPr = np.flatnonzero((tState == STATE_PROBATION) & tVis)
            if len(tPr) and np.isfinite(tQ[tPr, 0]).all():
                tVotes = PT[tJ, tPr] - tQ[tPr] @ tR.T
                tMed = np.median(tVotes, 0)
                tOk = np.linalg.norm(tVotes - tMed, axis=1) < tP.mInl
                tGood[tPr[tOk]] += 1
                tState[tPr[~tOk]] = STATE_DROPPED
                tState[tPr[tOk & (tGood[tPr] >= tP.mProbN)]] = STATE_TRUSTED
            if (tState == STATE_PROBATION).sum() < tP.mMinInl:
                tRecent = (qStart <= tJ) & (qStart >= tJ - RECENT_RESERVE_FRAMES)
                tRv = np.flatnonzero((tState == STATE_RESERVE) & tVis & tRecent & (kind == KIND_RESERVE))
                if len(tRv) >= tP.mMinInl:
                    tCen = np.median(PT[tJ, tRv], 0)
                    tFit = fitFn(tJ, tCen)
                    if tFit is not None and tFit[2] >= tP.mFitMin:
                        tGate = min(tP.mGate0 + tP.mGateV * (t[tJ] - tLastT) / NS_PER_S, tP.mGateMax)
                        if np.linalg.norm(tFit[0] - tLastC) <= tGate:
                            tSel = tRv[np.linalg.norm(PT[tJ, tRv] - tFit[0], axis=1) < tP.mBodyR]
                            tQ[tSel] = (PT[tJ, tSel] - tFit[0]) @ tR
                            tState[tSel] = STATE_PROBATION
                            tGood[tSel] = 0
                            tEvents += 1
    return tC, tConf, tNtr, tEvents


def solveDemoDir(demoDir: Path, hideOffset: int | None) -> None:
    """both arms of one pack demo dir -> solve2_<arm>[_h<off>].npz + printed metrics"""
    tMeta = json.loads((demoDir / "meta.json").read_text())
    tTake = tMeta["take"]
    tTbc = baseFromCamera(tTake)
    tKs = np.array(tMeta["ks"])
    tTracksZ = np.load(demoDir / "tracks.npz")
    tTracks, tVisAll = tTracksZ["tracks"], tTracksZ["vis"].astype(float)
    tTakeDir = paths.TAG_RIG / tTake
    tIntr = json.loads((tTakeDir / "meta.json").read_text())["intrinsics"]
    tArmOfQuery = np.array(tMeta["arm_of_query"])
    tKind = np.array(tMeta["query_kind"])
    tQStart = np.array(tMeta["queries"])[:, 0].astype(int)
    tRef = np.load(workdirs.trackPath(REFERENCE_TRACK, tTake))
    tFull = tRef["t_ns"]
    tZTable = float(tRef["z_table"])
    tT = tFull[tKs]
    tAsg = json.loads((paths.DERIVED / tTake / "assignment.json").read_text())
    tBaseName = BASELINE_TRACKS[0] if workdirs.trackPath(BASELINE_TRACKS[0], tTake).exists() else BASELINE_TRACKS[1]
    tBase = np.load(workdirs.trackPath(tBaseName, tTake))
    tDepth = [cv2.imread(str(tTakeDir / f"depth_{tK:05d}.png"), -1) for tK in tKs]
    tPts = liftTracks(tDepth, tTracks, tIntr, tTbc)
    for tArm in ARMS:
        tAnchorOk = None if hideOffset is None else ~hiddenMask(tFull, hideOffset)
        tRm = armRotations(tTake, tFull, tRef[f"c_{tArm}"], tArm, anchorOk=tAnchorOk)[tKs]
        tSel = np.flatnonzero(tArmOfQuery == tArm)
        tJ0 = tMeta["seeds"][tArm]["frame"]
        tModel = np.load(workdirs.rigModelPath(tAsg["rig_of_arm"][tArm]))["pts"]

        def _fit(j, cen, Rm=tRm, M=tModel):
            if not np.isfinite(Rm[j, 0, 0]):
                return None
            tGray = cv2.imread(str(tTakeDir / f"color_{tKs[j]:05d}.jpg"), cv2.IMREAD_GRAYSCALE)
            tDd = tDepth[j].astype(np.float32) / MM_PER_M
            tVv, tUu = np.mgrid[0:tDd.shape[0], 0:tDd.shape[1]]
            tX = np.stack([(tUu - tIntr["cx"]) / tIntr["fx"] * tDd, (tVv - tIntr["cy"]) / tIntr["fy"] * tDd, tDd],
                          -1) @ tTbc[:3, :3].T + tTbc[:3, 3]
            tFitPts = tX[(tDd > MIN_DEPTH_M) & (tGray < FIT_DARK) & (tX[...,
                                                                        2] - tZTable > FIT_H_MIN) & (np.linalg.norm(tX - cen,
                                                                                axis=-1) < FIT_RADIUS_M)]
            return rigidFit(tFitPts, Rm[j], M, cen) if len(tFitPts) > FIT_MIN_POINTS else None
        tC0 = np.array(tMeta["seeds"][tArm]["centre"])
        tF0 = _fit(tJ0, tC0)
        tT0 = tF0[0] if tF0 is not None else np.nanmean(tPts[tJ0, tSel[tKind[tSel] == KIND_SEED]], 0)
        tC, tConf, tNtr, tEvents = solveArm(tPts[:, tSel], tVisAll[:, tSel], tQStart[tSel], tKind[tSel], tRm, tT, tJ0, tT0,
                                            tModel, _fit)
        D = arm_data.loadArm(tTake, tArm, withSlam=False)
        tOffset = workdirs.bodyOffsetM(tAsg["rig_of_arm"][tArm])
        tGt = D["p_tag"][tKs] + np.einsum("nij,j->ni", tRm, tOffset)
        tOkGt = (D["ntags"][tKs] >= 1) & np.isfinite(tRm[:, 0, 0])
        tOkGt2 = (D["ntags"][tKs] >= 2) & np.isfinite(tRm[:, 0, 0])
        tMetricsC = trackMetrics(tC, tT, tGt, tOkGt, tOkGt2)
        tMetricsB = trackMetrics(tBase[f"c_{tArm}"][tKs], tT, tGt, tOkGt, tOkGt2)
        tLive = tNtr[tJ0:]
        tNoTrusted = runDurationsS(tNtr[tJ0:] < SOLVER_PARAMS.mMinInl, tT[tJ0:], ORBBEC_FRAME_S)
        tMetricsC.update(trusted_min=int(tLive.min()), trusted_med=int(np.median(tLive)), events=tEvents,
                         no_trusted_gap_s=round(max(tNoTrusted or [0.0]),
                                                2), high=round(float((tConf[tJ0:] == CONF_HIGH).mean()), 2))
        print(f"{demoDir.name} {tArm[0]} | CT-v2 {tMetricsC}\n{'':22s}| base[{tBaseName}] {tMetricsB}", flush=True)
        tName = f"solve2_{tArm}.npz" if hideOffset is None else f"solve2_{tArm}_h{hideOffset}.npz"
        np.savez(demoDir / tName, c=tC, conf=tConf, ntr=tNtr, t=tT, gt=tGt)


def main(argv: list[str] | None = None) -> int:
    tAp = argparse.ArgumentParser(description="CoTracker points -> UMI body origin per pack demo (solve2_<arm>.npz)")
    tAp.add_argument("pack")
    tAp.add_argument("hide_off", nargs="?", type=int, default=None,
                     help="hidden-tag run offset 0/1/2: yaw anchor from visible tags only -> solve2_<arm>_h<off>.npz")
    tArgs = tAp.parse_args(argv)
    for tDir in sorted(Path(tArgs.pack).iterdir()):
        if not (tDir / "tracks.npz").exists():
            continue
        solveDemoDir(tDir, tArgs.hide_off)
    return 0


if __name__ == "__main__":
    sys.exit(main())
