"""trajectory_fusion_v1 (was orbbec/v2b/fusion/trajectory_fusion_v1.py, 10-24): whole-demo batch factor graph for the tag-point
trajectory p(t) (= qa "TCP": body centre - R o) on a 20 Hz grid.

Rotation (all variants): Mahony AHRS (rest-period gyro bias), one yaw anchor at the demo's first >= 2-tag frame, per-demo IMU<->Orbbec
lag (imu_motion.syncLag on the CoTracker track).  Position = sparse linear IRLS (per-axis systems, shared Cauchy weights on the 3D
residual):
  CT    CoTracker v2 body centre (cotrack point solver output) -> y = c - R o;  sigma by conf (2: ct_hi, 1: ct_lo), Cauchy, + Orbbec bias b
  IMU   second difference = (R f(t + lag) - mean) DT^2, sigma imu_acc, Cauchy  (gravity removed by the demo mean, as the stitch v3on)
  TAG   p = p_tag (non-hidden frames, >= 1 tag), sigma tag2 / tag1 inflated by the tag-vs-AHRS rotation disagreement (quality), Cauchy
  SMOOTH second difference prior, acc_info near observations, acc_gap in unobserved spans (bridges gaps)
  SLAM  (variant D) wrist MASt3R relative displacement, per-segment scale -- needs SLAM for the take (not run for 155004)
Output is cropped to [first, last] observed node (no extrapolation).  Suffix g (e.g. Cg) = IMU factor only on unsupported nodes.
Variants: R = raw CoTracker y (no fusion) | A = CT + smooth (b = 0) | B = A + IMU | C = B + TAG (+ constant 3D Orbbec bias) | D = C + SLAM
          E = C + tag rotation correction (world-frame rotvec knots every rot_knot_s, fit to non-hidden R_tag) + random-walk bias knots.
Variant names are data: they name the output folders (fus_Cg_g = production, fus_R_g = raw) read by every later tool.
Hidden-tag protocol: runs h0/h1/h2 drop every tag in blocks block%3 == off (whole_demo_stitch.hiddenMask on the take's tag grid),
h-1 = production.  --strict (no leakage): nothing in run h<off> is estimated from hidden tags -- yaw anchor (here and in
solve2_<arm>_h<off>.npz from the point solver with <off>), tag position / rotation factors, bias; fixed body offset from dev take 123747.
Outputs FUSION_WORK/stitch/qa/fus_<V><suffix>/{take}_{arm}_dNN_h{off}.npz (t_ns, p_B, R_B_tcp, + info per node, bias, lag_ns); prints
the per-demo lag / node counts.
    python -m laundry_tracking.fusion.trajectory_fusion <ctpack dir> <variants e.g. A,B,C,E> [--demos 0,1] [--strict --suffix _s] [--ct solve3]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.spatial.transform import Rotation as R

from laundry_common import paths

from .. import imu_motion
from .. import workdirs
from ..constants import NS_PER_MS, NS_PER_S
from ..fgo import arm_data
from ..fgo import pose_smoother
from ..whole_demo_stitch import hiddenMask

NODE_DT_S = 0.05
ARMS = ("left", "right")
HIDE_RUNS = (-1, 0, 1, 2)          # production run first, then the three hidden-tag runs
DEMO_TAIL_S = 2.0                  # yaw anchor searched up to the window end + 2 s; nodes span the window +- 2 s
MIN_LAG_SAMPLES = 30
IMU_SMOOTH_TAPS = 5
IRLS_ITERS = 8
IRLS_REGULARIZER = 1e-10
MIN_ROT_TAGS = 5                   # rotation correction needs >= 5 non-hidden tag rotations
ROOT_PRIOR_SIGMA = 1.0             # the first rotation knot is held at 0 with this sigma
ROT_SINGLE_TAG_FACTOR = 2.0        # single-tag rotation sigma = 2 x
CONF_HIGH = 2


class FusionError(RuntimeError):
    """A requested variant or hidden-tag run cannot be computed from the inputs present."""


@dataclass(frozen=True)
class FusionSigmas:
    """factor sigmas and knot spacings (was the SIG dict)"""
    mCtHi: float = 0.010
    mCtLo: float = 0.025
    mTag2: float = 0.004
    mTag1: float = 0.008
    mTagRotDeg: float = 5.0
    mImuAcc: float = 2.0
    mAccInfo: float = 30.0
    mAccGap: float = 3.0
    mCauchy: float = 3.0
    mBiasPrior: float = 0.10
    mBiasRw: float = 0.010
    mBiasKnotS: float = 4.0
    mRotKnotS: float = 2.0
    mRotTagDeg: float = 3.0
    mRotRwDeg: float = 1.0


FUSION_SIGMAS = FusionSigmas()


def irls(rows: list, nv: int, iters: int = IRLS_ITERS, cauchy: float = FUSION_SIGMAS.mCauchy):
    """rows: list of (cols, coefs, target(3,), sigma, robust) -> (x (nv,3), robust weights); one sparse system per axis, shared
    weights on the 3D residual"""
    tRi, tCi, tVi = [], [], []
    for tR, (tCols, tCf, _, _, _) in enumerate(rows):
        tRi += [tR] * len(tCols)
        tCi += list(tCols)
        tVi += list(tCf)
    tA = sp.csr_matrix((tVi, (tRi, tCi)), shape=(len(rows), nv))
    tB = np.array([tR[2] for tR in rows])
    tS = np.array([tR[3] for tR in rows])
    tRb = np.array([tR[4] for tR in rows])
    tW = 1 / tS ** 2
    for _ in range(iters):
        tLu = splu((tA.T @ sp.diags(tW) @ tA).tocsc() + sp.identity(nv) * IRLS_REGULARIZER)
        tX = np.stack([tLu.solve(tA.T @ (tW * tB[:, tQ])) for tQ in range(3)], 1)
        tE = np.linalg.norm(tA @ tX - tB, axis=1) / tS
        tW = np.where(tRb, 1 / (1 + (tE / cauchy) ** 2), 1.0) / tS ** 2
    return tX, np.where(tRb, tW * tS ** 2, 1.0)


def lin(tn: np.ndarray, tq):
    """node index i and weight a such that p(tq) = (1 - a) P[i] + a P[i + 1]"""
    tU = np.clip((tq - tn[0]) / (NODE_DT_S * NS_PER_S), 0, len(tn) - 1 - 1e-9)
    tI = np.floor(tU).astype(int)
    return tI, tU - tI


def rotationCorrection(tn: np.ndarray, Ra: R, tq, RqTag: R, RqAhrs: R, sigQ) -> R:
    """world-frame rotvec correction knots (piecewise linear) fit to R_tag R_ahrs^-1 at non-hidden tag frames -> corrected R on tn
    (was trajectory_fusion_v1.rot_correction)"""
    tSig = FUSION_SIGMAS
    tKnots = np.arange(tn[0], tn[-1] + tSig.mRotKnotS * NS_PER_S, tSig.mRotKnotS * NS_PER_S)
    tK = len(tKnots)
    tRows = []
    if len(tq) >= MIN_ROT_TAGS:
        tD = (RqTag * RqAhrs.inv()).as_rotvec()
        tU = np.clip((tq - tKnots[0]) / (tKnots[1] - tKnots[0] if tK > 1 else 1), 0, tK - 1 - 1e-9)
        tI = np.floor(tU).astype(int)
        tA = tU - tI
        for tJ in range(len(tq)):
            tRows.append(([tI[tJ], min(tI[tJ] + 1, tK - 1)], [1 - tA[tJ], tA[tJ]], tD[tJ], sigQ[tJ], True))
    for tJ in range(tK - 1):
        tRows.append(([tJ, tJ + 1], [1.0, -1.0], np.zeros(3), np.radians(tSig.mRotRwDeg), False))
    tRows.append(([0], [1.0], np.zeros(3), ROOT_PRIOR_SIGMA, False))
    tX, _ = irls(tRows, tK)
    tDn = np.stack([np.interp(tn, tKnots, tX[:, tQ]) for tQ in range(3)], 1) if tK > 1 else np.repeat(tX, len(tn), 0)
    return R.from_rotvec(tDn) * Ra


def fuse(variant: str, D: dict, tn: np.ndarray, ct: dict, Rn: R, fB: np.ndarray, tag: dict, hidT: np.ndarray):
    """one run -> (P (N,3), bias knots, info per node, [first, last] observed node) or None; ct = dict(t, y, conf),
    tag = dict(t, p, n, q_deg) for this arm, hidT = hidden mask on the tag frames"""
    tSig = FUSION_SIGMAS
    tN = len(tn)
    tBase = variant[0]
    tUseBias = tBase in ("C", "D", "E")
    tRwBias = tBase == "E"
    tImuGap = variant.endswith("g")
    tBiasKnots = np.arange(tn[0], tn[-1] + tSig.mBiasKnotS * NS_PER_S, tSig.mBiasKnotS * NS_PER_S) if tRwBias else tn[:1]
    tK = len(tBiasKnots) if tUseBias else 0
    tRows = []
    tInfo = np.zeros(tN, bool)
    tM = ct["conf"] > 0
    tI, tA = lin(tn, ct["t"][tM])
    if tK > 1:
        tUb = np.clip((ct["t"][tM] - tBiasKnots[0]) / (tBiasKnots[1] - tBiasKnots[0]), 0, tK - 1 - 1e-9)
        tIb = np.floor(tUb).astype(int)
        tAb = tUb - tIb
    for tJ in range(tM.sum()):
        tCols, tCf = [tI[tJ], tI[tJ] + 1], [1 - tA[tJ], tA[tJ]]
        if tK == 1:
            tCols, tCf = tCols + [tN], tCf + [1.0]
        elif tK > 1:
            tCols, tCf = tCols + [tN + tIb[tJ], tN + min(tIb[tJ] + 1, tK - 1)], tCf + [1 - tAb[tJ], tAb[tJ]]
        tRows.append((tCols, tCf, ct["y"][tM][tJ], tSig.mCtHi if ct["conf"][tM][tJ] == CONF_HIGH else tSig.mCtLo, True))
        tInfo[tI[tJ]] = tInfo[tI[tJ] + 1] = True
    if tBase in ("C", "D", "E"):
        tMt = (tag["n"] >= 1) & ~hidT
        tI, tA = lin(tn, tag["t"][tMt])
        tS0 = np.where(tag["n"][tMt] >= 2, tSig.mTag2, tSig.mTag1) * (1 + (np.nan_to_num(tag["q_deg"][tMt],
                                                                                         nan=0.0) / tSig.mTagRotDeg) ** 2)
        for tJ in range(tMt.sum()):
            tRows.append(([tI[tJ], tI[tJ] + 1], [1 - tA[tJ], tA[tJ]], tag["p"][tMt][tJ], tS0[tJ], True))
            tInfo[tI[tJ]] = tInfo[tI[tJ] + 1] = True
    tAcc = fB - fB.mean(0)
    for tNode in range(1, tN - 1):
        tSa = (tSig.mAccInfo if tInfo[tNode - 1:tNode + 2].any() else tSig.mAccGap) * NODE_DT_S * NODE_DT_S
        tRows.append(([tNode - 1, tNode, tNode + 1], [1.0, -2.0, 1.0], np.zeros(3), tSa, False))
        if tBase != "A" and not (tImuGap and tInfo[tNode - 1:tNode + 2].any()):
            tRows.append(([tNode - 1, tNode, tNode + 1], [1.0, -2.0, 1.0], tAcc[tNode] * NODE_DT_S * NODE_DT_S,
                          tSig.mImuAcc * NODE_DT_S * NODE_DT_S, True))
    if tK:
        tRows.append(([tN], [1.0], np.zeros(3), tSig.mBiasPrior, False))
    for tJ in range(tK - 1):
        tRows.append(([tN + tJ, tN + tJ + 1], [1.0, -1.0], np.zeros(3), tSig.mBiasRw, False))
    if not tUseBias and not tM.any():
        return None
    tX, _ = irls(tRows, tN + tK)
    return tX[:tN], (tX[tN:] if tK else np.zeros((0, 3))), tInfo, np.flatnonzero(tInfo)[[0, -1]]


def fuseDemoDir(demoDir: Path, variants: list[str], strict: bool, suffix: str, ctName: str) -> None:
    """every variant x arm x hidden-tag run of one pack demo dir"""
    tSig = FUSION_SIGMAS
    tMeta = json.loads((demoDir / "meta.json").read_text())
    tTake, tDemo = tMeta["take"], int(tMeta["demo"])
    tWindows = arm_data.loadTrimWindows()[tTake]
    tWin = {tW.mDemo: tW for tW in tWindows}[tDemo]
    tAsg = json.loads((paths.DERIVED / tTake / "assignment.json").read_text())
    for tArm in ARMS:
        D = arm_data.loadArm(tTake, tArm, withSlam=False)
        pose_smoother.prepGyro(D)
        D["b_rest"] = imu_motion.restBias(D, tWindows)
        tT = D["t"]
        tOffset = workdirs.bodyOffsetM(tAsg["rig_of_arm"][tArm])
        for tOff in HIDE_RUNS:
            # leakage guard: in hidden run h<off> nothing is estimated from hidden-block tags -- CoTracker input from the solve2 run whose
            # yaw anchor was visible-only (solve2_<arm>_h<off>.npz), our own yaw anchor visible-only, tag factors / rot correction visible-only
            tHidFull = hiddenMask(tT, tOff) if tOff >= 0 else np.zeros(len(tT), bool)
            tFile = demoDir / (f"{ctName}_{tArm}.npz" if tOff < 0 else f"{ctName}_{tArm}_h{tOff}.npz")
            if not tFile.exists():
                if strict:
                    raise FusionError(f"missing {tFile} (run cotrack/solve2.py / gate.py <pack> {tOff})")
                tFile = demoDir / f"{ctName}_{tArm}.npz"
            tZ = np.load(tFile)
            tTct, tCct, tConf = tZ["t"].astype(np.int64), tZ["c"], tZ["conf"].astype(int)
            tTag2 = (D["ntags"] >= 2) & (D["n_tagrot"] >= 2) & np.isfinite(D["R_tag"][:, 0, 0]) & ~tHidFull
            tInWin = (tT >= tWin.mA0) & (tT <= tWin.mW1 + DEMO_TAIL_S * NS_PER_S)
            tAnc = np.flatnonzero(tInWin & tTag2)
            if not len(tAnc):
                tJ = np.flatnonzero(tTag2)
                tAnc = tJ[[np.argmin(np.abs(tT[tJ] - tWin.mA0))]]
            tTa, tRanc = tT[tAnc[0]], R.from_matrix(D["R_tag"][tAnc[0]])
            if np.isfinite(tCct[:, 0]).sum() > MIN_LAG_SAMPLES:
                tLag, tNcc = imu_motion.syncLag(D, tTct, tCct, np.arange(len(tTct)), tTa, tRanc)
            else:
                tLag, tNcc = 0, 0.0
            tT0 = max(tTct[0], tWin.mW0 - int(DEMO_TAIL_S * NS_PER_S))
            tT1 = min(tTct[-1], tWin.mW1 + int(DEMO_TAIL_S * NS_PER_S))
            tn = np.arange(tT0, tT1 + 1, int(NODE_DT_S * NS_PER_S))
            tRa = imu_motion.ahrsBase(D, tn + tLag, tTa + tLag, tRanc)
            tF = np.stack([np.interp(tn + tLag, D["imu_t"], D["acc"][:, tQ]) for tQ in range(3)], 1)
            tK5 = np.ones(IMU_SMOOTH_TAPS) / IMU_SMOOTH_TAPS
            tF = np.stack([np.convolve(tF[:, tQ], tK5, mode="same") for tQ in range(3)], 1)
            tMt = (tT >= tn[0]) & (tT <= tn[-1])
            tRq = imu_motion.ahrsBase(D, tT[tMt] + tLag, tTa + tLag, tRanc)
            tOkR = (D["n_tagrot"][tMt] >= 1) & np.isfinite(D["R_tag"][tMt][:, 0, 0])
            tQd = np.full(tMt.sum(), np.nan)
            if tOkR.any():
                tQd[tOkR] = np.degrees((R.from_matrix(D["R_tag"][tMt][tOkR]).inv() * tRq[tOkR]).magnitude())
            tTag = dict(t=tT[tMt], p=D["p_tag"][tMt], n=D["ntags"][tMt], q_deg=tQd)
            tHidT = tHidFull[tMt]
            for tVar in variants:
                tName = f"fus_{tVar}{suffix}"
                (workdirs.STITCH_QA_DIR / tName).mkdir(parents=True, exist_ok=True)
                tOut = workdirs.qaRunPath(tName, tTake, tArm, tDemo, tOff)
                tRn = tRa
                if tVar[0] == "E":
                    tOk = tOkR & ~tHidT & (tTag["n"] >= 1)
                    tSq = np.radians(tSig.mRotTagDeg) * np.where(D["n_tagrot"][tMt][tOk] >= 2, 1.0, ROT_SINGLE_TAG_FACTOR)
                    tRn = rotationCorrection(tn, tRa, tT[tMt][tOk], R.from_matrix(D["R_tag"][tMt][tOk]), tRq[tOk], tSq)
                tRct = tRn[np.clip(np.round((tTct - tn[0]) / (NODE_DT_S * NS_PER_S)).astype(int), 0, len(tn) - 1)]
                tInRange = (tTct >= tn[0]) & (tTct <= tn[-1]) & np.isfinite(tCct[:, 0])
                tCt = dict(t=tTct[tInRange], y=tCct[tInRange] - tRct[tInRange].apply(tOffset),
                           conf=np.where(np.isfinite(tCct[tInRange, 0]), tConf[tInRange], 0))
                if tVar == "R":                       # raw CoTracker (no fusion; qa_eval interpolates linearly across gaps)
                    tMm = tCt["conf"] > 0
                    np.savez(tOut, t_ns=tCt["t"][tMm], p_B=tCt["y"][tMm], R_B_tcp=tRct[tInRange][tMm].as_matrix())
                    continue
                tResult = fuse(tVar, D, tn, tCt, tRn, tRn.apply(tF), tTag, tHidT)
                if tResult is None:
                    continue
                tP, tBias, tInfo, (tLo, tHi) = tResult
                tSl = slice(tLo, tHi + 1)                 # no extrapolation beyond the first / last observation
                np.savez(tOut, t_ns=tn[tSl], p_B=tP[tSl], R_B_tcp=tRn.as_matrix()[tSl], info=tInfo[tSl], bias=tBias, lag_ns=tLag)
            if tOff < 0:
                print(f"{tTake} d{tDemo:02d} {tArm[0]} lag {tLag / NS_PER_MS:.0f} ms ncc {tNcc:.2f} | nodes {len(tn)} "
                      f"ct {int((tCt['conf'] > 0).sum())} tags {int((tTag['n'] >= 1).sum())}", flush=True)


def _buildParser() -> argparse.ArgumentParser:
    tAp = argparse.ArgumentParser()
    tAp.add_argument("pack")
    tAp.add_argument("variants")
    tAp.add_argument("--demos", default="")
    tAp.add_argument("--strict", action="store_true")
    tAp.add_argument("--suffix", default="")
    tAp.add_argument("--ct", default="solve2", help="CoTracker input: solve2 | solve3 (gated, cotrack/gate.py)")
    return tAp


def main(argv: list[str] | None = None) -> int:
    tArgs = _buildParser().parse_args(argv)
    tVariants = tArgs.variants.split(",")
    tDemos = {int(tX) for tX in tArgs.demos.split(",") if tX}
    try:
        if "D" in tVariants:
            raise FusionError("variant D needs wrist SLAM segments for the take (none for 155004 yet)")
        for tDir in sorted(Path(tArgs.pack).iterdir()):
            if not (tDir / f"{tArgs.ct}_left.npz").exists():
                continue
            if tDemos and int(json.loads((tDir / "meta.json").read_text())["demo"]) not in tDemos:
                continue
            fuseDemoDir(tDir, tVariants, tArgs.strict, tArgs.suffix, tArgs.ct)
    except FusionError as tErr:
        print(tErr, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
