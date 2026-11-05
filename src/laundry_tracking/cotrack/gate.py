"""CoTracker v2 validity gates (was orbbec/v2b/cotrack/gate.py, 10-25): solve2 output -> solve3_<arm>[_h<off>].npz (same keys + gate
masks; conf set to 0 where a gate fires).  Targets the three right-arm failure modes of the 10-24 held-out / new-20 review (thresholds
fixed a priori, checked on dev 155004 / 123747 only):
  G1 imu   IMU-motion consistency: track acceleration (imu_motion.trackAcc) vs IMU linear acceleration in the base frame (AHRS
           rotation, per-demo IMU<->Orbbec lag from the track, gravity = median), centred-window NCC (WIN_S, energy floor E_MIN).
           A window with NCC < NCC_BAD is inconsistent (the IMU moves and the track does not = hand left the view / stale points; or
           the track moves differently = hair, cloth).  Bad runs >= MIN_BAD_S, dilated by half a window -> invalid, UNLESS >= TAG_N
           visible tag frames in the run agree with the track (median < TAG_BAD): 10-25 audit, tag-checked G1 cuts were either
           325-600 mm wrong (true failures) or 40-70 mm = as good as the kept frames (123747: IMU<->track lag wanders -240..-40 ms,
           NCC 0.55-0.78 at best lag).
  G2 lr    L/R collision: both arms valid and |c_L - c_R| < LR_MIN -> per collision run (dilated LR_PAD) the arm whose own visible tags
           disagree (> TAG_BAD) or, without tag evidence, the lower mean G1 NCC loses (invalid); no evidence either way -> both invalid.
  G3 tag   visible (non-hidden) >= 1-tag body points vs the track: > TAG_BAD on >= TAG_N frames inside a TAG_WIN window -> invalid.
In hidden runs only non-hidden tags are used (yaw anchor too), as everywhere else.
    python -m laundry_tracking.cotrack.gate <pack> [hide_off]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

from laundry_common import paths

from .. import imu_motion
from .. import workdirs
from ..constants import NS_PER_S
from ..fgo import arm_data
from ..fgo import pose_smoother
from ..intervals import trueRuns
from ..whole_demo_stitch import hiddenMask

# Gate thresholds (dict: stored verbatim as JSON in every solve3 file, key gate_params).
GATE_PARAMS = dict(WIN_S=1.0, E_MIN=0.5, NCC_BAD=0.2, MIN_BAD_S=0.4, LR_MIN=0.08, LR_PAD_S=0.3, TAG_BAD=0.10, TAG_N=3,
                   TAG_WIN_S=1.0)
ARMS = ("left", "right")
DEMO_TAIL_S = 2.0                  # yaw anchor searched up to the window end + 2 s
MIN_LAG_SAMPLES = 30
UNDECIDED_NCC = 9.0                # placeholder NCC of undecided windows before interpolation to the pack frames
DECIDED_MIN = 0.5
TAG_MATCH_NS = 3e7                 # a tag frame within 30 ms of the pack frame
NCC_TIE = 0.15                     # collision without tags: the arm with the clearly lower mean NCC (> 0.15 apart) loses


def imuNcc(D: dict, t: np.ndarray, c: np.ndarray, window: arm_data.DemoWindow, hid: np.ndarray):
    """per Orbbec-pack frame: G1 window NCC (9.0 where undecided), decided mask, and the AHRS rotation on those frames
    (was gate.imu_ncc)"""
    tTt = D["t"]
    tTag2 = (D["ntags"] >= 2) & (D["n_tagrot"] >= 2) & np.isfinite(D["R_tag"][:, 0, 0]) & ~hid
    tWin = (tTt >= window.mA0) & (tTt <= window.mW1 + DEMO_TAIL_S * NS_PER_S)
    tAnc = np.flatnonzero(tWin & tTag2)
    if not len(tAnc):
        tJ = np.flatnonzero(tTag2)
        tAnc = tJ[[np.argmin(np.abs(tTt[tJ] - window.mA0))]]
    tTa, tRa = tTt[tAnc[0]], R.from_matrix(D["R_tag"][tAnc[0]])
    tValid = np.isfinite(c[:, 0])
    tLag, _ = imu_motion.syncLag(D, t, c, np.arange(len(t)), tTa, tRa) if tValid.sum() > MIN_LAG_SAMPLES else (0, 0.0)
    tG = imu_motion.motionGrid(t[0], t[-1])
    tRg = imu_motion.ahrsBase(D, tG + tLag, tTa + tLag, tRa)
    tF = np.stack([np.interp(tG + tLag, D["imu_t"], D["acc"][:, tK]) for tK in range(3)], 1)
    tAi = tRg.apply(tF)
    tAi -= np.median(tAi, 0)
    tAi = gaussian_filter1d(tAi, imu_motion.SMOOTH_SIGMA_S * imu_motion.GRID_HZ, axis=0, mode="nearest")
    tAo = imu_motion.trackAcc(t, c, tG)
    tNcc = imu_motion.windowNcc(tAo, tAi, winS=GATE_PARAMS["WIN_S"], eMin=GATE_PARAMS["E_MIN"])
    tRk = imu_motion.ahrsBase(D, t + tLag, tTa + tLag, tRa).as_matrix()
    return np.interp(t, tG, np.nan_to_num(tNcc, nan=UNDECIDED_NCC)), np.interp(t, tG,
                                                                               np.isfinite(tNcc).astype(float)) > DECIDED_MIN, tRk


def gateDemoDir(demoDir: Path, hideOffset: int | None) -> None:
    """both arms of one pack demo dir: solve2 -> solve3 with the three gates"""
    G = GATE_PARAMS
    tFiles = {tA: demoDir / (f"solve2_{tA}.npz" if hideOffset is None else f"solve2_{tA}_h{hideOffset}.npz") for tA in ARMS}
    if not all(tF.exists() for tF in tFiles.values()):
        return
    tMeta = json.loads((demoDir / "meta.json").read_text())
    tTake, tDemo = tMeta["take"], int(tMeta["demo"])
    tWindows = arm_data.loadTrimWindows()[tTake]
    tWin = {tW.mDemo: tW for tW in tWindows}[tDemo]
    tAsg = json.loads((paths.DERIVED / tTake / "assignment.json").read_text())
    Z = {tA: dict(np.load(tFiles[tA])) for tA in tFiles}
    tT = Z["left"]["t"].astype(np.int64)
    tN = len(tT)
    tDt = np.median(np.diff(tT)) / NS_PER_S
    tBad, tNcc, tTagD = {}, {}, {}
    for tA in ARMS:
        D = arm_data.loadArm(tTake, tA, withSlam=False)
        pose_smoother.prepGyro(D)
        D["b_rest"] = imu_motion.restBias(D, tWindows)
        tHid = hiddenMask(D["t"], hideOffset) if hideOffset is not None else np.zeros(len(D["t"]), bool)
        tC = np.where((Z[tA]["conf"] > 0)[:, None], Z[tA]["c"], np.nan)
        tNc, tDec, tRk = imuNcc(D, tT, tC, tWin, tHid)
        tNcc[tA] = np.where(tDec, tNc, np.nan)
        tK = np.clip(np.searchsorted(D["t"], tT), 0, len(D["t"]) - 1)
        tVis = (D["ntags"][tK] >= 1) & ~tHid[tK] & (np.abs(D["t"][tK] - tT) < TAG_MATCH_NS)
        tBody = D["p_tag"][tK] + np.einsum("nij,j->ni", tRk, workdirs.bodyOffsetM(tAsg["rig_of_arm"][tA]))
        tDd = np.linalg.norm(tC - tBody, axis=1)
        tDis = tVis & np.isfinite(tDd) & (tDd > G["TAG_BAD"])
        tTagD[tA] = np.where(tVis & np.isfinite(tDd), tDd, np.nan)
        tB1 = np.zeros(tN, bool)                                          # G1 (vetoed where visible tags agree with the track)
        tH = int(G["WIN_S"] / 2 / tDt)
        for tS, tE in trueRuns(tDec & (tNc < G["NCC_BAD"])):
            if (tE - tS) * tDt < G["MIN_BAD_S"]:
                continue
            tS2, tE2 = max(tS - tH, 0), tE + tH
            tTd = tTagD[tA][tS2:tE2]
            tTd = tTd[np.isfinite(tTd)]
            if len(tTd) >= G["TAG_N"] and np.median(tTd) < G["TAG_BAD"]:
                continue
            tB1[tS2:tE2] = True
        tB3 = np.zeros(tN, bool)                                          # G3
        tH3 = int(G["TAG_WIN_S"] / 2 / tDt)
        for tJ in np.flatnonzero(tDis):
            if tDis[max(tJ - tH3, 0):tJ + tH3 + 1].sum() >= G["TAG_N"]:
                tB3[max(tJ - tH3, 0):tJ + tH3 + 1] = True
        tBad[tA] = dict(imu=tB1, tag=tB3, lr=np.zeros(tN, bool))
    tCl, tCr = [np.where((Z[tA]["conf"] > 0)[:, None], Z[tA]["c"], np.nan) for tA in ARMS]
    tColl = np.linalg.norm(tCl - tCr, axis=1) < G["LR_MIN"]                # G2
    tHp = int(G["LR_PAD_S"] / tDt)
    for tS, tE in trueRuns(tColl):
        tS, tE = max(tS - tHp, 0), min(tE + tHp, tN)
        tEv = {tA: (np.nanmedian(tTagD[tA][tS:tE]) if np.isfinite(tTagD[tA][tS:tE]).any() else np.nan,
                    np.nanmean(tNcc[tA][tS:tE]) if np.isfinite(tNcc[tA][tS:tE]).any() else np.nan) for tA in ARMS}
        tTl, tTr = tEv["left"][0], tEv["right"][0]
        if np.isfinite(tTl) or np.isfinite(tTr):
            tLose = [tA for tA in ARMS if np.isfinite(tEv[tA][0]) and tEv[tA][0] > G["TAG_BAD"]] or \
                    ([] if (np.isfinite(tTl) and np.isfinite(tTr)) else [tA for tA in ARMS if not np.isfinite(tEv[tA][0])])
        elif np.isfinite(tEv["left"][1]) and np.isfinite(tEv["right"][1]) and abs(tEv["left"][1] - tEv["right"][1]) > NCC_TIE:
            tLose = ["left" if tEv["left"][1] < tEv["right"][1] else "right"]
        else:
            tLose = ["left", "right"]
        for tA in tLose:
            tBad[tA]["lr"][tS:tE] = True
    for tA in ARMS:
        tB = tBad[tA]
        tAny = tB["imu"] | tB["tag"] | tB["lr"]
        tZ = Z[tA]
        tZ["conf"] = np.where(tAny, 0, tZ["conf"])
        tZ["gate_imu"] = tB["imu"]
        tZ["gate_tag"] = tB["tag"]
        tZ["gate_lr"] = tB["lr"]
        tZ["ncc"] = tNcc[tA]
        tZ["gate_params"] = json.dumps(G)
        np.savez(demoDir / (f"solve3_{tA}.npz" if hideOffset is None else f"solve3_{tA}_h{hideOffset}.npz"), **tZ)
        print(f"{demoDir.name} {tA[0]} h{hideOffset if hideOffset is not None else -1}: cut imu {tB['imu'].mean() * 100:4.1f}% "
              f"tag {tB['tag'].mean() * 100:4.1f}% lr {tB['lr'].mean() * 100:4.1f}% | valid {np.mean(Z[tA]['conf'] > 0) * 100:4.1f}%", flush=True)


def main(argv: list[str] | None = None) -> int:
    tAp = argparse.ArgumentParser(description="CoTracker validity gates: solve2 -> solve3 per pack demo")
    tAp.add_argument("pack")
    tAp.add_argument("hide_off", nargs="?", type=int, default=None)
    tArgs = tAp.parse_args(argv)
    for tDir in sorted(Path(tArgs.pack).iterdir()):
        gateDemoDir(tDir, tArgs.hide_off)
    return 0


if __name__ == "__main__":
    sys.exit(main())
