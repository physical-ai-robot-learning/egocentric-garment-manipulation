"""Episode-level quality table (was orbbec/v2b/cotrack/episode_quality.py, 10-25; whole episodes are kept or dropped, never cut) from
the gated CoTracker production run (solve3_<arm>.npz, conf > 0 = valid) of every pack demo, per arm, inside the idle-trim window:
  cov        valid fraction                         gap_s      longest invalid run [s]
  far10      valid frames > 10 cm from the >= 1-tag body point (fixed rig offset)   burst_s  longest > 10 cm run [s]
  swap_s     frames removed by the L/R-collision gate [s]     seed_ok   first 1 s of the RAW track within 10 cm of the tags (or no tags)
  grasp_ok   no invalid run >= 1.0 s (> fusion bridging range) overlapping +-1 s of a jaw open/close event of that arm's wrist camera
class  GOLD    both arms: cov >= 0.95, gap <= 1.0 s, far10 <= 0.05, burst <= 0.5 s, swap 0, seed_ok, grasp_ok
       REJECT  any arm: gap > 3 s, cov < 0.80, far10 > 0.20, burst > 2 s or swap > 1 s
       SILVER  otherwise
in_130 = the episode is in the v31speed fullmask set (EGO_DESKTOP/orbbec_v31speed_QA_all/episode_stats.csv, else
reports/orbbec_v31speed_episode_stats.csv).  Also reads <take>/<wrist cam>/umi_jaw.npz and the arm record.
    python -m laundry_tracking.cotrack.episode_quality <out.csv> <pack> [<pack> ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from laundry_common import paths

from ..constants import NS_PER_S
from ..fgo import arm_data
from ..intervals import runDurationsS, trueRuns

STATS = paths.DESKTOP / "orbbec_v31speed_QA_all/episode_stats.csv"
STATS_FALLBACK = paths.REPORTS / "orbbec_v31speed_episode_stats.csv"
ARMS = ("left", "right")
TAG_MATCH_NS = 3e7                 # a tag frame within 30 ms of the pack frame
FAR_M = 0.10
SEED_S = 1.0                       # seed check over the first second of the raw track
GRASP_GAP_S = 1.0                  # invalid runs this long are not bridged by the fusion
GRASP_NEAR_NS = 1e9                # +-1 s around a jaw event
# class thresholds
GOLD_COV, GOLD_GAP_S, GOLD_FAR10, GOLD_BURST_S = 0.95, 1.0, 0.05, 0.5
REJECT_GAP_S, REJECT_COV, REJECT_FAR10, REJECT_BURST_S, REJECT_SWAP_S = 3, 0.80, 0.20, 2, 1


def graspOk(invalid: np.ndarray, near: np.ndarray, t: np.ndarray, dt: float, minS: float = GRASP_GAP_S) -> bool:
    """False if an invalid run >= minS overlaps the +-1 s window of a jaw event (short gaps are bridged by the fusion)
    (was episode_quality.grasp_ok)"""
    for tS, tE in trueRuns(invalid):
        if (t[tE - 1] - t[tS]) / NS_PER_S + dt >= minS and near[tS:tE].any():
            return False
    return True


def armRow(demoDir: Path, take: str, window: arm_data.DemoWindow, arm: str, asg: dict) -> dict:
    """quality numbers of one arm of one demo (was episode_quality.arm_row)"""
    tZ = np.load(demoDir / f"solve3_{arm}.npz")
    tRaw = np.load(demoDir / f"solve2_{arm}.npz")
    tT = tZ["t"].astype(np.int64)
    tDt = float(np.median(np.diff(tT))) / NS_PER_S
    tIn = (tT >= window.mW0) & (tT <= window.mW1)
    tTw = tT[tIn]
    tValid = (tZ["conf"] > 0)[tIn]
    tC = tZ["c"][tIn]
    tGt = tZ["gt"][tIn]
    D = arm_data.loadArm(take, arm, withSlam=False)
    tK = np.clip(np.searchsorted(D["t"], tTw), 0, len(D["t"]) - 1)
    tTag = (D["ntags"][tK] >= 1) & (np.abs(D["t"][tK] - tTw) < TAG_MATCH_NS)
    tE = np.linalg.norm(tC - tGt, axis=1)
    tM = tValid & tTag & np.isfinite(tE)
    tFar = tM & (tE > FAR_M)
    tCr = np.where((tRaw["conf"] > 0)[:, None], tRaw["c"], np.nan)[tIn]
    tFirst = np.flatnonzero(np.isfinite(tCr[:, 0]))
    tS0 = tFirst[(tTw[tFirst] - tTw[tFirst[0]]) < SEED_S * NS_PER_S] if len(tFirst) else tFirst
    tEs = np.linalg.norm(tCr[tS0] - tGt[tS0], axis=1)[tTag[tS0]] if len(tS0) else np.zeros(0)
    tJaw = np.load(paths.TAG_RIG / take / asg["cam_of_arm"][arm] / "umi_jaw.npz")
    tJawEvents = tJaw["t_ns"][1:][np.diff(tJaw["state"].astype(int)) != 0]
    tJawEvents = tJawEvents[(tJawEvents >= window.mW0) & (tJawEvents <= window.mW1)]
    tNear = np.zeros(len(tTw), bool)
    for tX in tJawEvents:
        tNear |= np.abs(tTw - tX) <= GRASP_NEAR_NS
    return dict(cov=float(tValid.mean()), gap_s=max(runDurationsS(~tValid, tTw, tDt) or [0.0]),
                far10=float(tFar.sum() / max(tM.sum(), 1)), burst_s=max(runDurationsS(tFar, tTw, tDt) or [0.0]),
                swap_s=float(tZ["gate_lr"][tIn].sum() * tDt), seed_ok=bool(len(tEs) == 0 or np.median(tEs) < FAR_M),
                grasp_ok=graspOk(~tValid, tNear, tTw, tDt), n_tag=int(tM.sum()), n_grasp_ev=len(tJawEvents))


def qualityTable(out: str, packs: list[str]) -> pd.DataFrame:
    tSrc = STATS if STATS.exists() else STATS_FALLBACK
    tStats = pd.read_csv(tSrc, dtype={"take": str}) if tSrc.exists() else pd.DataFrame(columns=["take", "demo"])
    tIn130 = set(zip(tStats["take"], tStats["demo"].astype(int)))
    tRows = []
    tWindows = arm_data.loadTrimWindows()
    for tPack in packs:
        for tDir in sorted(Path(tPack).iterdir()):
            if not (tDir / "solve3_left.npz").exists():
                continue
            tMeta = json.loads((tDir / "meta.json").read_text())
            tTake, tDemo = tMeta["take"], int(tMeta["demo"])
            tWin = {tW.mDemo: tW for tW in tWindows[tTake]}[tDemo]
            tAsg = json.loads((paths.DERIVED / tTake / "assignment.json").read_text())
            tRow = dict(take=tTake, demo=tDemo, in_130=(tTake, tDemo) in tIn130, dur_s=round((tWin.mW1 - tWin.mW0) / NS_PER_S, 1))
            tArmRows = {tA: armRow(tDir, tTake, tWin, tA, tAsg) for tA in ARMS}
            for tA in tArmRows:
                for tKey, tVal in tArmRows[tA].items():
                    tRow[f"{tA[0].upper()}_{tKey}"] = tVal
            tGold = all(tX["cov"] >= GOLD_COV and tX["gap_s"] <= GOLD_GAP_S and tX["far10"] <= GOLD_FAR10 and tX["burst_s"] <= GOLD_BURST_S
                        and tX["swap_s"] == 0 and tX["seed_ok"] and tX["grasp_ok"] for tX in tArmRows.values())
            tReject = any(tX["gap_s"] > REJECT_GAP_S or tX["cov"] < REJECT_COV or tX["far10"] > REJECT_FAR10
                          or tX["burst_s"] > REJECT_BURST_S or tX["swap_s"] > REJECT_SWAP_S for tX in tArmRows.values())
            tRow["class"] = "GOLD" if tGold else ("REJECT" if tReject else "SILVER")
            tWhy = []
            for tA in tArmRows:
                tX = tArmRows[tA]
                tL = tA[0].upper()
                for tCond, tTxt in ((tX["cov"] < GOLD_COV, f"{tL} cov {tX['cov']:.2f}"),
                                    (tX["gap_s"] > GOLD_GAP_S, f"{tL} gap {tX['gap_s']:.1f}s"),
                                    (tX["far10"] > GOLD_FAR10, f"{tL} far10 {tX['far10']:.2f}"),
                                    (tX["burst_s"] > GOLD_BURST_S, f"{tL} burst {tX['burst_s']:.1f}s"),
                                    (tX["swap_s"] > 0, f"{tL} swap {tX['swap_s']:.1f}s"), (not tX["seed_ok"], f"{tL} seed"),
                                    (not tX["grasp_ok"], f"{tL} grasp-gap")):
                    if tCond:
                        tWhy.append(tTxt)
            tRow["why"] = "; ".join(tWhy)
            tRows.append(tRow)
    tDf = pd.DataFrame(tRows).sort_values(["take", "demo"])
    tDf.to_csv(out, index=False)
    print(tDf["class"].value_counts().to_dict(), "| in_130:", tDf[tDf["in_130"]]["class"].value_counts().to_dict())
    return tDf


def main(argv: list[str] | None = None) -> int:
    tAp = argparse.ArgumentParser(description="GOLD / SILVER / REJECT episode table from gated CoTracker packs")
    tAp.add_argument("out")
    tAp.add_argument("packs", nargs="*")
    tArgs = tAp.parse_args(argv)
    qualityTable(tArgs.out, tArgs.packs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
