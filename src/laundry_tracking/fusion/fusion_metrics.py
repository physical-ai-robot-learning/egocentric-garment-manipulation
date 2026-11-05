"""Per-arm fusion ablation table (was orbbec/v2b/fusion/fusion_metrics.py, 10-24) from FUSION_WORK/stitch/qa/<var> runs +
stitch/qa/eval_<tag>.csv (qa_eval hidden-tag e32):
  raw_cov     CoTracker conf > 0 fraction of the trimmed window (Orbbec frames);  cont_cov = window fraction inside the run's time span
  sup_cov     nodes within 0.25 s of any used observation (production run h-1; raw var = raw_cov);  gap_s = longest unsupported span
  e32         n-weighted mean over demos of the qa_eval hidden-block e32 median / p90 [mm]
  abs         hidden-block absolute |p - p_tag| (>= 2 tags) median / p90 [mm];  far10 = rate > 10 cm;  burst_s = longest > 10 cm run
  gapbr       gap-bridge error: hidden >= 1-tag frames inside CoTracker gaps >= 0.2 s (+-0.15 s, seed start excluded), |e - demo median e|
              [mm] max / med
Output: stitch/qa/fusion_metrics_<tag>.csv (per demo) + the printed per-arm / variant aggregate.
    python -m laundry_tracking.fusion.fusion_metrics <ctpack> <eval csv tag> VAR1,VAR2,...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .. import workdirs
from ..constants import MM_PER_M, NS_PER_S, ORBBEC_FRAME_S
from ..fgo import arm_data
from ..intervals import runDurationsS, trueRuns
from ..whole_demo_stitch import hiddenMask

ARMS = ("left", "right")
HIDE_RUNS = (0, 1, 2)
GRID_NS = int(5e7)                 # 50 ms grid of the continuity coverage
GAP_MIN_S = 0.2                    # CoTracker gaps this long are bridge tests
GAP_PAD_NS = 1.5e8                 # +-0.15 s around a gap
MIN_OUTSIDE_FOR_BIAS = 10          # demo bias removed only with > 10 hidden frames outside the gaps
FAR_M = 0.10
DISPLAY_WIDTH = 220


def interp(t, p, tq) -> np.ndarray:
    return np.stack([np.interp(tq, t, p[:, tAx]) for tAx in range(3)], 1)


def _weightedAverage(x, w):
    tM = np.isfinite(x)
    return np.average(x[tM], weights=w[tM]) if tM.any() else np.nan


def _aggregate(g) -> pd.Series:
    tW = g["n"].clip(lower=1)
    tWa = g["n_abs"].clip(lower=1)
    return pd.Series(dict(raw_cov=g["raw_cov"].mean(), cont_cov=g["cont_cov"].mean(), sup_cov=g["sup_cov"].mean(),
                          gap_s=g["gap_s"].max(), e32_med=_weightedAverage(g["e32_med"], tW),
                          e32_p90=_weightedAverage(g["e32_p90"], tW), abs_med=_weightedAverage(g["abs_med"], tWa),
                          abs_p90=_weightedAverage(g["abs_p90"], tWa), far10=_weightedAverage(g["far10"], tWa),
                          burst_s=g["burst_s"].max(), gapbr_max=g["rejoin"].max(), gapbr_med=g["gapbr_med"].median(),
                          n_gap=g["n_gap"].sum()))


def fusionMetrics(pack: str, tag: str, variants: list[str]) -> pd.DataFrame:
    tEval = pd.read_csv(workdirs.STITCH_QA_DIR / f"eval_{tag}.csv")
    tRows = []
    tWindows = arm_data.loadTrimWindows()
    for tDir in sorted(Path(pack).iterdir()):
        if not (tDir / "solve2_left.npz").exists():
            continue
        tMeta = json.loads((tDir / "meta.json").read_text())
        tTake, tDemo = tMeta["take"], int(tMeta["demo"])
        tWin = {tW.mDemo: tW for tW in tWindows[tTake]}[tDemo]
        for tArm in ARMS:
            D = arm_data.loadArm(tTake, tArm, withSlam=False)
            tT = D["t"]
            tInWin = (tT >= tWin.mW0) & (tT <= tWin.mW1)
            tZ = np.load(tDir / f"solve2_{tArm}.npz")
            tTct = tZ["t"].astype(np.int64)
            tOk = tZ["conf"] > 0
            tOkW = tOk[(tTct >= tWin.mW0) & (tTct <= tWin.mW1)]
            tTw = tTct[(tTct >= tWin.mW0) & (tTct <= tWin.mW1)]
            tGaps = runDurationsS(~tOkW, tTw, ORBBEC_FRAME_S)
            tRawCov = float(tOkW.mean())
            for tVar in variants:
                tFile = workdirs.qaRunPath(tVar, tTake, tArm, tDemo, -1)
                if not tFile.exists():
                    continue
                tR = np.load(tFile)
                tTr = tR["t_ns"]
                tGrid = np.arange(tWin.mW0, tWin.mW1, GRID_NS)
                tCont = float(((tGrid >= tTr[0]) & (tGrid <= tTr[-1])).mean())
                if "info" in tR:
                    tM = (tTr >= tWin.mW0) & (tTr <= tWin.mW1)
                    tSup = tR["info"][tM]
                    tSupCov = float(tSup.mean())
                    tSgap = max(runDurationsS(~tSup, tTr[tM], ORBBEC_FRAME_S) or [0.0])
                else:
                    tSupCov, tSgap = tRawCov, max(tGaps or [0.0])
                tRejoin = []                       # gap-bridge error: hidden >= 1-tag frames inside CT gaps >= 0.2 s (+-0.15 s), bias-removed
                tInGap = np.zeros(len(tT), bool)
                for tI, tJ in trueRuns(~tOkW):
                    if (tTw[tJ - 1] - tTw[tI]) / NS_PER_S + ORBBEC_FRAME_S >= GAP_MIN_S and tTw[tI] > tTct[np.flatnonzero(tOk)[0]]:
                        tInGap |= (tT >= tTw[tI] - GAP_PAD_NS) & (tT <= tTw[tJ - 1] + GAP_PAD_NS)
                for tOff in HIDE_RUNS:
                    tFile3 = workdirs.qaRunPath(tVar, tTake, tArm, tDemo, tOff)
                    if not tFile3.exists():
                        continue
                    tR3 = np.load(tFile3)
                    tHm = hiddenMask(tT, tOff) & tInWin & (D["ntags"] >= 1) & (tT >= tR3["t_ns"][0]) & (tT <= tR3["t_ns"][-1])
                    if not (tHm & tInGap).any():
                        continue
                    tErr = interp(tR3["t_ns"], tR3["p_B"], tT[tHm]) - D["p_tag"][tHm]
                    tBias = np.median(tErr[~tInGap[tHm]], 0) if (~tInGap[tHm]).sum() > MIN_OUTSIDE_FOR_BIAS else 0
                    tRejoin += list(np.linalg.norm(tErr[tInGap[tHm]] - tBias, axis=1) * MM_PER_M)
                tAbs = []
                tFarBursts = []
                for tOff in HIDE_RUNS:
                    tFile2 = workdirs.qaRunPath(tVar, tTake, tArm, tDemo, tOff)
                    if not tFile2.exists():
                        continue
                    tR2 = np.load(tFile2)
                    tHid = hiddenMask(tT, tOff) & tInWin & (D["ntags"] >= 2)
                    if not tHid.any():
                        continue
                    tE = np.linalg.norm(interp(tR2["t_ns"], tR2["p_B"], tT[tHid]) - D["p_tag"][tHid], axis=1)
                    tAbs.append(tE)
                    tBad = np.zeros(len(tT), bool)
                    tBad[np.flatnonzero(tHid)[tE > FAR_M]] = True
                    tFarBursts += runDurationsS(tBad[tHid], tT[tHid], ORBBEC_FRAME_S)
                tAbs = np.concatenate(tAbs) if tAbs else np.zeros(0)
                tE32 = tEval[(tEval["var"] == tVar) & (tEval["arm"] == tArm) & (tEval["demo"] == tDemo)
                             & (tEval["take"].astype(str) == tTake)]
                tRows.append(dict(var=tVar, take=tTake, arm=tArm, demo=tDemo, raw_cov=tRawCov, cont_cov=tCont, sup_cov=tSupCov,
                                  gap_s=tSgap, e32_med=float(tE32["e32_med"].iloc[0]) if len(tE32) else np.nan,
                                  e32_p90=float(tE32["e32_p90"].iloc[0]) if len(tE32) else np.nan,
                                  n=int(tE32["n"].iloc[0]) if len(tE32) else 0,
                                  abs_med=np.median(tAbs) * MM_PER_M if len(tAbs) else np.nan,
                                  abs_p90=np.percentile(tAbs, 90) * MM_PER_M if len(tAbs) else np.nan,
                                  far10=float((tAbs > FAR_M).mean()) if len(tAbs) else np.nan, burst_s=max(tFarBursts or [0.0]),
                                  rejoin=max(tRejoin or [np.nan]), gapbr_med=np.median(tRejoin) if tRejoin else np.nan,
                                  n_gap=len(tRejoin), n_abs=len(tAbs)))
    tDf = pd.DataFrame(tRows)
    tDf.to_csv(workdirs.STITCH_QA_DIR / f"fusion_metrics_{tag}.csv", index=False)
    tOut = tDf.groupby(["arm", "var"], sort=False).apply(_aggregate).reset_index()
    pd.set_option("display.width", DISPLAY_WIDTH)
    print(tOut.round(3).to_string(index=False))
    return tDf


def main(argv: list[str] | None = None) -> int:
    tAp = argparse.ArgumentParser(description="per-arm fusion ablation table of stitch/qa runs")
    tAp.add_argument("pack")
    tAp.add_argument("tag", help="eval csv tag: stitch/qa/eval_<tag>.csv")
    tAp.add_argument("variants", help="VAR1,VAR2,... (folder names under stitch/qa, e.g. fus_Cg_g)")
    tArgs = tAp.parse_args(argv)
    fusionMetrics(tArgs.pack, tArgs.tag, tArgs.variants.split(","))
    return 0


if __name__ == "__main__":
    sys.exit(main())
