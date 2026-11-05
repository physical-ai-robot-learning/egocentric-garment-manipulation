#!/usr/bin/env python3
"""11-04 CoTracker gap audit inside the trimmed window (dataset stage; was ctgap_trimmed.py). User: "cp gap이 조금이라도 발생하면 test나
val로, train에서는 제외" -- the output feeds split_gold_silver_reject (gap_frames == 0 -> clean, max_gap_s >= 0.2 s -> goldgap).

Per demo of each CoTracker pack (<pack>/<take>_d<N>/solve2_{left,right}.npz: t ns, conf), both arms, on the Orbbec frame times
inside the idle-trimmed window (reports/idle_trim.csv): a frame is a gap when solve2 conf == 0 (linearly interpolated, rounded) or
it lies outside the tracked span. gap_frames = gap frames of both arms; <arm>_gap_s = the longest gap run of that arm (s);
max_gap_s = max of the two. Demos without both solve2 files or without an idle-trim row are skipped.
Output csv: take, demo, gap_frames, max_gap_s, L_gap_s, R_gap_s.
usage: python -m laundry_dataset.builds.ct_gap_audit OUT.csv PACK [PACK ...]
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np

from laundry_common import paths as egopaths

from ..takes import demoWindows, loadIdleTrim

NS_PER_S = 1e9
REPORT_GAPS_S = (0.2, 0.5)           # summary line: demos with a gap longer than these


def armGap(frameNs, trackNs, conf) -> tuple:
    """gap mask of one arm on the frame times -> (gap frames, longest gap run in s)"""
    tOk = conf > 0
    tGap = ~np.interp(frameNs, trackNs, tOk.astype(float)).round().astype(bool) | (frameNs < trackNs[0]) | (frameNs > trackNs[-1])
    tX = np.r_[0, tGap.astype(int), 0]
    tS, tE = np.flatnonzero(np.diff(tX) == 1), np.flatnonzero(np.diff(tX) == -1)
    if not tGap.any():
        return 0, 0.0
    tN = len(frameNs)
    tLongest = max([(frameNs[min(tB, tN - 1)] - frameNs[tA]) / NS_PER_S if tB < tN else (frameNs[-1] - frameNs[tA]) / NS_PER_S
                    for tA, tB in zip(tS, tE)] + [0.0])
    return int(tGap.sum()), tLongest


def main(argv: list[str] | None = None) -> int:
    tAp = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tAp.add_argument("out")
    tAp.add_argument("packs", nargs="+")
    tA = tAp.parse_args(argv)
    tTrim = loadIdleTrim(egopaths.IDLE_TRIM_CSV)
    tDemoStart = {}
    tRows = []
    for tPack in tA.packs:
        for tD in sorted(pathlib.Path(tPack).iterdir()):
            if not ((tD / "solve2_left.npz").exists() and (tD / "solve2_right.npz").exists()):
                continue
            tTake, tDemo = tD.name.rsplit("_d", 1)
            tDemo = int(tDemo)
            if tTake not in tDemoStart:
                tDemoStart[tTake] = {tX.mIdx: tX.mT0Ns for tX in demoWindows(egopaths.TAG_RIG / tTake)}
            tR = tTrim.get((tTake, tDemo))
            if tR is None:
                continue
            tW0 = tDemoStart[tTake][tDemo] + float(tR["keep_start_s"]) * NS_PER_S
            tW1 = tDemoStart[tTake][tDemo] + float(tR["keep_end_s"]) * NS_PER_S
            tTt = np.load(egopaths.TAG_RIG / tTake / "t_ns.npy").astype(np.int64)
            tTt = tTt[(tTt >= tW0) & (tTt <= tW1)]
            tLongest = {}
            tFrames = 0
            for tArm, tSide in (("left", "L"), ("right", "R")):
                tZ = np.load(tD / f"solve2_{tArm}.npz")
                tN, tLongest[tSide] = armGap(tTt, tZ["t"].astype(np.int64), tZ["conf"])
                tFrames += tN
            tRows.append(dict(take=tTake, demo=tDemo, gap_frames=tFrames, max_gap_s=round(max(tLongest.values()), 2),
                              L_gap_s=round(tLongest["L"], 2), R_gap_s=round(tLongest["R"], 2)))
    with open(tA.out, "w", newline="") as tFh:
        tW = csv.DictWriter(tFh, fieldnames=list(tRows[0]))
        tW.writeheader()
        tW.writerows(tRows)
    tNDemos = len(tRows)
    print(tNDemos, "demos;", "any gap", sum(tR["gap_frames"] > 0 for tR in tRows), "| >0.2s",
          sum(tR["max_gap_s"] > REPORT_GAPS_S[0] for tR in tRows),
          "| >0.5s", sum(tR["max_gap_s"] > REPORT_GAPS_S[1] for tR in tRows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
