"""Edge-gap trim (was orbbec/v2b/edge_gap_trim.py, 10-27, user, 122533 d8): a one-arm CoTracker invalid run (solve3 conf == 0) that
touches the start or the end of a demo's idle-trim window is cut from reports/idle_trim.csv instead of grading the demo down.
Rule: run >= MIN_RUN_S, starts < EDGE_S after the window start (cut the head to run end + PAD_S) or ends < EDGE_S before the window
end (cut the tail to run start - PAD_S); never leaves < KEEP_MIN_S.  Only tightens; backup idle_trim.csv.bak_edge_<time> first.
Inputs: CoTracker pack dirs (<take>_dNN/solve3_<arm>.npz), the demo windows.
    python -m laundry_tracking.edge_gap_trim <pack> [<pack> ...]
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from laundry_common import paths

from .constants import NS_PER_S
from .fgo import arm_data
from .intervals import trueRuns

MIN_RUN_S = 0.8                    # was MIN_S
EDGE_S = 0.5
KEEP_MIN_S = 8.0
PAD_S = 0.1
TAKE_NAME_CHARS = 15               # <take> = first 15 characters of a pack dir name, the demo = its last 2
DEMO_DIGITS = 2


def demoEdgeCuts(packDir: Path, window: arm_data.DemoWindow) -> tuple[float, float]:
    """(cut_head_s, cut_tail_s) relative to the current window for one demo dir (was edge_gap_trim.edges)"""
    tHead = tTail = 0.0
    for tArm in ("left", "right"):
        tZ = np.load(packDir / f"solve3_{tArm}.npz")
        tT = tZ["t"].astype(np.int64)
        tIn = (tT >= window.mW0) & (tT <= window.mW1)
        if tIn.sum() < 2:
            continue
        tValid = tZ["conf"][tIn] > 0
        tTs = (tT[tIn] - window.mW0) / NS_PER_S
        tLen = (window.mW1 - window.mW0) / NS_PER_S
        for tS, tE in trueRuns(~tValid):
            tA, tB = tTs[tS], tTs[tE - 1]
            if tB - tA < MIN_RUN_S:
                continue
            if tA < EDGE_S:
                tHead = max(tHead, tB + PAD_S)
            if tB > tLen - EDGE_S:
                tTail = max(tTail, tLen - tA + PAD_S)
    return tHead, tTail


def trimEdges(packs: list[str]) -> None:
    tWindows = arm_data.loadTrimWindows()
    tCuts = {}
    for tPack in packs:
        for tDir in sorted(Path(tPack).iterdir()):
            if not (tDir / "solve3_left.npz").exists() or not tDir.name.startswith("2026"):
                continue
            tTake, tDemo = tDir.name[:TAKE_NAME_CHARS], int(tDir.name[-DEMO_DIGITS:])
            tWin = {tW.mDemo: tW for tW in tWindows[tTake]}[tDemo]
            tH, tT = demoEdgeCuts(tDir, tWin)
            tLen = (tWin.mW1 - tWin.mW0) / NS_PER_S
            if (tH > 0 or tT > 0) and tLen - tH - tT >= KEEP_MIN_S:
                tCuts[(tTake, tDemo)] = (tH, tT)
            elif tH > 0 or tT > 0:
                print("skip (would leave < %.0f s)" % KEEP_MIN_S, tTake, tDemo, round(tH, 1), round(tT, 1))
    if not tCuts:
        print("edge-gap trim: nothing to cut")
        return
    tCsv = paths.IDLE_TRIM_CSV
    shutil.copy2(tCsv, tCsv.with_suffix(f".csv.bak_edge_{time.strftime('%m%d_%H%M%S')}"))
    tRows = list(csv.DictReader(open(tCsv)))
    tFields = list(tRows[0])
    for tRow in tRows:
        tCut = tCuts.get((tRow["take"], int(tRow["demo"])))
        if not tCut:
            continue
        tKs0, tKe0, tDl = float(tRow["keep_start_s"]), float(tRow["keep_end_s"]), float(tRow["demo_len_s"])
        tKs, tKe = tKs0 + tCut[0], tKe0 - tCut[1]
        tRow.update(keep_start_s=round(tKs, 2), keep_end_s=round(tKe, 2), cut_head_s=round(tKs, 1),
                    cut_tail_s=round(tDl - tKe, 1), kept_s=round(tKe - tKs, 1))
        print("edge-gap cut", tRow["take"], tRow["demo"], "head %.1f s tail %.1f s -> %.1f s" % (tCut[0], tCut[1], tKe - tKs))
    with open(tCsv, "w", newline="") as tHandle:
        tWriter = csv.DictWriter(tHandle, fieldnames=tFields)
        tWriter.writeheader()
        tWriter.writerows(tRows)
    print("edge-gap trim:", len(tCuts), "demos")


def main(argv: list[str] | None = None) -> int:
    tAp = argparse.ArgumentParser(description="cut one-arm CoTracker gaps at demo-window edges from idle_trim.csv")
    tAp.add_argument("packs", nargs="*")
    trimEdges(tAp.parse_args(argv).packs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
