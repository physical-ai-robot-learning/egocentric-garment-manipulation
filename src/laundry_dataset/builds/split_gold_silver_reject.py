#!/usr/bin/env python3
"""11-04 flatten GOLD / SILVER / REJECT sets (dataset stage; was flat_gsr_build.py): gated position masks (FILL_MASK_S=0.3),
timeline and images never cut. Per take: <prefix>_<HHMMSS>_S03G_{train,val,test,silver,reject} built by tag_rig_unsplit (--label
fused, --speed 0.3, --mask-mode gated, --grasp-fill-drop off, --ik-fail-mm 30, --grip-label cont), up to three takes in parallel.

Inputs: the episode classes (quality csv(s): take, demo, class; a later file wins for a repeated key) and the CoTracker gap audit
(ct_gap_audit csv(s): take, demo, gap_frames, max_gap_s, ...; a later file wins).
Decision rules (user 11-04):
  1. keys = every classified demo minus the excluded keys; GOLD -> train, SILVER -> silver (auxiliary robustness eval), REJECT ->
     reject (analysis only, never trained).
  0. per-demo class overrides from <work-dir>/class_overrides_1104.json ({key: {"class": ...}}) when present.
  2. "0.2 이상은 못쓰겠네 train은 0.2만 남겨줘", then 11-04 19:5x "0.25까지는 사용해도 되는거 같아": a GOLD demo with a CT gap
     max_gap_s >= GOLD_GAP_MAX (0.25 s; no audit row = 9.0 s) is goldgap (not trained, not the clean holdout).
  3. clean holdout: gap-free GOLD (gap_frames == 0; no audit row = 1 frame), grouped by take; takes in order of their candidate
     count (largest first, round-robin); each visit takes the MIDDLE remaining candidate of a take that still has more than one,
     alternating val / test (val while #val <= #test) until NHOLD + NHOLD are held out or no take has two candidates left.
  4. goldgap: GOLDGAP_TO=train -> train ("그럼 써야지", positions masked by the gated 0.3 s fill); else with GOLDGAP_TO_VALTEST=1
     (default, "goldgap 은 test나 발로 쓰자") the goldgap keys in sorted order alternate val, test, val, ...; else they stay goldgap.
  5. SILVER_NOGAP_TRAIN=1 (default, "silver에서 cp gap 없는거는 다 train으로"): SILVER with 0 CT-gap frames in the trimmed window -> train.
  6. FORCE_TRAIN=<key,...> (11-04 22:2x, per demo after the user watched it) -> train, removed from the holdout lists.
The final 11-04 / 11-05 builds ran with NHOLD=0 (the clean-GOLD holdout returned to train). ONLY_TAKES=<take,...> rebuilds just
those takes (the split json still covers all).
Writes to --work-dir: <prefix>_gsr_split.json, <name>_split.json per take (the builder's --old-split) and build_<name>.log.
Environment: NHOLD (5), GOLD_GAP_MAX (0.25), GOLDGAP_TO, GOLDGAP_TO_VALTEST (1), SILVER_NOGAP_TRAIN (1), FORCE_TRAIN, ONLY_TAKES.
usage: python -m laundry_dataset.builds.split_gold_silver_reject PREFIX DERIVED Q.csv[,Q2.csv] CTGAP.csv[,..] [EXCLUDE_KEYS]
[--work-dir DIR]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from laundry_common import paths as egopaths

from .common import TAIL_CHARS, TAKE_DATE_LEN, builderEnv, builtCounts, lerobotRoot, log, runBuilder

CLASS_SPLIT = {"GOLD": "train", "SILVER": "silver", "REJECT": "reject"}
NO_AUDIT_MAX_GAP_S = 9.0             # a demo without a CT-gap row counts as gapped (never clean, goldgap if GOLD)
NO_AUDIT_GAP_FRAMES = 1
BUILD_WORKERS = 3
FILL_MASK_S = "0.3"                  # gated position masks for every set of this build
SPEED = "0.3"
SUFFIX = "S03G"
CLASS_OVERRIDES = "class_overrides_1104.json"   # 11-04 user per-demo class overrides (e.g. a REJECT fixed by a manual front trim)


def _keyed(frames):
    tAll = pd.concat(frames)
    tAll["key"] = [f"{tR.take}/demo{int(tR.demo):02d}" for tR in tAll.itertuples()]
    return tAll.drop_duplicates("key", keep="last").set_index("key")


def assignSplits(cls, gapFrames, maxGap, exclude, nHold: int, goldGapMax: float, goldgapTo: str, goldgapToValTest: str,
                 silverNoGapTrain: str = "0", forceTrain: tuple[str, ...] = ()):
    """the decision rules of the module docstring -> (sorted keys, {key: split}, held {"val": [...], "test": [...]})"""
    tKeys = sorted(tK for tK in cls.index if tK not in exclude)
    tSplit = {tK: CLASS_SPLIT[cls[tK]] for tK in tKeys}
    for tK in tKeys:
        if cls[tK] == "GOLD" and maxGap.get(tK, NO_AUDIT_MAX_GAP_S) >= goldGapMax:
            tSplit[tK] = "goldgap"
    tBy = collections.defaultdict(list)
    for tK in tKeys:
        if cls[tK] == "GOLD" and gapFrames.get(tK, NO_AUDIT_GAP_FRAMES) == 0:
            tBy[tK.split("/")[0]].append(tK)
    tOrder = sorted(tBy, key=lambda t: -len(tBy[t]))
    tHeld = {"val": [], "test": []}
    tR = 0
    while len(tHeld["val"]) + len(tHeld["test"]) < 2 * nHold and any(len(tV) > 1 for tV in tBy.values()):
        tT = tOrder[tR % len(tOrder)]
        tR += 1
        if len(tBy[tT]) > 1:
            tK = tBy[tT].pop(len(tBy[tT]) // 2)
            tSp = "val" if len(tHeld["val"]) <= len(tHeld["test"]) else "test"
            tHeld[tSp].append(tK)
            tSplit[tK] = tSp
    if goldgapTo == "train":
        for tK in tKeys:
            if tSplit[tK] == "goldgap":
                tSplit[tK] = "train"
    elif goldgapToValTest == "1":
        tGoldgap = sorted(tK for tK in tKeys if tSplit[tK] == "goldgap")
        for tI, tK in enumerate(tGoldgap):
            tSplit[tK] = "val" if tI % 2 == 0 else "test"
        tHeld = {"val": tHeld["val"] + tGoldgap[0::2], "test": tHeld["test"] + tGoldgap[1::2]}
    if silverNoGapTrain == "1":
        for tK in tKeys:
            if tSplit[tK] == "silver" and gapFrames.get(tK, NO_AUDIT_GAP_FRAMES) == 0:
                tSplit[tK] = "train"
    for tK in forceTrain:
        if tK in tSplit:
            tSplit[tK] = "train"
            tHeld = {tS: [tX for tX in tV if tX != tK] for tS, tV in tHeld.items()}
    return tKeys, tSplit, tHeld


def _buildTake(t, keys, split, prefix, derived, work):
    tKeys = [tK for tK in keys if tK.startswith(t + "/")]
    tName = f"{prefix}_{t[TAKE_DATE_LEN:]}_{SUFFIX}"
    tSplitFile = work / f"{tName}_split.json"
    tEps = collections.defaultdict(list)
    for tK in tKeys:
        tEps[split[tK]].append(tK)
    tSplitFile.write_text(json.dumps({"method": "11-04 GOLD/SILVER/REJECT, gated FILL_MASK_S 0.3, clean-GOLD holdout",
                                      "episodes": tEps}))
    tArgs = ["--label", "fused", "--speed", SPEED, "--mask-mode", "gated", "--grasp-fill-drop", "off", "--ik-fail-mm", "30",
             "--render", "0",
             "--overwrite", "--grip-label", "cont", "--out", f"data/lerobot/{tName}", "--takes", str(egopaths.TAG_RIG / t),
             "--derived", derived,
             "--demos", *tKeys, "--old-split", str(tSplitFile)]
    tR = runBuilder(tArgs, work / f"build_{tName}.log", builderEnv(FILL_MASK_S=FILL_MASK_S))
    tRes = {}
    for tS in tEps:
        tCounts = builtCounts(lerobotRoot(tName, tS))
        if tCounts is not None:
            tRes[tS] = tCounts
    tMiss = {tS: len(tV) - tRes.get(tS, (0, 0))[0] for tS, tV in tEps.items() if len(tV) != tRes.get(tS, (0, 0))[0]}
    log(f"built {tName}: {tRes}" + (f"  MISSING {tMiss} -> {(tR.stdout + tR.stderr)[-TAIL_CHARS:]}" if tMiss else ""))
    return t, tRes


def main(argv: list[str] | None = None) -> int:
    tAp = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tAp.add_argument("prefix")
    tAp.add_argument("derived",
                     help="derived dir of the takes, relative to the training root (e.g. data/tag_rig_derived_cgg_1103n)")
    tAp.add_argument("quality", help="comma-separated quality csv(s) (take, demo, class), relative to --work-dir")
    tAp.add_argument("ctgap", help="comma-separated ct_gap_audit csv(s), relative to --work-dir")
    tAp.add_argument("exclude", nargs="?", default="", help="comma-separated take/demoNN keys never used")
    tAp.add_argument("--work-dir", default=".", help="where the inputs live and the split files / build logs are written")
    tA = tAp.parse_args(argv)
    tWork = Path(tA.work_dir)
    tExclude = set(tA.exclude.split(",")) if tA.exclude else set()
    tQ = _keyed([pd.read_csv(tWork / tF, dtype={"take": str}) for tF in tA.quality.split(",")])
    tOverrides = tWork / CLASS_OVERRIDES
    if tOverrides.exists():
        for tK, tV in json.loads(tOverrides.read_text()).items():
            if tK in tQ.index:
                tQ.loc[tK, "class"] = tV["class"]
    tGap = _keyed([pd.read_csv(tWork / tF, dtype={"take": str}) for tF in tA.ctgap.split(",")])
    tKeys, tSplit, tHeld = assignSplits(tQ["class"], tGap["gap_frames"], tGap["max_gap_s"], tExclude,
                                        int(os.environ.get("NHOLD", "5")), float(os.environ.get("GOLD_GAP_MAX", "0.25")),
                                        os.environ.get("GOLDGAP_TO", ""), os.environ.get("GOLDGAP_TO_VALTEST", "1"),
                                        os.environ.get("SILVER_NOGAP_TRAIN", "1"),
                                        tuple(filter(None, os.environ.get("FORCE_TRAIN", "").split(","))))
    tCount = collections.Counter(tSplit.values())
    log(f"{tA.prefix}: {dict(tCount)}  val {tHeld['val']}  test {tHeld['test']}")
    tBySplit = {tS: sorted(tK for tK, tV in tSplit.items() if tV == tS) for tS in tCount}
    (tWork / f"{tA.prefix}_gsr_split.json").write_text(json.dumps(tBySplit, indent=1))
    tOnly = set(filter(None, os.environ.get("ONLY_TAKES", "").split(",")))
    tTakes = sorted({tK.split("/")[0] for tK in tKeys if not tOnly or tK.split("/")[0] in tOnly})
    with ThreadPoolExecutor(BUILD_WORKERS) as tPool:
        tOut = dict(tPool.map(lambda t: _buildTake(t, tKeys, tSplit, tA.prefix, tA.derived, tWork), tTakes))
    tTotal = collections.Counter()
    for tV in tOut.values():
        for tS, (tE, _) in tV.items():
            tTotal[tS] += tE
    log(f"{tA.prefix}_GSR_DONE episodes {dict(tTotal)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
