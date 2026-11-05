#!/usr/bin/env python3
"""11-03 new-camera flatten build (dataset stage; was flat1103n_build.py). User: "100데모 ... 5개 val 5개 test ... 90train",
"속도는 0.3으로", train on the 2x RTX PRO 6000 (bh-ai-6000). Per-take LeRobot sets of the FL1103N chain (new camera pose,
robot_cam_v006): Cg labels (--label fused), 0.3x, continuous jaw, stitched masks, built by tag_rig_unsplit (three takes in
parallel) and rsync'ed to the NFS server the 6000 pulls from through Ray.

Holdout rule (--n-hold 5 -> 5 val + 5 test): takes in order of their key count (largest first), visited round-robin; a visit takes
the MIDDLE not-yet-held key of a take with more than two free keys, alternating val / test (val while #val <= #test), until 2 x
n-hold keys are held. The holdout is written to --split-file; FIXED_SPLIT=1 reads it back instead (identical sets on a rebuild).
Inputs: --keys (GOLD + SILVER take/demoNN keys, whitespace separated). Writes to --work-dir: <name>_split.json, build_<name>.log,
fl1103n_build_done.json. Environment: FIXED_SPLIT (0), UPLOAD (1; 0 = build only).
usage: python -m laundry_dataset.builds.per_take_holdout_build [--work-dir DIR] [--keys FILE] [--upload-dest HOST:DIR/]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from laundry_common import paths as egopaths

from .common import (BUILD_FAIL_TAIL_CHARS, RSYNC, TAKE_DATE_LEN, UPLOAD_TAIL_CHARS, builtCounts, lerobotRoot, log, readKeys,
                     runBuilder)

DERIVED = "data/tag_rig_derived_cgg_1103n"
UPLOAD_DEST = "bh-ai-5090@gpu-5090:/srv/data/johann/ldt6000/data/lerobot/"   # NFS server of the 6000 node
N_HOLD = 5
MIN_FREE_KEYS = 2                    # a take gives a holdout key only while it has more free keys than this
BUILD_WORKERS = 3
SPLITS = ("train", "val", "test")
SPEED = "0.3"
SUFFIX = "S03CT"


def pickHoldout(keys: dict, nHold: int = N_HOLD) -> dict:
    """the holdout rule of the module docstring; keys = {take: sorted keys} -> {"val": [...], "test": [...]}"""
    tHeld = {"val": [], "test": []}
    tUsed = set()
    tOrder = sorted(keys, key=lambda t: -len(keys[t]))
    tR = 0
    while len(tHeld["val"]) + len(tHeld["test"]) < 2 * nHold:
        tT = tOrder[tR % len(tOrder)]
        tFree = [tK for tK in keys[tT] if tK not in tUsed]
        if len(tFree) > MIN_FREE_KEYS:
            tK = tFree[len(tFree) // 2]
            tUsed.add(tK)
            tSp = "val" if len(tHeld["val"]) <= len(tHeld["test"]) else "test"
            tHeld[tSp].append(tK)
        tR += 1
    return tHeld


def _upload(d: Path, dest: str, rsync: str) -> bool:
    if os.environ.get("UPLOAD", "1") == "0":
        return True
    tU = subprocess.run(f"{rsync} -rlt --delete --exclude '._*' {d}/ {dest}{d.name}/", shell=True, capture_output=True, text=True)
    if tU.returncode != 0:
        log(f"UPLOAD FAILED {d.name}: {tU.stderr[-UPLOAD_TAIL_CHARS:]}")
        return False
    return True


def _buildTake(t, keys, val, test, a):
    tName = f"{a.prefix}_{t[TAKE_DATE_LEN:]}_{SUFFIX}"
    tKs = keys[t]
    tSplitFile = Path(a.work_dir) / f"{tName}_split.json"
    tSplitFile.write_text(json.dumps({"method": "11-03 new-camera flatten: Cg, 0.3x, continuous jaw; VAL/TEST spread over takes",
                                      "episodes": {"train": [tK for tK in tKs if tK not in val | test],
                                                   "val": [tK for tK in tKs if tK in val],
                                                   "test": [tK for tK in tKs if tK in test]}}))
    tArgs = ["--label", "fused", "--speed", SPEED, "--mask-mode", "stitched", "--ik-fail-mm", "30", "--render", "0",
             "--overwrite",
             "--grip-label", "cont", "--out", f"data/lerobot/{tName}", "--takes", str(egopaths.TAG_RIG / t), "--derived",
             a.derived, "--demos", *tKs,
             "--old-split", str(tSplitFile)]
    tR = runBuilder(tArgs, Path(a.work_dir) / f"build_{tName}.log")
    tRes = {}
    for tS in SPLITS:
        tRoot = lerobotRoot(tName, tS)
        tCounts = builtCounts(tRoot)
        if tCounts is not None:
            tRes[tS] = tCounts
            if not _upload(tRoot, a.upload_dest, a.rsync):
                return t, None
    if "train" not in tRes:
        log(f"BUILD FAILED {tName}: {(tR.stdout + tR.stderr)[-BUILD_FAIL_TAIL_CHARS:]}")
        return t, None
    log(f"built+uploaded {tName}: {tRes}")
    return t, tRes


def main(argv: list[str] | None = None) -> int:
    tAp = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tAp.add_argument("--work-dir", default=".", help="key list, split files, build logs, done json")
    tAp.add_argument("--keys", default="", help="GOLD + SILVER key list (default <work-dir>/keys_fl1103n_gs.txt)")
    tAp.add_argument("--split-file", default="", help="holdout json (default <work-dir>/fl1103n_split.json)")
    tAp.add_argument("--prefix", default="flat1103n")
    tAp.add_argument("--derived", default=DERIVED, help="relative to the training root")
    tAp.add_argument("--n-hold", type=int, default=N_HOLD)
    tAp.add_argument("--upload-dest", default=UPLOAD_DEST)
    tAp.add_argument("--rsync", default=RSYNC)
    tA = tAp.parse_args(argv)
    tWork = Path(tA.work_dir)
    tKeys = collections.defaultdict(list)
    for tK in readKeys(Path(tA.keys) if tA.keys else tWork / "keys_fl1103n_gs.txt"):
        tKeys[tK.split("/")[0]].append(tK)
    tTakes = sorted(tKeys)
    for tT in tTakes:
        tKeys[tT] = sorted(set(tKeys[tT]))
    tSplitFile = Path(tA.split_file) if tA.split_file else tWork / "fl1103n_split.json"
    if os.environ.get("FIXED_SPLIT") == "1":
        tHeld = json.loads(tSplitFile.read_text())
    else:
        tHeld = pickHoldout(tKeys, tA.n_hold)
        tSplitFile.write_text(json.dumps(tHeld, indent=1))
    tVal, tTest = set(tHeld["val"]), set(tHeld["test"])
    if not (tVal | tTest) <= {tK for tT in tTakes for tK in tKeys[tT]}:
        raise SystemExit("held-out key not in GOLD+SILVER keys")
    log(f"VAL {sorted(tVal)}\nTEST {sorted(tTest)}")
    with ThreadPoolExecutor(BUILD_WORKERS) as tPool:
        tOut = dict(tPool.map(lambda t: _buildTake(t, tKeys, tVal, tTest, tA), tTakes))
    (tWork / "fl1103n_build_done.json").write_text(json.dumps(tOut, indent=1))
    tTotal = collections.Counter()
    for tV in tOut.values():
        for tS, (tE, tF) in (tV or {}).items():
            tTotal[tS + "_eps"] += tE
            tTotal[tS + "_frames"] += tF
    log(f"FL1103N_BUILD_DONE failed={[tT for tT, tV in tOut.items() if not tV]} {dict(tTotal)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
