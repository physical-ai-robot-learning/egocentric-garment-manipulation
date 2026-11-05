"""Derived root for the in-view training set (was orbbec/v2b/fusion/make_derived_cgg.py, 10-25):
<EGO_TRAIN_ROOT>/data/<CGG_DST>/<take>/ = the source derived take (assignment.json, segments.json, arm-labelled umi_traj_<arm>.npz) with
  p_B          replaced inside each demo's fusion span by FUSION_WORK/stitch/qa/fus_Cg_g (gated CoTracker v2 + tags, production run h-1)
  fused_valid  = gated CoTracker valid (solve3 conf > 0) or >= 1 tag on that Orbbec frame  (tag_rig_unsplit 'measured'; fills > 0.5 s
                 are masked as training targets there)
rotation (R_B_tcp, tag_rig_fuse gyro + tags) and everything else unchanged.
Environment: CGG_SRC (default tag_rig_derived_orbbec_v31speed) / CGG_DST (default tag_rig_derived_cgg) under <EGO_TRAIN_ROOT>/data.
10-26: new-camera takes have no v31speed derived take: CGG_SRC=tag_rig_derived CGG_DST=tag_rig_derived_cgg_1026 (tag-only prep base).
    python -m laundry_tracking.fusion.make_derived <pack> [<pack> ...]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from laundry_common import paths

from .. import workdirs

SRC_ENV, DST_ENV = "CGG_SRC", "CGG_DST"
DEFAULT_SRC = "tag_rig_derived_orbbec_v31speed"
DEFAULT_DST = "tag_rig_derived_cgg"
FUSION_VARIANT = "fus_Cg_g"
PRODUCTION_RUN = -1
POSE_SRC = "fus_Cg_g (gated CoTracker v2 + tags), 10-25"
TAKE_NAME_CHARS = 15
DEMO_DIGITS = 2
ARMS = ("left", "right")


def makeDerived(packs: list[str]) -> None:
    tSrc = paths.TRAIN_ROOT / "data" / os.environ.get(SRC_ENV, DEFAULT_SRC)
    tDst = paths.TRAIN_ROOT / "data" / os.environ.get(DST_ENV, DEFAULT_DST)
    tLoc = {}
    for tPack in packs:
        for tDir in Path(tPack).iterdir():
            if (tDir / "solve3_left.npz").exists():
                tLoc[tDir.name] = tDir
    for tTake in sorted({tKey[:TAKE_NAME_CHARS] for tKey in tLoc}):
        tOut = tDst / tTake
        tOut.mkdir(parents=True, exist_ok=True)
        for tName in ("assignment.json", "segments.json"):
            shutil.copy2(tSrc / tTake / tName, tOut / tName)
        for tArm in ARMS:
            tZ = dict(np.load(tSrc / tTake / f"umi_traj_{tArm}.npz", allow_pickle=True))
            tT = tZ["t_ns"].astype(np.int64)
            tP = tZ["p_B"].copy()
            tValid = (tZ["ntags"] >= 1).copy()
            tNRep = 0
            for tKey in sorted(tK for tK in tLoc if tK.startswith(tTake)):
                tDemo = int(tKey[-DEMO_DIGITS:])
                tRun = workdirs.qaRunPath(FUSION_VARIANT, tTake, tArm, tDemo, PRODUCTION_RUN)
                tS3 = np.load(tLoc[tKey] / f"solve3_{tArm}.npz")
                if tRun.exists():
                    tR = np.load(tRun)
                    if len(tR["t_ns"]) >= 2:
                        tM = (tT >= tR["t_ns"][0]) & (tT <= tR["t_ns"][-1])
                        tP[tM] = np.stack([np.interp(tT[tM], tR["t_ns"], tR["p_B"][:, tJ]) for tJ in range(3)], 1)
                        tNRep += int(tM.sum())
                tK = np.searchsorted(tT, tS3["t"].astype(np.int64))
                tOk = (tK < len(tT)) & (tS3["conf"] > 0)
                tValid[tK[tOk]] = True
            tZ["p_B"] = tP
            tZ["fused_valid"] = tValid
            tZ["pose_src"] = np.array(POSE_SRC)
            np.savez(tOut / f"umi_traj_{tArm}.npz", **tZ)
            print(tTake, tArm, "replaced", tNRep, "frames, measured", round(float(tValid.mean()), 3), flush=True)


def main(argv: list[str] | None = None) -> int:
    tAp = argparse.ArgumentParser(description="Cg derived root (fus_Cg_g positions + gated CoTracker validity); env CGG_SRC / CGG_DST")
    tAp.add_argument("packs", nargs="*")
    makeDerived(tAp.parse_args(argv).packs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
