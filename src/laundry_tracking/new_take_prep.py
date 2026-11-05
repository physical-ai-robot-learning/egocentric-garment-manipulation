"""Prep for takes recorded after the camera was raised (no fused / SLAM, tag-only), without touching existing rows / files
(was orbbec/v2b/new_take_prep.py, 10-26):
  1. idle trim  -- same rule as laundry_dataset idle_trim (both IMUs |gyro| > GYRO_ACTIVE or both tag-TCP speeds > SPEED_ACTIVE,
                   0.3 s smoothing, MARGIN_S on each side), APPENDED to reports/idle_trim.csv for takes not already in it (backup
                   idle_trim.csv.bak_1026 written first)
  2. start seed -- FUSION_WORK/orbbec/v2b/start_seed_<take>.json: per ARM the rig body (tag TCP + R o_rig) median over the first 15 tag
                   frames of demo 1 (the fixed start spots moved with the 70 mm tags on 10-26; orbbec_tracker seeds / workspace ROI)
Inputs: datasets/tag_rig/<take> (meta.json, umi_traj_<rig>.npz, imu_<rig>.npz), data/tag_rig_derived/<take>/assignment.json.
    python -m laundry_tracking.new_take_prep <take> [<take> ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys

import numpy as np

from laundry_common import paths

from . import workdirs
from .constants import NS_PER_S
from .fgo import arm_data

GYRO_ACTIVE = 0.35                 # rad/s, smoothed |gyro| of an active hand (was GYR)
SPEED_ACTIVE = 0.03                # m/s, smoothed tag-TCP speed of an active hand (was VEL)
MARGIN_S = 0.5                     # kept on each side of the active span (was MARGIN)
GRID_S = 0.02                      # activity grid step
SMOOTH_SAMPLES = 15                # 0.3 s box filter on the 20 ms grid
SEED_TAG_FRAMES = 15               # start seed = median over the first 15 tag frames of the first demo
SEED_DECIMALS = 3
BACKUP_SUFFIX = ".csv.bak_1026"
IDLE_TRIM_FIELDS = ["take", "demo", "demo_len_s", "keep_start_s", "keep_end_s", "cut_head_s", "cut_tail_s", "kept_s", "pose_src"]
ARMS = ("left", "right")


def boxSmooth(x: np.ndarray, n: int) -> np.ndarray:
    """centred moving average of n samples (was new_take_prep.sm)"""
    return np.convolve(x, np.ones(n) / n, mode="same")


def prepareTakes(takes: list[str]) -> list[dict]:
    """writes the start seeds, appends the idle-trim rows of new takes -> the appended rows"""
    tCsv = paths.IDLE_TRIM_CSV
    tHave = {tRow["take"] for tRow in csv.DictReader(open(tCsv))}
    shutil.copy2(tCsv, tCsv.with_suffix(BACKUP_SUFFIX))
    tRows = []
    for tTake in takes:
        tMeta = json.loads((paths.TAG_RIG / tTake / "meta.json").read_text())
        tT0 = int(tMeta["t_start_monotonic_ns"])
        tRigOf = json.loads((paths.DERIVED / tTake / "assignment.json").read_text())["rig_of_arm"]
        tTraj = {tA: np.load(paths.TAG_RIG / tTake / f"umi_traj_{tRigOf[tA]}.npz") for tA in ARMS}
        tEnd = min(tTraj[tA]["t_ns"][-1] for tA in tTraj)
        tSeeds = {}
        for tDemo, tCue in enumerate(tC for tC in tMeta["protocol"] if tC["text"].startswith(arm_data.FREE_CUE_PREFIX)):
            tA0 = tT0 + int(tCue["t0"] * NS_PER_S)
            tA1 = tT0 + int((tCue["t0"] + tCue["dur"]) * NS_PER_S)
            if tA0 >= tEnd:
                continue
            tGrid = np.arange(tA0, min(tA1, tEnd), int(GRID_S * NS_PER_S))
            tActive = np.zeros(len(tGrid), bool)
            for tA in ARMS:
                tImu = np.load(paths.TAG_RIG / tTake / f"imu_{tRigOf[tA]}.npz")
                tGyro = np.interp(tGrid, tImu["host_ns"], np.linalg.norm(tImu["gyro"], axis=1))
                tU = tTraj[tA]
                tSpeed = np.linalg.norm(np.gradient(tU["p_B"], tU["t_ns"] / NS_PER_S, axis=0), axis=1)
                tActive |= (boxSmooth(tGyro, SMOOTH_SAMPLES) > GYRO_ACTIVE) | (boxSmooth(np.interp(tGrid, tU["t_ns"], tSpeed),
                                                                                         SMOOTH_SAMPLES) > SPEED_ACTIVE)
                if tDemo == 0:
                    tK = np.flatnonzero((tU["t_ns"] >= tA0) & (tU["ntags"] > 0))[:SEED_TAG_FRAMES]
                    tBody = tU["p_B"][tK] + np.einsum("nij,j->ni", tU["R_B_tcp"][tK], workdirs.bodyOffsetM(tRigOf[tA]))
                    tSeeds[tA] = np.median(tBody, 0).round(SEED_DECIMALS).tolist()
            if not tActive.any():
                continue
            tIdx = np.flatnonzero(tActive)
            tKeepStart = max(0.0, (tGrid[tIdx[0]] - tA0) / NS_PER_S - MARGIN_S)
            tKeepEnd = min((tGrid[-1] - tA0) / NS_PER_S, (tGrid[tIdx[-1]] - tA0) / NS_PER_S + MARGIN_S)
            tDemoLen = (tA1 - tA0) / NS_PER_S
            if tTake not in tHave:
                tRows.append(dict(take=tTake, demo=tDemo, demo_len_s=round(tDemoLen, 1), keep_start_s=round(tKeepStart, 2),
                                  keep_end_s=round(tKeepEnd, 2), cut_head_s=round(tKeepStart,
                                                                                  1), cut_tail_s=round(tDemoLen - tKeepEnd, 1),
                                  kept_s=round(tKeepEnd - tKeepStart, 1), pose_src="tag_only"))
        workdirs.ORBBEC_DIR.mkdir(parents=True, exist_ok=True)
        workdirs.startSeedPath(tTake).write_text(json.dumps(tSeeds, indent=1))
        print(tTake, "seeds", tSeeds)
    with open(tCsv, "a", newline="") as tHandle:
        tWriter = csv.DictWriter(tHandle, fieldnames=IDLE_TRIM_FIELDS)
        for tRow in tRows:
            tWriter.writerow(tRow)
    return tRows


def main(argv: list[str] | None = None) -> int:
    tAp = argparse.ArgumentParser(description="idle-trim rows + start seeds of newly recorded takes")
    tAp.add_argument("takes", nargs="*")
    tArgs = tAp.parse_args(argv)
    for tRow in prepareTakes(tArgs.takes):
        print(tRow)
    return 0


if __name__ == "__main__":
    sys.exit(main())
