"""Idle head/tail trim per demo from BOTH IMUs (|gyro|) and BOTH fused TCP speeds (dataset stage, before the LeRobot builders):
active = any arm moving (|gyro| > GYRO_ACTIVE rad/s or |v| > SPEED_ACTIVE m/s, 0.3 s smoothing); trimmed window = the main working
block (active runs closer than GAP_MERGE_S merged, the block with the most active time) with MARGIN_S on each side.
Inputs: every 10-22 take with a fused derived copy + the two valid 10-23 takes (laundry_common.paths TAG_RIG / DERIVED).
Output: reports/idle_trim.csv (laundry_common.paths.IDLE_TRIM_CSV), read by tag_rig_unsplit and the build scripts.
    python -m laundry_dataset.idle_trim
"""
from __future__ import annotations

import argparse
import csv
import json
import sys

import numpy as np

from laundry_common import paths as egopaths

from .takes import FREE_DEMO_PREFIX

GYRO_ACTIVE = 0.35                   # rad/s
SPEED_ACTIVE = 0.03                  # m/s
MARGIN_S = 0.5
GAP_MERGE_S = 2.0
GRID_S = 0.02                        # 50 Hz activity grid
SMOOTH_SAMPLES = 15                  # 0.3 s at 50 Hz
TAKE_GLOB = "20261022_*"
EXTRA_TAKES = ["20261023_113550", "20261023_114817"]   # the two valid 10-23 takes
DEFAULT_RIG_OF_ARM = {"left": "right", "right": "left"}   # tag-only takes: rig labels swapped (10-22)
NS_PER_S = 1e9


def _smooth(x, n):
    return np.convolve(x, np.ones(n) / n, mode="same")


def takeList(tagRig, derived) -> list:
    tFused = sorted(tP.name for tP in tagRig.glob(TAKE_GLOB) if (derived / f"{tP.name}_fused/assignment.json").exists())
    return tFused + EXTRA_TAKES


def mainBlock(active) -> tuple:
    """main working block: merge active runs separated by < GAP_MERGE_S s, keep the block with the most active time (isolated
    blips after the task -- putting the UMI down, re-gripping -- are idle, not demo) -> (first, last) active grid index"""
    tOn = np.flatnonzero(active)
    tBreaks = np.flatnonzero(np.diff(tOn) * GRID_S > GAP_MERGE_S)
    tBest = max(np.split(tOn, tBreaks + 1), key=len)
    return tBest[0], tBest[-1]


def trimTake(take, tagRig, derived) -> list:
    """idle-trim rows of every free demo of one take"""
    tMeta = json.loads((tagRig / take / "meta.json").read_text())
    tStart = int(tMeta["t_start_monotonic_ns"])
    tFused = (derived / f"{take}_fused").exists()
    if tFused:
        tSrc = lambda arm, rig: derived / f"{take}_fused/umi_traj_{arm}.npz"
        tRigOf = json.loads((derived / f"{take}_fused/assignment.json").read_text())["rig_of_arm"]
    else:
        tSrc = lambda arm, rig: tagRig / take / f"umi_traj_{rig}.npz"
        tRigOf = DEFAULT_RIG_OF_ARM
    tEnd = min(np.load(tSrc(tA, tRigOf[tA]))["t_ns"][-1] for tA in ("left", "right"))
    tRows = []
    for tDemo, tProto in enumerate(tP for tP in tMeta["protocol"] if tP["text"].startswith(FREE_DEMO_PREFIX)):
        tA0 = tStart + int(tProto["t0"] * NS_PER_S)
        tA1 = tStart + int((tProto["t0"] + tProto["dur"]) * NS_PER_S)
        if tA0 >= tEnd:
            continue
        tGrid = np.arange(tA0, min(tA1, tEnd), int(GRID_S * NS_PER_S))
        tAct = np.zeros(len(tGrid), bool)
        for tArm in ("left", "right"):
            tImu = np.load(tagRig / take / f"imu_{tRigOf[tArm]}.npz")
            tGyro = np.interp(tGrid, tImu["host_ns"], np.linalg.norm(tImu["gyro"], axis=1))
            tU = np.load(tSrc(tArm, tRigOf[tArm]))
            tSpeed = np.linalg.norm(np.gradient(tU["p_B"], tU["t_ns"] / NS_PER_S, axis=0), axis=1)
            tMoving = _smooth(np.interp(tGrid, tU["t_ns"], tSpeed), SMOOTH_SAMPLES) > SPEED_ACTIVE
            tAct |= (_smooth(tGyro, SMOOTH_SAMPLES) > GYRO_ACTIVE) | tMoving
        if not tAct.any():
            continue
        tI0, tI1 = mainBlock(tAct)
        tLen = (tGrid[-1] - tA0) / NS_PER_S
        tS0 = max(0.0, (tGrid[tI0] - tA0) / NS_PER_S - MARGIN_S)
        tS1 = min(tLen, (tGrid[tI1] - tA0) / NS_PER_S + MARGIN_S)
        tRows.append(dict(take=take, demo=tDemo, demo_len_s=round(tLen, 1), keep_start_s=round(tS0, 2), keep_end_s=round(tS1, 2),
                          cut_head_s=round(tS0, 1), cut_tail_s=round(tLen - tS1, 1), kept_s=round(tS1 - tS0, 1),
                          pose_src="fused" if tFused else "tag"))
    return tRows


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args(argv)
    tRows = []
    for tTake in takeList(egopaths.TAG_RIG, egopaths.DERIVED):
        tRows.extend(trimTake(tTake, egopaths.TAG_RIG, egopaths.DERIVED))
    tOut = egopaths.IDLE_TRIM_CSV
    with open(tOut, "w", newline="") as tFh:
        tW = csv.DictWriter(tFh, fieldnames=list(tRows[0]))
        tW.writeheader()
        tW.writerows(tRows)
    tHead = np.array([tR["cut_head_s"] for tR in tRows])
    tTail = np.array([tR["cut_tail_s"] for tR in tRows])
    tKept = np.array([tR["kept_s"] for tR in tRows])
    print(tOut, len(tRows), "demos | head cut median", np.median(tHead), "max", tHead.max(), "| tail cut median",
          np.median(tTail), "max", tTail.max(),
          "| kept median", np.median(tKept))
    return 0


if __name__ == "__main__":
    sys.exit(main())
