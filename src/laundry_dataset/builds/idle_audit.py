#!/usr/bin/env python3
"""11-04 still-time audit inside the current idle-trim window (dataset stage; was idle_audit_1103.py). User: "움직임 시작전에 잠깐
멈춰있는 구간 / 끝나고 멈춰있는 구간 trim". The output csv feeds tighten_trim.py (laundry_tracking), which keeps a margin on each side.

Per demo of every derived take matching GLOB (<training root>/data/<DERIVED>/<take>/umi_traj_{left,right}.npz, fused Cg TCP p_B,
only fused_valid samples when present): on a 30 Hz grid inside the idle_trim.csv window, speed = max over both arms of the
9-sample (0.3 s) smoothed TCP speed; moving = speed > V_MOVE for at least SUSTAIN s (drops 1-frame jumps); head_s / tail_s = still
time before the first / after the last moving sample (NaN when never moving).
Output csv: take, demo, kept_s, head_s, tail_s, vmax.
usage: python -m laundry_dataset.builds.idle_audit OUT.csv [V_MOVE 0.04] [SUSTAIN_S 0.5] [DERIVED tag_rig_derived_cgg_1103n] [GLOB
20261103_*]
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from laundry_common import paths as egopaths

from ..takes import demoWindows, loadIdleTrim

GRID_HZ = 30
V_MOVE_MPS = 0.04                    # default speed above which a sample counts as moving
SUSTAIN_S = 0.5                      # default: moving this long at least (drops 1-frame jumps)
SMOOTH_SAMPLES = 9
NS_PER_S = 1e9
P90 = 90


def stillHeadTail(speed, vMove: float, sustainS: float) -> tuple:
    """(head_s, tail_s) still time before the first / after the last sustained movement, NaN when there is none"""
    tMoving = speed > vMove
    tLen = int(sustainS * GRID_HZ)
    # moving for >= sustainS (drops 1-frame jumps)
    tSustained = np.convolve(tMoving.astype(int), np.ones(tLen, int), "valid") == tLen
    tStarts = np.flatnonzero(tSustained)
    tMv = np.r_[tStarts[:1], tStarts[-1:] + tLen - 1] if len(tStarts) else tStarts
    return (tMv[0] / GRID_HZ, (len(speed) - 1 - tMv[-1]) / GRID_HZ) if len(tMv) else (np.nan, np.nan)


def armSpeed(z, grid):
    """0.3 s smoothed TCP speed (m/s) of one arm on the grid (fused_valid samples only when present)"""
    tTz = z["t_ns"].astype(np.float64)
    tOk = z["fused_valid"].astype(bool) if "fused_valid" in z.files else np.ones(len(tTz), bool)
    tP = np.stack([np.interp(grid, tTz[tOk], z["p_B"][tOk, tI]) for tI in range(3)], 1)
    tK = SMOOTH_SAMPLES
    tPs = np.stack([np.convolve(np.pad(tP[:, tI], tK // 2, mode="edge"), np.ones(tK) / tK, "valid") for tI in range(3)], 1)
    return np.r_[0, np.linalg.norm(np.diff(tPs, axis=0), axis=1) * GRID_HZ]


def main(argv: list[str] | None = None) -> int:
    tAp = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tAp.add_argument("out")
    tAp.add_argument("v_move", nargs="?", type=float, default=V_MOVE_MPS, help="m/s")
    tAp.add_argument("sustain_s", nargs="?", type=float, default=SUSTAIN_S)
    tAp.add_argument("derived", nargs="?", default="tag_rig_derived_cgg_1103n", help="under <training root>/data")
    tAp.add_argument("glob", nargs="?", default="20261103_*")
    tA = tAp.parse_args(argv)
    tDerived = egopaths.TRAIN_ROOT / "data" / tA.derived
    tTrim = loadIdleTrim(egopaths.IDLE_TRIM_CSV)
    tOut = []
    for tTd in sorted(tDerived.glob(tA.glob)):
        tT = egopaths.TAG_RIG / tTd.name
        tZ = {tS: np.load(tTd / f"umi_traj_{tS}.npz") for tS in ("left", "right")}
        for tD in demoWindows(tT):
            tR = tTrim.get((tT.name, tD.mIdx))
            if tR is None:
                continue
            tW0, tW1 = tD.mT0Ns + float(tR["keep_start_s"]) * NS_PER_S, tD.mT0Ns + float(tR["keep_end_s"]) * NS_PER_S
            tGrid = np.arange(tW0, tW1, NS_PER_S / GRID_HZ)
            tSpeed = np.zeros(len(tGrid))
            for tTraj in tZ.values():
                tSpeed = np.maximum(tSpeed, armSpeed(tTraj, tGrid))
            tH, tTl = stillHeadTail(tSpeed, tA.v_move, tA.sustain_s)
            tOut.append(dict(take=tT.name, demo=tD.mIdx, kept_s=round((tW1 - tW0) / NS_PER_S, 1), head_s=round(tH, 2),
                             tail_s=round(tTl, 2),
                             vmax=round(tSpeed.max(), 3)))
    with open(tA.out, "w", newline="") as tFh:
        tW = csv.DictWriter(tFh, fieldnames=list(tOut[0]))
        tW.writeheader()
        tW.writerows(tOut)
    tH = np.array([tO["head_s"] for tO in tOut])
    tTl = np.array([tO["tail_s"] for tO in tOut])
    print(len(tOut), "demos  head med/p90/max", np.nanmedian(tH), np.nanpercentile(tH, P90), np.nanmax(tH), " tail",
          np.nanmedian(tTl),
          np.nanpercentile(tTl, P90), np.nanmax(tTl))
    return 0


if __name__ == "__main__":
    sys.exit(main())
