"""Plot a (gap-filled, robotized) bimanual trajectory: top view, height over time, displacement.

Was scripts/plot_example.py, then laundry_reference/viz/plot_example.py.

Pipeline stage (reference implementation): visualisation. Input: robotized CSV (+ optional replay CSV); output: PNG figure.

Usage:
    python -m laundry_reference.plot_example out/robotized.csv out/example.png [--robot assets/laundry/examples/synthetic_robot_replay.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import trajio
from .gap_fill import INTERPOLATED, UNFILLED, runs
from .robotize import DEFAULT_Q1_MM, DEFAULT_Q2_MM

COL = {"R": "#c0392b", "L": "#1f5fa8"}
NAME = {"R": "right", "L": "left"}
FIGSIZE_IN = (11, 7.2)
DPI = 130
MM_PER_M = 1000
# `jaw` is 0/1; closed below this.
JAW_OPEN_MIN = 0.5
# Dashed lines of the displacement panel: robotize's quality class boundaries.
QUALITY_BOUNDARIES_MM = (int(DEFAULT_Q1_MM), int(DEFAULT_Q2_MM))
JAW_CLOSED_COLOUR, INTERPOLATED_COLOUR, UNFILLED_COLOUR = "#f39c12", "#8e44ad", "#000000"


def shade(ax, t, mask, color, alpha, label=None):
    """Shade every run of True in `mask` (label on the first span only)."""
    tFirst = True
    for tA, tB in runs(np.asarray(mask, bool)):
        ax.axvspan(t[tA], t[min(tB, len(t) - 1)], color=color, alpha=alpha, lw=0, label=label if tFirst else None)
        tFirst = False


def plotTrajectory(data: dict, robot: dict | None, dst: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tT = data["t"]
    tHands = trajio.handsIn(data)
    tFig = plt.figure(figsize=FIGSIZE_IN, constrained_layout=True)
    tGrid = tFig.add_gridspec(len(tHands) + 1, 2, width_ratios=[1, 1.6])
    tTop = tFig.add_subplot(tGrid[:, 0])
    for tHand in tHands:
        tHx, tHy = data.get(f"{tHand}_human_x", data[f"{tHand}_x"]), data.get(f"{tHand}_human_y", data[f"{tHand}_y"])
        tTop.plot(tHy, tHx, color="0.7", lw=4, solid_capstyle="round", label="human (measured + filled)" if tHand == "R" else None)
        tTop.plot(data[f"{tHand}_y"], data[f"{tHand}_x"], color=COL[tHand], lw=1.4, label=f"{NAME[tHand]}: robotized")
        if robot is not None:
            tTop.plot(robot[f"{tHand}_y"], robot[f"{tHand}_x"], color=COL[tHand], lw=0.8, ls="--", label=f"{NAME[tHand]}: replay")
    tTop.axvline(0, color="k", lw=0.6, ls=":")
    tTop.invert_xaxis()  # +y (left) drawn on the left, as seen from behind the robot
    tTop.set_xlabel("y [m]  (+ = left)")
    tTop.set_ylabel("x [m]  (forward)")
    tTop.set_aspect("equal", adjustable="datalim")
    tTop.set_title("Top view (world frame)")
    tTop.legend(fontsize=7, loc="lower center")

    for tK, tHand in enumerate(tHands):
        tAx = tFig.add_subplot(tGrid[tK, 1])
        tHz = data.get(f"{tHand}_human_z", data[f"{tHand}_z"])
        tAx.plot(tT, MM_PER_M * tHz, color="0.6", lw=2.5, label="human")
        tAx.plot(tT, MM_PER_M * data[f"{tHand}_z"], color=COL[tHand], lw=1.2, label="robotized")
        if robot is not None:
            tAx.plot(robot["t"], MM_PER_M * robot[f"{tHand}_z"], color="k", lw=0.7, ls="--", label="replay")
        shade(tAx, tT, data[f"{tHand}_jaw"] < JAW_OPEN_MIN, JAW_CLOSED_COLOUR, 0.18, "jaw closed")
        if f"{tHand}_filled" in data:
            shade(tAx, tT, data[f"{tHand}_filled"] == INTERPOLATED, INTERPOLATED_COLOUR, 0.25, "interpolated")
            shade(tAx, tT, data[f"{tHand}_filled"] == UNFILLED, UNFILLED_COLOUR, 0.15, "unfilled gap")
        tAx.axhline(0, color="k", lw=0.5)
        tAx.set_ylabel(f"{NAME[tHand]} z [mm]")
        tAx.set_title(f"{NAME[tHand].capitalize()} jaw-tip height", fontsize=9)
        tAx.legend(fontsize=6, ncol=3, loc="upper right")

    tAx = tFig.add_subplot(tGrid[len(tHands), 1])
    for tHand in tHands:
        if f"{tHand}_disp_mm" in data:
            tAx.plot(tT, data[f"{tHand}_disp_mm"], color=COL[tHand], lw=1, label=NAME[tHand])
    for tY in QUALITY_BOUNDARIES_MM:
        tAx.axhline(tY, color="k", lw=0.6, ls="--")
    tAx.set_xlabel("time [s]")
    tAx.set_ylabel("displacement [mm]")
    tAx.set_title("Robotization displacement per frame (dashed: quality class boundaries)", fontsize=9)
    tAx.legend(fontsize=7)
    tFig.suptitle("SYNTHETIC example - not real data", color="#c0392b", fontsize=10)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    tFig.savefig(dst, dpi=DPI)


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    tParser.add_argument("src")
    tParser.add_argument("dst")
    tParser.add_argument("--robot", help="optional replay CSV to overlay")
    return tParser


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    tData = trajio.load(tArgs.src)
    tRobot = trajio.load(tArgs.robot) if tArgs.robot else None
    plotTrajectory(tData, tRobot, tArgs.dst)
    print("wrote", tArgs.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
