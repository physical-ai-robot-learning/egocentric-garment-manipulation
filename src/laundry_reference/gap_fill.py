"""Fill short marker dropouts in a bimanual trajectory and flag what was filled.

Was scripts/gap_fill.py, then laundry_reference/pose/gap_fill.py.

Pipeline stage (reference implementation): pose gap filling. Input: trajectory CSV with ``visible``; output: CSV with short gaps
filled and a per-hand ``<H>_filled`` column. For each hand, runs of frames with ``visible == 0`` are treated as gaps.

* A gap no longer than ``--max-gap`` seconds, bounded by measured frames on
  both sides, is filled: position by linear interpolation (or shape-preserving
  cubic with ``--cubic``), orientation by spherical linear interpolation (slerp)
  between the two bounding measured rotations.
* Longer gaps and gaps touching the start/end are left NaN.

``<H>_filled``: 0 = measured, 1 = interpolated, 2 = unfilled (too long or open-ended). Downstream code should treat class 1 as
a label of lower confidence, never as a measurement.

In a real pipeline the rotation inside a gap would come from a gyroscope
anchored to the measured frames, and a visual-SLAM filler could be accepted
where it agrees with the markers; this tool implements only the generic,
sensor-free baseline.

Usage:
    python -m laundry_reference.gap_fill assets/laundry/examples/synthetic_bimanual_demo.csv out/filled.csv --max-gap 1.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp

from . import trajio

MEASURED, INTERPOLATED, UNFILLED = 0, 1, 2
# `visible` is 0/1; a frame counts as measured above this.
VISIBLE_MIN = 0.5
DEFAULT_MAX_GAP_S = 1.0
PERCENT = 100


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) index pairs of consecutive True runs."""
    tPadded = np.concatenate([[False], mask.astype(bool), [False]])
    tEdges = np.diff(tPadded.astype(int))
    return list(zip(np.flatnonzero(tEdges == 1), np.flatnonzero(tEdges == -1)))


def fillHand(t: np.ndarray, p: np.ndarray, q: np.ndarray, visible: np.ndarray,
             maxGapS: float, cubic: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(positions, quaternions, flags) of one hand with every closed gap of at most `maxGapS` filled."""
    p = p.copy()
    q = q.copy()
    tValid = (visible > VISIBLE_MIN) & np.all(np.isfinite(p), axis=1) & np.all(np.isfinite(q), axis=1)
    tFlag = np.where(tValid, MEASURED, UNFILLED).astype(float)
    if tValid.sum() < 2:
        return p, q, tFlag
    tValidIdx = np.flatnonzero(tValid)
    tCubic = PchipInterpolator(t[tValidIdx], p[tValidIdx], axis=0) if cubic else None
    for tA, tB in runs(~tValid):
        if tA == 0 or tB == len(t):
            continue  # open-ended: no anchor on one side
        tT0, tT1 = t[tA - 1], t[tB]
        if tT1 - tT0 > maxGapS:
            continue
        tGapT = t[tA:tB]
        if cubic:
            p[tA:tB] = tCubic(tGapT)
        else:
            tW = ((tGapT - tT0) / (tT1 - tT0))[:, None]
            p[tA:tB] = (1 - tW) * p[tA - 1] + tW * p[tB]
        tKeys = Rotation.from_quat(np.stack([q[tA - 1], q[tB]]))
        q[tA:tB] = Slerp([tT0, tT1], tKeys)(tGapT).as_quat()
        tFlag[tA:tB] = INTERPOLATED
    return p, q, tFlag


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    tParser.add_argument("src")
    tParser.add_argument("dst")
    tParser.add_argument("--max-gap", type=float, default=DEFAULT_MAX_GAP_S, help="longest gap to fill [s] (default 1.0)")
    tParser.add_argument("--cubic", action="store_true", help="PCHIP position interpolation instead of linear")
    return tParser


def run(argv: list[str] | None = None) -> dict:
    """The command line: fill, print the per-hand shares, write the CSV; returns {hand: {measured, interpolated, unfilled, frames}}."""
    tArgs = buildParser().parse_args(argv)
    tData = trajio.load(tArgs.src)
    tReport = {}
    for tHand in trajio.handsIn(tData):
        tP, tQ, tFlag = fillHand(tData["t"], trajio.position(tData, tHand), trajio.quat(tData, tHand),
                                 tData[f"{tHand}_visible"], tArgs.max_gap, tArgs.cubic)
        trajio.setPose(tData, tHand, tP, tQ)
        tData[f"{tHand}_filled"] = tFlag
        tReport[tHand] = {"measured": float(np.mean(tFlag == MEASURED)), "interpolated": float(np.mean(tFlag == INTERPOLATED)),
                          "unfilled": float(np.mean(tFlag == UNFILLED)), "frames": len(tFlag)}
        tShares = tReport[tHand]
        print(f"{tHand}: measured {PERCENT*tShares['measured']:.1f} %  interpolated "
              f"{PERCENT*tShares['interpolated']:.1f} %  unfilled {PERCENT*tShares['unfilled']:.1f} %")
    Path(tArgs.dst).parent.mkdir(parents=True, exist_ok=True)
    trajio.save(tArgs.dst, tData)
    return tReport


def main(argv: list[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
