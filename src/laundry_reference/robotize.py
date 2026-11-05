"""Make a human bimanual jaw-tip path more robot-feasible with smooth warps only.

Was scripts/robotize.py, then laundry_reference/retarget/robotize.py.

Pipeline stage (reference implementation): retargeting. Input: trajectory CSV; output: CSV with warped x, y, z, the human path as
``<H>_human_x/y/z``, per-frame ``<H>_disp_mm`` and ``<H>_quality``.

The human path is never clipped. Two continuous warps are applied per hand:

1. **Side warp (lateral, y).** Each arm should stay on its own side of the
   body midline with a minimum hand separation ``d``. For the right hand
   (y < 0 side) let ``y_o`` / ``y_i`` be the 1st / 99th percentile of y (outer /
   inner extent). If ``y_i`` already respects ``-d/2`` nothing changes;
   otherwise the path is compressed affinely toward the outer side,

       s  = clip((-d/2 - y_o) / (y_i - y_o), s_min, 1)
       y' = softceil(y_o + s (y - y_o), -d/2, w)

   so the shape is kept and only the residual excursion is bent smoothly by a
   softplus ceiling. The left hand is mirrored.

2. **Table floor (z).** z' = softfloor(z, z_table + clearance, w), with
   softfloor(z, f, w) = f + w log(1 + exp((z - f) / w)). Note that points
   exactly at the floor are lifted by w ln 2.

For every frame the displacement ``|p' - p|`` (mm) and a quality class are
stored: 0 if displacement < q1, 1 if < q2, else 2. These labels are meant as
per-frame weights for later learning: large shifts during contact phases are
the harmful ones.

All numeric defaults here are illustrative, not tuned for any specific robot.

Usage:
    python -m laundry_reference.robotize out/filled.csv out/robotized.csv --sep 0.05
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import trajio

DEFAULT_SEP_M = 0.05
DEFAULT_WIDTH_M = 0.01
DEFAULT_Z_TABLE_M = 0.0
DEFAULT_CLEARANCE_M = 0.0
DEFAULT_S_MIN = 0.4
DEFAULT_Q1_MM = 20.0
DEFAULT_Q2_MM = 50.0
# The side warp's outer / inner extent: these percentiles of the mirrored y.
OUTER_PERCENTILE, INNER_PERCENTILE = 1, 99
# Below this y extent [m] the path is a point: no compression.
MIN_EXTENT_M = 1e-9
RIGHT_SIDE, LEFT_SIDE = -1.0, 1.0
MM_PER_M = 1000.0
QUALITY_CLASSES = (0, 1, 2)
PERCENT = 100
P95 = 95
# Key of the per-hand lateral scales in the dict `robotize` returns (popped by the callers before saving).
SCALE_KEY = "_scale"


def softFloor(z: np.ndarray, f: float, w: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):  # NaN (unfilled) frames stay NaN
        return f + w * np.logaddexp(0.0, (z - f) / w)


def softCeil(y: np.ndarray, c: float, w: float) -> np.ndarray:
    return -softFloor(-y, -c, w)


def sideWarp(y: np.ndarray, side: float, sep: float, w: float, sMin: float) -> tuple[np.ndarray, float]:
    """(warped y, lateral scale s); side = -1 for the right hand (y<0 half), +1 for the left hand."""
    tMirrored = -side * y  # mirror so that the hand's own side is y < 0
    tFinite = np.isfinite(tMirrored)
    if not tFinite.any():
        return y.copy(), 1.0
    tOuter, tInner = np.percentile(tMirrored[tFinite], [OUTER_PERCENTILE, INNER_PERCENTILE])
    tLimit = -sep / 2
    if tInner <= tLimit or tInner - tOuter < MIN_EXTENT_M:
        tScale = 1.0
    else:
        tScale = float(np.clip((tLimit - tOuter) / (tInner - tOuter), sMin, 1.0))
    tOut = softCeil(tOuter + tScale * (tMirrored - tOuter), tLimit, w)
    return -side * tOut, tScale


def quality(dispMm: np.ndarray, q1: float, q2: float) -> np.ndarray:
    """0 below q1, 1 below q2, else 2 (mm); NaN where the displacement is NaN."""
    tQuality = np.full(dispMm.shape, 2.0)
    tQuality[dispMm < q2] = 1.0
    tQuality[dispMm < q1] = 0.0
    tQuality[~np.isfinite(dispMm)] = np.nan
    return tQuality


def robotize(data: dict, sep: float = DEFAULT_SEP_M, width: float = DEFAULT_WIDTH_M, zTable: float = DEFAULT_Z_TABLE_M,
             clearance: float = DEFAULT_CLEARANCE_M, sMin: float = DEFAULT_S_MIN, q1Mm: float = DEFAULT_Q1_MM,
             q2Mm: float = DEFAULT_Q2_MM) -> dict:
    """Copy of `data` with every hand warped; SCALE_KEY holds {hand: lateral scale}."""
    tOut = {tKey: tValue.copy() for tKey, tValue in data.items()}
    for tHand in trajio.handsIn(data):
        tSide = RIGHT_SIDE if tHand == "R" else LEFT_SIDE
        tP = trajio.position(data, tHand)
        tWarped = tP.copy()
        tWarped[:, 1], tScale = sideWarp(tP[:, 1], tSide, sep, width, sMin)
        tWarped[:, 2] = softFloor(tP[:, 2], zTable + clearance, width)
        trajio.setPose(tOut, tHand, tWarped)
        for tI, tAxis in enumerate("xyz"):
            tOut[f"{tHand}_human_{tAxis}"] = tP[:, tI]
        tDisp = MM_PER_M * np.linalg.norm(tWarped - tP, axis=1)
        tOut[f"{tHand}_disp_mm"] = tDisp
        tOut[f"{tHand}_quality"] = quality(tDisp, q1Mm, q2Mm)
        tOut.setdefault(SCALE_KEY, {})[tHand] = tScale
    return tOut


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    tParser.add_argument("src")
    tParser.add_argument("dst")
    tParser.add_argument("--sep", type=float, default=DEFAULT_SEP_M, help="minimum hand separation across the midline [m]")
    tParser.add_argument("--width", type=float, default=DEFAULT_WIDTH_M, help="softplus width w [m]")
    tParser.add_argument("--z-table", type=float, default=DEFAULT_Z_TABLE_M, help="table height in the world frame [m]")
    tParser.add_argument("--clearance", type=float, default=DEFAULT_CLEARANCE_M, help="extra floor clearance [m]")
    tParser.add_argument("--s-min", type=float, default=DEFAULT_S_MIN, help="strongest allowed lateral compression")
    tParser.add_argument("--q1", type=float, default=DEFAULT_Q1_MM, help="quality 0/1 boundary [mm]")
    tParser.add_argument("--q2", type=float, default=DEFAULT_Q2_MM, help="quality 1/2 boundary [mm]")
    return tParser


def run(argv: list[str] | None = None) -> dict:
    """The command line: warp, print per hand, write the CSV; returns {hand: lateral scale, displacement stats, quality shares}."""
    tArgs = buildParser().parse_args(argv)
    tData = trajio.load(tArgs.src)
    tOut = robotize(tData, tArgs.sep, tArgs.width, tArgs.z_table, tArgs.clearance, tArgs.s_min, tArgs.q1, tArgs.q2)
    tScales = tOut.pop(SCALE_KEY)
    tReport = {}
    for tHand in trajio.handsIn(tData):
        tDisp = tOut[f"{tHand}_disp_mm"]
        tQuality = tOut[f"{tHand}_quality"]
        tFinite = np.isfinite(tDisp)
        tReport[tHand] = {"lateral_scale": tScales[tHand], "disp_median_mm": float(np.median(tDisp[tFinite])),
                          "disp_p95_mm": float(np.percentile(tDisp[tFinite], P95)), "disp_max_mm": float(tDisp[tFinite].max()),
                          "quality_share": [float(np.mean(tQuality[tFinite] == tClass)) for tClass in QUALITY_CLASSES]}
        tRow = tReport[tHand]
        print(f"{tHand}: lateral scale {tRow['lateral_scale']:.2f}  displacement median {tRow['disp_median_mm']:.1f} / "
              f"p95 {tRow['disp_p95_mm']:.1f} / max {tRow['disp_max_mm']:.1f} mm  quality 0/1/2 = "
              + "/".join(f"{PERCENT*tShare:.0f}%" for tShare in tRow["quality_share"]))
    Path(tArgs.dst).parent.mkdir(parents=True, exist_ok=True)
    trajio.save(tArgs.dst, tOut)
    return tReport


def main(argv: list[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
