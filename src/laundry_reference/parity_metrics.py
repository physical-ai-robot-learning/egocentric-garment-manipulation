"""Compare two time-stamped bimanual trajectories (e.g. human demo vs robot replay).

Was scripts/parity_metrics.py, then laundry_reference/evaluation/parity_metrics.py.

Pipeline stage (reference implementation): evaluation. Input: reference CSV + test CSV; output: printed metrics, optional JSON.
For each hand and an optional time window:

* both trajectories are resampled onto the reference time grid;
* both are made *relative* to their own first valid pose, so constant offsets
  between the two frames are not counted (see the caveat below);
* the test trajectory is shifted by the lag L* in [0, --max-lag] that minimises
  the mean squared position error, so a pure delay is not counted as error;
* reported: position RMSE / p95 / max [mm], rotation error median / p95 [deg]
  (geodesic angle between the relative rotations), amplitude ratio (RMS
  distance from the start, test over reference), the lag, and the fraction of
  frames that entered the comparison.

Caveat: relative comparison and lag search hide constant offsets and delays.
They answer "does the motion have the same shape?", not "is it in the right
place?". Absolute placement must be checked separately (e.g. against an
external camera or a tape measure).

Usage:
    python -m laundry_reference.parity_metrics REF.csv TEST.csv [--t0 S --t1 S] [--max-lag 1.5] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from . import trajio

DEFAULT_MAX_LAG_S = 1.5
DEFAULT_LAG_STEP_S = 0.02
# Closes the lag range: the last step is included despite float accumulation.
LAG_RANGE_EPS_S = 1e-9
# A lag is only scored with at least this many frames valid in both trajectories.
MIN_OVERLAP_FRAMES = 10
MM_PER_M = 1000.0
P95 = 95
PERCENT = 100
IDENTITY_QUAT = [0, 0, 0, 1]
# `visible` is 0/1; a reference frame counts as measured at or above this (with --only-measured).
VISIBLE_MIN = 0.5
JSON_INDENT = 2


def resample(tSrc: np.ndarray, x: np.ndarray, tDst: np.ndarray) -> np.ndarray:
    """Linear interpolation of the finite rows of `x` at `tDst`; NaN outside the source range or with < 2 finite rows."""
    tOut = np.full((len(tDst), x.shape[1]), np.nan)
    tOk = np.all(np.isfinite(x), axis=1)
    if tOk.sum() < 2:
        return tOut
    for tJ in range(x.shape[1]):
        tOut[:, tJ] = np.interp(tDst, tSrc[tOk], x[tOk, tJ], left=np.nan, right=np.nan)
    return tOut


def continuousQuat(q: np.ndarray) -> np.ndarray:
    """Flip quaternion signs so consecutive valid samples lie in the same hemisphere."""
    q = q.copy()
    tPrevious = None
    for tI in range(len(q)):
        if not np.all(np.isfinite(q[tI])):
            continue
        if tPrevious is not None and np.dot(q[tI], tPrevious) < 0:
            q[tI] = -q[tI]
        tPrevious = q[tI]
    return q


def relative(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, Rotation | None, np.ndarray]:
    """(positions minus the first valid one, rotations relative to the first valid one, valid mask)."""
    tOk = np.all(np.isfinite(p), axis=1) & np.all(np.isfinite(q), axis=1)
    if not tOk.any():
        return p, None, tOk
    tI0 = np.flatnonzero(tOk)[0]
    tR0 = Rotation.from_quat(q[tI0])
    tQ = q.copy()
    tQ[~tOk] = IDENTITY_QUAT
    tRel = tR0.inv() * Rotation.from_quat(tQ / np.linalg.norm(tQ, axis=1, keepdims=True))
    return p - p[tI0], tRel, tOk


def compareHand(ref: dict, test: dict, hand: str, t0: float, t1: float, maxLagS: float, lagStepS: float) -> dict:
    """Metrics of one hand (see the module docstring), or {"error": "no overlap"}."""
    tT = ref["t"]
    tWindow = (tT >= t0) & (tT <= t1)
    tRefT = tT[tWindow]
    tPA, tQA = trajio.position(ref, hand)[tWindow], trajio.quat(ref, hand)[tWindow]
    tPARel, tRA, tOkA = relative(tPA, tQA)

    tBest = None
    for tLag in np.arange(0.0, maxLagS + LAG_RANGE_EPS_S, lagStepS):
        tPC = resample(test["t"], trajio.position(test, hand), tRefT + tLag)
        tOk = tOkA & np.all(np.isfinite(tPC), axis=1)
        if tOk.sum() < MIN_OVERLAP_FRAMES:
            continue
        tI0 = np.flatnonzero(np.all(np.isfinite(tPC), axis=1))[0]
        tError = np.linalg.norm((tPC - tPC[tI0])[tOk] - tPARel[tOk], axis=1)
        tMse = float(np.mean(tError**2))
        if tBest is None or tMse < tBest[0]:
            tBest = (tMse, tLag)
    if tBest is None:
        return {"error": "no overlap"}
    tLag = tBest[1]

    tPC = resample(test["t"], trajio.position(test, hand), tRefT + tLag)
    tQC = resample(test["t"], continuousQuat(trajio.quat(test, hand)), tRefT + tLag)  # nlerp after normalisation
    tPCRel, tRC, tOkC = relative(tPC, tQC)
    tOk = tOkA & tOkC
    tError = MM_PER_M * np.linalg.norm(tPCRel[tOk] - tPARel[tOk], axis=1)
    tAngle = np.degrees((tRA[tOk].inv() * tRC[tOk]).magnitude())
    tAmpA = np.sqrt(np.mean(np.sum(tPARel[tOk] ** 2, axis=1)))
    tAmpC = np.sqrt(np.mean(np.sum(tPCRel[tOk] ** 2, axis=1)))
    return {
        "lag_s": round(float(tLag), 3),
        "pos_rmse_mm": float(np.sqrt(np.mean(tError**2))),
        "pos_p95_mm": float(np.percentile(tError, P95)),
        "pos_max_mm": float(tError.max()),
        "rot_median_deg": float(np.median(tAngle)),
        "rot_p95_deg": float(np.percentile(tAngle, P95)),
        "amplitude_ratio": float(tAmpC / tAmpA) if tAmpA > 0 else float("nan"),
        "coverage": float(tOk.mean()),
        "reference_range_mm": float(MM_PER_M * np.nanmax(np.linalg.norm(tPARel, axis=1))),
    }


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    tParser.add_argument("ref", help="reference trajectory CSV (e.g. human)")
    tParser.add_argument("test", help="trajectory to evaluate (e.g. robot)")
    tParser.add_argument("--t0", type=float, default=-np.inf)
    tParser.add_argument("--t1", type=float, default=np.inf)
    tParser.add_argument("--max-lag", type=float, default=DEFAULT_MAX_LAG_S, help="lag search range [s]")
    tParser.add_argument("--lag-step", type=float, default=DEFAULT_LAG_STEP_S, help="lag search step [s]")
    tParser.add_argument("--only-measured", action="store_true",
                         help="ignore reference frames that were not measured (visible == 0 or filled != 0)")
    tParser.add_argument("--json", help="write the metrics to this JSON file")
    return tParser


def run(argv: list[str] | None = None) -> dict:
    """The command line: compare every hand present in both files, print, optionally write JSON; returns {hand: metrics}."""
    tArgs = buildParser().parse_args(argv)
    tRef, tTest = trajio.load(tArgs.ref), trajio.load(tArgs.test)
    tResults = {}
    for tHand in trajio.handsIn(tRef):
        if f"{tHand}_x" not in tTest:
            continue
        if tArgs.only_measured:
            tBad = tRef.get(f"{tHand}_visible", np.ones_like(tRef["t"])) < VISIBLE_MIN
            if f"{tHand}_filled" in tRef:
                tBad |= tRef[f"{tHand}_filled"] != 0
            for tKey in trajio.POSE:
                tRef[f"{tHand}_{tKey}"] = np.where(tBad, np.nan, tRef[f"{tHand}_{tKey}"])
        tResults[tHand] = compareHand(tRef, tTest, tHand, tArgs.t0, tArgs.t1, tArgs.max_lag, tArgs.lag_step)
        tRow = tResults[tHand]
        if "error" in tRow:
            print(f"{tHand}: {tRow['error']}")
            continue
        print(f"{tHand}: lag {tRow['lag_s']:.2f} s  pos RMSE {tRow['pos_rmse_mm']:.1f} / p95 {tRow['pos_p95_mm']:.1f} mm  "
              f"rot median {tRow['rot_median_deg']:.1f} / p95 {tRow['rot_p95_deg']:.1f} deg  "
              f"amplitude {tRow['amplitude_ratio']:.3f}  coverage {PERCENT*tRow['coverage']:.0f} %")
    if tArgs.json:
        Path(tArgs.json).parent.mkdir(parents=True, exist_ok=True)
        Path(tArgs.json).write_text(json.dumps(tResults, indent=JSON_INDENT))
    return tResults


def main(argv: list[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
