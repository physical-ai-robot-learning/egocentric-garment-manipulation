"""The bimanual trajectory CSV format every laundry_reference tool reads and writes.

Was scripts/trajio.py, then laundry_reference/io/trajio.py.

Format (one row per frame, wide, per-hand prefix ``R_`` / ``L_``)::

    t, R_x, R_y, R_z, R_qx, R_qy, R_qz, R_qw, R_jaw, R_visible, L_x, ... , L_visible

* positions in metres, in a single world frame (here: the robot base frame,
  +x forward, +y left, +z up, table plane at z = 0);
* orientation as a unit quaternion (x, y, z, w);
* ``jaw``: 1 = open, 0 = closed;
* ``visible``: 1 if the pose was measured in this frame (marker seen), 0 otherwise.
  Unmeasured poses are NaN.

Extra per-hand columns written by the tools (``filled``, ``disp_mm``, ``quality`` ...)
are carried along unchanged.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

HANDS = ("R", "L")
POSE = ("x", "y", "z", "qx", "qy", "qz", "qw")
POSITION_KEYS = ("x", "y", "z")
QUAT_KEYS = ("qx", "qy", "qz", "qw")
NAN_CELLS = ("", "nan", "NaN")


def load(path: str | Path) -> dict[str, np.ndarray]:
    """Load a trajectory CSV into a dict of float arrays keyed by column name."""
    with open(path, newline="") as tHandle:
        tReader = csv.reader(tHandle)
        tHeader = next(tReader)
        tRows = [[float(tCell) if tCell not in NAN_CELLS else np.nan for tCell in tRow] for tRow in tReader]
    tTable = np.asarray(tRows, dtype=float).reshape(-1, len(tHeader))
    return {tName: tTable[:, tI].copy() for tI, tName in enumerate(tHeader)}


def save(path: str | Path, data: dict[str, np.ndarray]) -> None:
    """Save a dict of equal-length 1-D arrays as CSV (``t`` first)."""
    tColumns = ["t"] + [tName for tName in data if tName != "t"]
    tCount = len(data["t"])
    with open(path, "w", newline="") as tHandle:
        tWriter = csv.writer(tHandle)
        tWriter.writerow(tColumns)
        for tI in range(tCount):
            tWriter.writerow([_fmt(data[tName][tI]) for tName in tColumns])


def _fmt(value: float) -> str:
    """'nan', an integer without decimals, else 6 decimals."""
    if np.isnan(value):
        return "nan"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}"


def position(data: dict, hand: str) -> np.ndarray:
    return np.stack([data[f"{hand}_{tKey}"] for tKey in POSITION_KEYS], axis=1)


def quat(data: dict, hand: str) -> np.ndarray:
    return np.stack([data[f"{hand}_{tKey}"] for tKey in QUAT_KEYS], axis=1)


def setPose(data: dict, hand: str, p: np.ndarray, q: np.ndarray | None = None) -> None:
    """Write positions `p` (N, 3) and, when given, quaternions `q` (N, 4) into the hand's columns (copies)."""
    for tI, tKey in enumerate(POSITION_KEYS):
        data[f"{hand}_{tKey}"] = p[:, tI].copy()
    if q is not None:
        for tI, tKey in enumerate(QUAT_KEYS):
            data[f"{hand}_{tKey}"] = q[:, tI].copy()


def handsIn(data: dict) -> list[str]:
    return [tHand for tHand in HANDS if f"{tHand}_x" in data]
