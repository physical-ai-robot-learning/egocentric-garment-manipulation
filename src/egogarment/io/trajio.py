"""Small I/O helpers for the bimanual trajectory CSV format used in this repository.

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


def load(path: str | Path) -> dict[str, np.ndarray]:
    """Load a trajectory CSV into a dict of float arrays keyed by column name."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(v) if v not in ("", "nan", "NaN") else np.nan for v in r] for r in reader]
    arr = np.asarray(rows, dtype=float).reshape(-1, len(header))
    return {name: arr[:, i].copy() for i, name in enumerate(header)}


def save(path: str | Path, data: dict[str, np.ndarray]) -> None:
    """Save a dict of equal-length 1-D arrays as CSV (``t`` first)."""
    cols = ["t"] + [c for c in data if c != "t"]
    n = len(data["t"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n):
            w.writerow([_fmt(data[c][i]) for c in cols])


def _fmt(v: float) -> str:
    if np.isnan(v):
        return "nan"
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.6f}"


def position(data: dict, hand: str) -> np.ndarray:
    return np.stack([data[f"{hand}_{k}"] for k in ("x", "y", "z")], axis=1)


def quat(data: dict, hand: str) -> np.ndarray:
    return np.stack([data[f"{hand}_{k}"] for k in ("qx", "qy", "qz", "qw")], axis=1)


def set_pose(data: dict, hand: str, p: np.ndarray, q: np.ndarray | None = None) -> None:
    for i, k in enumerate(("x", "y", "z")):
        data[f"{hand}_{k}"] = p[:, i].copy()
    if q is not None:
        for i, k in enumerate(("qx", "qy", "qz", "qw")):
            data[f"{hand}_{k}"] = q[:, i].copy()


def hands_in(data: dict) -> list[str]:
    return [h for h in HANDS if f"{h}_x" in data]
