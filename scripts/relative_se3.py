"""Relative SE(3) trajectory representation (the learning contract used in this project).

Absolute poses T(t) are stored in one world frame (the robot base frame). A
learning target is expressed relative to a reference pose T_0 (e.g. the pose at
the start of an action chunk):

    T_rel(t) = T_0^{-1} T(t)

so the target does not depend on where the motion happens in the workspace,
and the same contract applies to the human demonstration and to the robot.
This module converts between the two and verifies the round trip on a CSV.

Usage:
    python scripts/relative_se3.py examples/synthetic_bimanual_demo.csv --chunk 1.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trajio  # noqa: E402


def to_matrix(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    T = np.tile(np.eye(4), (len(p), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(q).as_matrix()
    T[:, :3, 3] = p
    return T


def to_relative(T: np.ndarray, T0: np.ndarray) -> np.ndarray:
    """T_rel = T0^{-1} T for a batch T (N,4,4) and one reference T0 (4,4)."""
    return np.linalg.inv(T0)[None] @ T


def from_relative(T_rel: np.ndarray, T0: np.ndarray) -> np.ndarray:
    return T0[None] @ T_rel


def chunks(T: np.ndarray, t: np.ndarray, horizon: float) -> list[tuple[int, np.ndarray]]:
    """Split a trajectory into chunks of ``horizon`` seconds, each relative to its first pose."""
    out = []
    i = 0
    while i < len(t):
        j = int(np.searchsorted(t, t[i] + horizon))
        j = max(j, i + 1)
        out.append((i, to_relative(T[i:j], T[i])))
        i = j
    return out


def main(argv: list[str] | None = None) -> float:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", help="trajectory CSV (gaps must be filled or will be skipped)")
    ap.add_argument("--chunk", type=float, default=1.0, help="chunk horizon [s]")
    a = ap.parse_args(argv)
    data = trajio.load(a.src)
    worst = 0.0
    for h in trajio.hands_in(data):
        p, q = trajio.position(data, h), trajio.quat(data, h)
        ok = np.all(np.isfinite(p), axis=1) & np.all(np.isfinite(q), axis=1)
        t, T = data["t"][ok], to_matrix(p[ok], q[ok])
        ch = chunks(T, t, a.chunk)
        rec = np.concatenate([from_relative(Tr, T[i]) for i, Tr in ch])
        err = float(np.abs(rec - T).max())
        worst = max(worst, err)
        mags = [1000 * np.linalg.norm(Tr[-1, :3, 3]) for _, Tr in ch]
        print(f"{h}: {len(ch)} chunks of {a.chunk:.1f} s, end-of-chunk translation median {np.median(mags):.1f} mm, "
              f"round-trip error {err:.2e}")
    return worst


if __name__ == "__main__":
    main()
