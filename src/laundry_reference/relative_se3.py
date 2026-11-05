"""Relative SE(3) trajectory representation (the learning contract used in this project).

Was scripts/relative_se3.py, then laundry_reference/pose/relative_se3.py.

Absolute poses T(t) are stored in one world frame (the robot base frame). A
learning target is expressed relative to a reference pose T_0 (e.g. the pose at
the start of an action chunk):

    T_rel(t) = T_0^{-1} T(t)

so the target does not depend on where the motion happens in the workspace,
and the same contract applies to the human demonstration and to the robot.
This module converts between the two and verifies the round trip on a CSV (input: trajectory CSV; output: printed check).

Usage:
    python -m laundry_reference.relative_se3 assets/laundry/examples/synthetic_bimanual_demo.csv --chunk 1.0
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from . import trajio

DEFAULT_CHUNK_S = 1.0
MM_PER_M = 1000


def toMatrix(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """(N, 4, 4) homogeneous poses from positions (N, 3) and quaternions xyzw (N, 4)."""
    tT = np.tile(np.eye(4), (len(p), 1, 1))
    tT[:, :3, :3] = Rotation.from_quat(q).as_matrix()
    tT[:, :3, 3] = p
    return tT


def toRelative(poses: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """T_rel = T0^{-1} T for a batch `poses` T (N,4,4) and one `reference` pose T0 (4,4)."""
    return np.linalg.inv(reference)[None] @ poses


def fromRelative(relativePoses: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """T = T0 T_rel: the inverse of `toRelative`."""
    return reference[None] @ relativePoses


def chunks(poses: np.ndarray, times: np.ndarray, horizon: float) -> list[tuple[int, np.ndarray]]:
    """Split a trajectory into chunks of ``horizon`` seconds, each relative to its first pose."""
    tOut = []
    tI = 0
    while tI < len(times):
        tJ = int(np.searchsorted(times, times[tI] + horizon))
        tJ = max(tJ, tI + 1)
        tOut.append((tI, toRelative(poses[tI:tJ], poses[tI])))
        tI = tJ
    return tOut


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    tParser.add_argument("src", help="trajectory CSV (gaps must be filled or will be skipped)")
    tParser.add_argument("--chunk", type=float, default=DEFAULT_CHUNK_S, help="chunk horizon [s]")
    return tParser


def run(argv: list[str] | None = None) -> float:
    """The command line: chunk, reconstruct, print per hand; returns the worst absolute round-trip error over both hands."""
    tArgs = buildParser().parse_args(argv)
    tData = trajio.load(tArgs.src)
    tWorst = 0.0
    for tHand in trajio.handsIn(tData):
        tP, tQ = trajio.position(tData, tHand), trajio.quat(tData, tHand)
        tOk = np.all(np.isfinite(tP), axis=1) & np.all(np.isfinite(tQ), axis=1)
        tTimes, tT = tData["t"][tOk], toMatrix(tP[tOk], tQ[tOk])
        tChunks = chunks(tT, tTimes, tArgs.chunk)
        tRebuilt = np.concatenate([fromRelative(tRel, tT[tI]) for tI, tRel in tChunks])
        tError = float(np.abs(tRebuilt - tT).max())
        tWorst = max(tWorst, tError)
        tEndMm = [MM_PER_M * np.linalg.norm(tRel[-1, :3, 3]) for _, tRel in tChunks]
        print(f"{tHand}: {len(tChunks)} chunks of {tArgs.chunk:.1f} s, end-of-chunk translation median {np.median(tEndMm):.1f} mm, "
              f"round-trip error {tError:.2e}")
    return tWorst


def main(argv: list[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
