"""Runs of True in a boolean mask (gaps, invalid stretches, collisions) and their durations.

New in the 11-04 refactor: was the runs / runs_s helpers copied into gate, solve2, episode_quality, fusion_metrics,
build_candidate_v0 and imu_motion.
"""
from __future__ import annotations

import numpy as np

from .constants import NS_PER_S


def trueRuns(mask) -> list[tuple[int, int]]:
    """[(start, end)) index pairs of every run of True"""
    tEdges = np.diff(np.r_[0, np.asarray(mask).astype(int), 0])
    return list(zip(np.flatnonzero(tEdges == 1), np.flatnonzero(tEdges == -1)))


def runDurationsS(mask, tNs, padS: float) -> list[float]:
    """duration of every run of True: first-to-last sample time + padS (one sample period, so a 1-sample run lasts padS)"""
    return [(tNs[tEnd - 1] - tNs[tStart]) / NS_PER_S + padS for tStart, tEnd in trueRuns(mask)]
