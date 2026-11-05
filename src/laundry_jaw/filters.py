"""Time-series helpers shared by the jaw labellers: the odd sliding-median window that spans a given duration of frames, and
the sliding median itself (edges use the shorter window that fits)."""
from __future__ import annotations

import numpy as np

NS_PER_S = 1e9
# Guards the frame-rate estimate of a take whose first and last timestamps coincide.
MIN_SPAN_S = 1e-9


def medianWindowLen(tNs: np.ndarray, windowS: float, count: int) -> int:
    """Odd number of frames covering ~`windowS` at the mean frame rate of the first `count` stamps of `tNs` (at least 1)."""
    return max(1, int(round(windowS * (count - 1) / max((tNs[-1] - tNs[0]) / NS_PER_S, MIN_SPAN_S)))) | 1


def slidingMedian(values: np.ndarray, window: int, dtype=None) -> np.ndarray:
    """Median of values[i - window // 2 : i + window // 2 + 1] (clipped at the start) for every i."""
    return np.array([np.median(values[max(0, tI - window // 2):tI + window // 2 + 1]) for tI in range(len(values))], dtype)
