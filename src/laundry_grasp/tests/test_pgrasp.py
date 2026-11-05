"""The causal p_grasp(t) rule: window, restart at each jaw state change, causality, and equality with the original list form."""
from __future__ import annotations

import numpy as np
import pytest

from laundry_grasp import pgrasp

NS = 1_000_000_000


def _reference(ts, fp, events, windowS):
    """the 11-05 scripts' expression, kept verbatim as the oracle"""
    chg = np.array([t for t, _ in events])
    last = np.array([chg[chg <= t].max() if (chg <= t).any() else ts[0] for t in ts])
    return np.array([fp[(ts >= max(last[i], ts[i] - windowS * 1e9)) & (ts <= ts[i])].mean() for i in range(len(ts))])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("windowS", [0.2, 0.4, 0.6])
def test_matches_reference_bit_for_bit(dtype, windowS):
    tRng = np.random.default_rng(3)
    tTs = pgrasp.tickTimes(5 * NS, 25 * NS)
    tFp = tRng.random(len(tTs)).astype(dtype)
    tEvents = [(int(6.3 * NS), "close"), (int(9.0 * NS), "open"), (int(9.05 * NS), "close"), (int(20 * NS), "open")]
    tLast = pgrasp.lastChangeTimes(tTs, pgrasp.changeTimes(tEvents))
    tP = pgrasp.causalWindowMean(tTs, tFp, tLast, windowS)
    tRef = _reference(tTs, tFp, tEvents, windowS)
    assert tP.dtype == tRef.dtype
    assert np.array_equal(tP, tRef)


def test_window_restarts_at_jaw_change_and_is_causal():
    tTs = pgrasp.tickTimes(0, 4 * NS)                  # 15 Hz
    tChange = int(2 * NS)
    tFp = np.where(tTs < tChange, 0.0, 1.0)            # classifier flips exactly at the jaw change
    tP = pgrasp.causalWindowMean(tTs, tFp, pgrasp.lastChangeTimes(tTs, [tChange]), 0.4)
    tAfter = tTs >= tChange
    # restart: the first tick after the change already sees only post-change frames
    assert np.all(tP[tAfter] == 1.0)
    assert np.all(tP[~tAfter] == 0.0)
    # causal: changing a future frame does not change earlier p
    tFp2 = tFp.copy()
    tFp2[-1] = 0.0
    tP2 = pgrasp.causalWindowMean(tTs, tFp2, pgrasp.lastChangeTimes(tTs, [tChange]), 0.4)
    assert np.array_equal(tP[:-1], tP2[:-1]) and tP2[-1] < 1.0


def test_window_length_without_change():
    tTs = pgrasp.tickTimes(0, 2 * NS)
    tFp = np.arange(len(tTs), dtype=np.float64)
    tP = pgrasp.causalWindowMean(tTs, tFp, pgrasp.lastChangeTimes(tTs, []), 0.4)
    # 0.4 s at 15 Hz: ticks t-0.4 .. t -> 7 values once the series is long enough (t - 0.4 s lands on a tick boundary +- 1 ns)
    tI = 20
    tLow = np.searchsorted(tTs, tTs[tI] - 0.4 * NS)
    assert tP[tI] == tFp[tLow:tI + 1].mean()
    assert tP[0] == 0.0                                 # first tick: only itself


def test_last_change_before_first_event_is_first_tick():
    tTs = np.array([10, 20, 30, 40], np.int64)
    assert pgrasp.lastChangeTimes(tTs, [25, 5 + 30]).tolist() == [10, 10, 25, 35]
    assert pgrasp.lastChangeTimes(tTs, np.array([35, 25])).tolist() == [10, 10, 25, 35]    # unsorted input


def test_closed_at_ticks():
    tJawT = np.array([0, 10, 20, 30], np.int64)
    tState = np.array([1, 0, 0, 1])
    assert pgrasp.closedAt(tJawT, tState, np.array([-5, 5, 10, 25, 99])).tolist() == [False, True, True, False, False]
