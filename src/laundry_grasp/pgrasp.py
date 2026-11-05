"""The causal grasp probability p_grasp(t) (the rule shared by grasp_render_prob, grasp_ts_cv and the frozen grasp_pgrasp_gen).

Every tick t of a regular grid (15 Hz) is scored, open and closed jaw alike (11-05 user: "열려있는 동안도 판정해줘"): the per-frame
P(grasp) = 1 - P(not held) of the jaw-ROI crop at each tick, then p(t) = mean of the per-frame values of the ticks in
[max(last v2 jaw state change <= t, t - W), t]; the window restarts at every jaw open / close. Only ticks <= t are used (causal).
Before the first jaw change of the series the window starts at the first tick.
"""
from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Executor

import numpy as np

from .model import DEVICE, predict

VERSION = "grasp-v2-W0.4"             # frozen generator: name of the side files pgrasp_<VERSION>.npy and of PGRASP=<VERSION>
HZ = 15.0
WINDOW_S = 0.4                        # frozen window (grasp-v2-W0.4)
WARM_S = 0.3                          # 'judged' ticks of the CV: jaw closed and t >= close + 0.3 s
NS_PER_S = 1e9


def tickTimes(t0: float, t1: float, hz: float = HZ) -> np.ndarray:
    """int64 tick grid [t0, t1) at hz"""
    return np.arange(t0, t1, NS_PER_S / hz).astype(np.int64)


def lastChangeTimes(ts: np.ndarray, changeTimes) -> np.ndarray:
    """per tick: the latest jaw change time <= tick, else ts[0] (int64)"""
    tChanges = np.sort(np.asarray(changeTimes))
    if len(tChanges) == 0:
        return np.full(len(ts), ts[0], np.int64)
    tIdx = np.searchsorted(tChanges, ts, side="right") - 1
    return np.where(tIdx >= 0, tChanges[np.maximum(tIdx, 0)], ts[0]).astype(np.int64)


def causalWindowMean(ts: np.ndarray, framePGrasp: np.ndarray, lastChange: np.ndarray, windowS: float) -> np.ndarray:
    """p(t_i) = mean of framePGrasp over the ticks in [max(lastChange_i, t_i - windowS), t_i] (ts sorted ascending); the result has
    the dtype of framePGrasp's mean"""
    tOut = []
    for tI in range(len(ts)):
        tLow = max(lastChange[tI], ts[tI] - windowS * NS_PER_S)
        tA = np.searchsorted(ts, tLow, side="left")
        tB = np.searchsorted(ts, ts[tI], side="right")
        tOut.append(framePGrasp[tA:tB].mean())
    return np.array(tOut)


def framePGrasp(model, crops: np.ndarray, batch: int, device: str = DEVICE) -> np.ndarray:
    """per-frame P(grasp) = 1 - P(not held) of (n, 224, 224, 3) crops, scored `batch` at a time (float32)"""
    return np.concatenate([1 - predict(model, crops[tB:tB + batch][:, None], device) for tB in range(0, len(crops), batch)])


def cropsAt(crop: Callable, ts: np.ndarray, pool: Executor | None = None) -> np.ndarray:
    """crops of one camera at every tick (optionally decoded on a thread pool; the result is the same)"""
    return np.stack(list(pool.map(crop, ts)) if pool is not None else [crop(tT) for tT in ts])


def closedAt(jawTimes: np.ndarray, jawState: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """v2 jaw closed (bool) at each tick: the state of the first jaw sample at or after the tick"""
    return jawState[np.searchsorted(jawTimes, ts).clip(0, len(jawTimes) - 1)] == 0


def changeTimes(events: list[tuple[int, str]]) -> np.ndarray:
    return np.array([tT for tT, _ in events])
