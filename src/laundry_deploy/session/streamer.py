"""30 Hz command streamer of a --go run: sends q interpolated along an INDEXED timeline of joint targets (tick i = tBase + i * dt).

[10-29] indy7 c4fba14 (bh_indy7_LeRobot inference/action_timeline.merge_chunk): a chunk row is keyed by the tick it was predicted
for; merge() drops rows for ticks already driven (no pulling back along the covered path), blends rows for ticks still pending
(crossfade, new weight rising) and appends the rest.  setTimeline() = restart the timeline from where the stream is now (recovery,
step).
"""

from __future__ import annotations
import bisect
import threading
import time

import numpy as np

from ..core.contract import ROW_DT_S, SIDES


class Streamer(threading.Thread):
    def __init__(self, bridge, q0: dict, g0: dict, hz: float, dt: float | None = None) -> None:
        super().__init__(daemon=True)
        self.mBridge, self.mHz = bridge, hz
        self.mG = dict(g0)
        self.mLock = threading.Lock()
        self.mStop = False
        self.mError = None
        self.mDt = float(dt or ROW_DT_S)  # tick period (look-ahead: the policy's row period / speed)
        self.mSent = 0
        self.mLast = None
        self.mTBase = time.monotonic()
        self.mTimeline = {0: {tSide: q0[tSide].copy() for tSide in SIDES}}
        self.mDryTicks = 0

    def tick(self, t: float | None = None) -> float:
        return ((time.monotonic() if t is None else t) - self.mTBase) / self.mDt

    def executed(self, t: float | None = None) -> int:
        return int(np.floor(self.tick(t)))

    @property
    def lastIndex(self) -> int:
        return max(self.mTimeline)

    def qAt(self, x: float) -> dict:
        """Timeline joints at (fractional) tick x; caller holds mLock."""
        tKeys = sorted(self.mTimeline)
        x = max(x, tKeys[0])
        if x >= tKeys[-1]:
            return self.mTimeline[tKeys[-1]]
        tJ = bisect.bisect_right(tKeys, x) - 1
        tI0, tI1 = tKeys[tJ], tKeys[tJ + 1]
        tF = (x - tI0) / (tI1 - tI0)
        return {
            tSide: self.mTimeline[tI0][tSide] + tF * (self.mTimeline[tI1][tSide] - self.mTimeline[tI0][tSide]) for tSide in SIDES
        }

    def plannedAfter(self, tick: int, rows: int) -> list:
        """The queued joints of ticks tick+1 .. tick+rows that are still on the timeline (the previous plan for RTC)."""
        with self.mLock:
            return [self.qAt(tick + tK) for tK in range(1, rows + 1) if tick + tK <= self.lastIndex]

    def anchor(self, t: float):
        """The tick index at time t and the COMMANDED q at that tick (the pose a chunk predicted from an observation at t starts from)."""
        with self.mLock:
            tI = self.executed(t)
            return tI, {tSide: tV.copy() for tSide, tV in self.qAt(tI).items()}

    def merge(self, i0: int, rows: list, g: dict):
        """rows[j] = q for tick i0 + 1 + j -> (rows dropped as already driven, rows blended)."""
        with self.mLock:
            tEx = self.executed() + 1  # the tick being interpolated toward right now is left untouched
            tNew = {i0 + 1 + tJ: tRow for tJ, tRow in enumerate(rows) if i0 + 1 + tJ > tEx}
            tOverlap = sorted(tI for tI in tNew if tI in self.mTimeline)
            for tM, tI in enumerate(tOverlap):
                tW = (tM + 1) / (len(tOverlap) + 1)
                self.mTimeline[tI] = {tSide: (1 - tW) * self.mTimeline[tI][tSide] + tW * tNew[tI][tSide] for tSide in SIDES}
            for tI in tNew:
                if tI not in self.mTimeline:
                    self.mTimeline[tI] = tNew[tI]
            for tI in [tI for tI in self.mTimeline if tI < tEx - 1]:
                del self.mTimeline[tI]
            self.mG = dict(g)
            return len(rows) - len(tNew), len(tOverlap)

    def setTimeline(self, waypoints: list, durEach: float, g: dict) -> None:
        """waypoints: list of {side: q} (the committed steps); each reached after one tick (mDt), from where the stream is now.
        Was set()."""
        with self.mLock:
            tNow = time.monotonic()
            tQ = self.mLast if self.mLast is not None else self.qAt(self.tick(tNow))
            self.mTBase = tNow
            self.mTimeline = {0: {tSide: tQ[tSide].copy() for tSide in SIDES}}
            for tI, tWaypoint in enumerate(waypoints):
                self.mTimeline[tI + 1] = tWaypoint
            self.mG = dict(g)

    def _currentTarget(self):
        with self.mLock:
            tX = self.tick()
            if tX > self.lastIndex:
                self.mDryTicks += 1  # timeline ran dry: hold the last target (counted at stream rate)
            return self.qAt(tX), self.mG

    def run(self) -> None:
        tNext = time.monotonic()
        while not self.mStop:
            tQ, tG = self._currentTarget()
            self.mLast = {tSide: tQ[tSide].copy() for tSide in SIDES}
            tReply = self.mBridge.send(tQ, tG)
            self.mSent += 1
            if not tReply.accepted:
                self.mError = f"bridge reject: {tReply.error}"
                return
            tNext += 1.0 / self.mHz
            time.sleep(max(0.0, tNext - time.monotonic()))
