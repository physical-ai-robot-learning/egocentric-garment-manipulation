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

JAW_MID = 0.5  # jaw command above = open side, at or below = closed side (the binary threshold scale of the jaw command)


class Streamer(threading.Thread):
    def __init__(self, bridge, q0: dict, g0: dict, hz: float, dt: float | None = None, gripHoldS: float = 0.0,
                 execLeadS: dict | None = None, jawLeadS: dict | None = None, floorGuard=None) -> None:
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
        self.mGTimeline = {}  # 11-06 --grip-at row: jaw command per tick (empty = one jaw command mG for the whole timeline)
        # 11-06 user (jaw open-close-open-close: every new plan moved the close row later): a jaw that crossed open <-> closed keeps
        # that side for gripHoldS seconds before it may cross back (0 = off)
        self.mGripHoldS = float(gripHoldS)
        self.mJawOut = {tSide: float(g0[tSide]) for tSide in SIDES}
        self.mJawSwitchT = {tSide: -1e9 for tSide in SIDES}
        self.mDryTicks = 0
        # 11-06 UMI latency matching (tools.latency_probe): every side's joints / jaw are SENT this far ahead on the timeline so the
        # motor gets there at the tick's own time; anchor() stays the physical time (the pose the robot is at when observed)
        self.mExecLeadS = {tSide: float((execLeadS or {}).get(tSide, 0.0)) for tSide in SIDES}
        self.mJawLeadS = {tSide: float((jawLeadS or {}).get(tSide, 0.0)) for tSide in SIDES}
        # 11-06 core.contact: floorGuard(side, q) -> True when q puts that TCP below its (contact-raised) floor; the side then keeps
        # its last sent q until the timeline comes back above it (the next plan is already clamped to the new floor)
        self.mFloorGuard = floorGuard
        self.mGuardHolds = 0

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

    def merge(self, i0: int, rows: list, g: dict, gRows: list | None = None):
        """rows[j] = q for tick i0 + 1 + j -> (rows dropped as already driven, rows blended). gRows[j] = jaw command of the same tick
        (--grip-at row); None = g for the whole timeline."""
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
            # 11-06: keep the newest target -- a merge whose rows were all already driven (commit 1, plan latency > 1 row) emptied
            # the timeline and lastIndex / qAt raised "max() iterable argument is empty" on the first GO cycle
            tNewest = max(self.mTimeline)
            for tI in [tI for tI in self.mTimeline if tI < tEx - 1 and tI != tNewest]:
                del self.mTimeline[tI]
            self.mG = dict(g)
            if gRows is None:
                self.mGTimeline = {}
            else:
                self.mGTimeline.update({i0 + 1 + tJ: dict(tG) for tJ, tG in enumerate(gRows) if i0 + 1 + tJ > tEx})
                for tI in [tI for tI in self.mGTimeline if tI < tEx - 1]:
                    del self.mGTimeline[tI]
            return len(rows) - len(tNew), len(tOverlap)

    def setTimeline(self, waypoints: list, durEach: float, g: dict, gRows: list | None = None) -> None:
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
            self.mGTimeline = {} if gRows is None else {tI + 1: dict(tG) for tI, tG in enumerate(gRows)}

    def _currentTarget(self):
        with self.mLock:
            tNow = time.monotonic()
            tX = self.tick(tNow)
            if tX > self.lastIndex:
                self.mDryTicks += 1  # timeline ran dry: hold the last target (counted at stream rate)
            tQ = {tSide: self.qAt(self.tick(tNow + self.mExecLeadS[tSide]))[tSide] for tSide in SIDES}
            if self.mFloorGuard is not None and self.mLast is not None:
                for tSide in SIDES:
                    if self.mFloorGuard(tSide, tQ[tSide]):
                        tQ[tSide] = self.mLast[tSide]
                        self.mGuardHolds += 1
            tG = dict(self.mG)
            if self.mGTimeline:  # --grip-at row: the jaw command of the tick being driven toward
                for tSide in SIDES:
                    tXg = self.tick(tNow + self.mJawLeadS[tSide])
                    tKeys = [tI for tI in self.mGTimeline if tI <= np.ceil(tXg)]
                    if tKeys:
                        tG[tSide] = self.mGTimeline[max(tKeys)][tSide]
            return tQ, self._holdJaw(tG)

    def _holdJaw(self, g: dict) -> dict:
        """jaw re-open debounce: a jaw that CLOSED may not re-open for mGripHoldS seconds; closing is never delayed (11-06 user: the
        jaw closed late, after the arm had already started lifting)"""
        if self.mGripHoldS <= 0:
            return g
        tNow = time.monotonic()
        tOut = {}
        for tSide in SIDES:
            tNew, tOld = float(g[tSide]), self.mJawOut[tSide]
            if (tNew > JAW_MID) != (tOld > JAW_MID):
                if tNew > JAW_MID and tNow - self.mJawSwitchT[tSide] < self.mGripHoldS:
                    tNew = tOld  # too soon after the last switch: hold
                else:
                    self.mJawSwitchT[tSide] = tNow
            self.mJawOut[tSide] = tNew
            tOut[tSide] = tNew
        return tOut

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
