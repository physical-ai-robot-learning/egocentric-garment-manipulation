"""Runtime state of one connected arm of the MIT bridge (plugin handles, measured / commanded / goal pose, segment tracker,
park and recovery bookkeeping). Mutated by the control thread and the HTTP handlers under the bridge lock.
Deploy stage: robot bridge.
"""

from __future__ import annotations

# A goal "never sent": far enough in the past that the first goal after connect is never treated as a segment continuation.
NO_GOAL_YET_S = -1e9


class ArmState:
    """mGoal / mCmd / mQ: raw motor deg (6,); jaw values are the plugin's norm. mSegEnd is mGoal (same array) while a goal
    segment is being tracked; any other writer of mGoal replaces the array, which drops the segment."""

    def __init__(self, rebot, fol, cfg) -> None:
        self.mRebot = rebot
        self.mFol = fol
        self.mCfg = cfg
        self.mGoal = None
        self.mJawGoal = None
        self.mCmd = None
        self.mQ = None
        self.mJawNorm = None
        self.mTRead = None
        self.mBlocked = None
        self.mVel = None
        self.mParked = None  # None = never parked, True / False after a park / unpark
        self.mSegStart = None
        self.mSegEnd = None
        self.mSegT = 0.0
        self.mTSeg = 0.0
        self.mTGoal = NO_GOAL_YET_S
        self.mGapEma = None
        self.mReconnecting = False
        self.mLastReconnect = 0.0
        self.mLastRecover = 0.0

    def seedAt(self, q, jawNorm, tRead: float) -> None:
        """Hold at the measured pose: goal = command = measured."""
        self.mQ = q
        self.mJawNorm = jawNorm
        self.mGoal = q.copy()
        self.mCmd = q.copy()
        self.mJawGoal = jawNorm
        self.mTRead = tRead
