"""Run control object of one UI-started run (handed to session.run_loop.runSession as `ctl`): the stop request (button / Space /
watchdog), the GO heartbeat, [한 스텝 실행] step tokens and the latest snapshot + k1 jaw history for the views."""

from __future__ import annotations
import threading
import time

from .settings import HB_TIMEOUT_S

K1_HISTORY = 400  # k1 jaw samples kept (the page shows the last 300)


class UiRunControl:
    """Was UICtl."""

    def __init__(self, go: bool) -> None:
        self.mGo = go
        self.mStopReq = None
        self.mHb = time.monotonic()
        self.mSnap = None
        self.mK1 = []
        self.mLock = threading.Lock()
        self.mStepMode = False  # 10-28 step mode: [한 스텝 실행] sets ONE token
        self.mStepN = 0
        self.mStepsDone = 0

    def takeStep(self) -> bool:
        with self.mLock:
            if self.mStepN <= 0:
                return False
            self.mStepN -= 1
            self.mStepsDone += 1
            return True

    def poll(self):
        if self.mStopReq:
            return self.mStopReq
        if self.mGo and time.monotonic() - self.mHb > HB_TIMEOUT_S:
            self.mStopReq = f"UI heartbeat lost (> {HB_TIMEOUT_S:.0f} s)"
            return self.mStopReq
        return None

    def publish(self, snapshot: dict) -> None:
        with self.mLock:
            self.mSnap = snapshot
            self.mK1.append((snapshot["t"], float(snapshot["grip"][0, 0]), float(snapshot["grip"][0, 1])))
            self.mK1 = self.mK1[-K1_HISTORY:]
