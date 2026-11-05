"""Thread-safe ring buffer of (host monotonic ns, value) samples: camera frames and robot reads of the deploy loop."""

from __future__ import annotations
import collections
import threading

DEFAULT_LENGTH = 12
NS_PER_S = 1e9


class Ring:
    def __init__(self, n: int = DEFAULT_LENGTH) -> None:
        self.mBuf = collections.deque(maxlen=n)
        self.mLock = threading.Lock()

    def put(self, tNs: int, value) -> None:
        with self.mLock:
            self.mBuf.append((tNs, value))

    def latest(self):
        with self.mLock:
            return self.mBuf[-1] if self.mBuf else (0, None)

    def near(self, tNs: int, tolS: float):
        """Entry closest to tNs if within tolS, else None."""
        with self.mLock:
            tEntries = list(self.mBuf)
        if not tEntries:
            return None
        tT, tValue = min(tEntries, key=lambda tEntry: abs(tEntry[0] - tNs))
        return (tT, tValue) if abs(tT - tNs) <= tolS * NS_PER_S else None
