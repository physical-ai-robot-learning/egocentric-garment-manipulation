"""Terminal keys (SPACE / q / r / y) for the CLI run, read from /dev/tty in cbreak mode (tag_rig_record pattern); KeysCtl is the
CLI's run control (the UI has its own)."""

from __future__ import annotations
import os
import threading
import time

TTY = "/dev/tty"
READ_BYTES = 16
SELECT_TIMEOUT_S = 0.1
WAIT_POLL_S = 0.05
STOP_KEY = " "
QUIT_KEYS = ("q", "Q")


class Keys:
    def __init__(self) -> None:
        self.mQueue = []
        self.mFd = None
        try:
            import atexit
            import termios
            import tty

            self.mFd = os.open(TTY, os.O_RDONLY | os.O_NONBLOCK)
            tOld = termios.tcgetattr(self.mFd)
            tty.setcbreak(self.mFd)
            atexit.register(lambda: termios.tcsetattr(self.mFd, termios.TCSADRAIN, tOld))
            threading.Thread(target=self._loop, daemon=True).start()
        except Exception as tExc:
            print(f"[배포] 키 입력 불가 (/dev/tty: {tExc}) -- Ctrl+C = 정지", flush=True)

    def _loop(self) -> None:
        import select

        while True:
            tReady, _, _ = select.select([self.mFd], [], [], SELECT_TIMEOUT_S)
            if not tReady:
                continue
            try:
                tData = os.read(self.mFd, READ_BYTES).decode(errors="ignore")
            except BlockingIOError:
                continue
            self.mQueue.extend(tData)

    def pop(self) -> list:
        tOut, self.mQueue = self.mQueue, []
        return tOut

    def wait(self, valid: str, timeout: float | None = None):
        tT0 = time.monotonic()
        while timeout is None or time.monotonic() - tT0 < timeout:
            for tCh in self.pop():
                if tCh.lower() in valid:
                    return tCh.lower()
            time.sleep(WAIT_POLL_S)
        return None


class KeysCtl:
    """CLI control: SPACE / q from this terminal stop the run."""

    def __init__(self) -> None:
        self.mKeys = Keys()

    def poll(self):
        tKeys = self.mKeys.pop()
        if STOP_KEY in tKeys:
            return "SPACE"
        if any(tQuit in tKeys for tQuit in QUIT_KEYS):
            return "q"
        return None

    def publish(self, snapshot) -> None:
        pass
