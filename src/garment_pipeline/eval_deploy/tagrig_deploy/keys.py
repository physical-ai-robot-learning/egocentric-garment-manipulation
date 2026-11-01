"""Terminal keys (SPACE / q / r / y) for the CLI run; KeysCtl is the CLI's run control (the UI has its own).
"""
from __future__ import annotations
import os
import threading
import time


# ---------------------------------------------------------------- keyboard (tag_rig_record /dev/tty cbreak pattern)
class Keys:
    def __init__(self):
        self.q = []; self.fd = None
        try:
            import termios, tty, atexit
            self.fd = os.open("/dev/tty", os.O_RDONLY | os.O_NONBLOCK); old = termios.tcgetattr(self.fd); tty.setcbreak(self.fd)
            atexit.register(lambda: termios.tcsetattr(self.fd, termios.TCSADRAIN, old))
            threading.Thread(target=self._loop, daemon=True).start()
        except Exception as e: print(f"[배포] 키 입력 불가 (/dev/tty: {e}) -- Ctrl+C = 정지", flush=True)

    def _loop(self):
        import select
        while True:
            r, _, _ = select.select([self.fd], [], [], 0.1)
            if not r: continue
            try: data = os.read(self.fd, 16).decode(errors="ignore")
            except BlockingIOError: continue
            self.q.extend(data)

    def pop(self):
        out, self.q = self.q, []; return out

    def wait(self, valid, timeout=None):
        t0 = time.monotonic()
        while timeout is None or time.monotonic() - t0 < timeout:
            for ch in self.pop():
                if ch.lower() in valid: return ch.lower()
            time.sleep(0.05)
        return None


class KeysCtl:
    """CLI control: SPACE / q from this terminal stop the run"""
    def __init__(self): self.keys = Keys()

    def poll(self):
        ks = self.keys.pop()
        if " " in ks: return "SPACE"
        if "q" in ks or "Q" in ks: return "q"
        return None

    def publish(self, snap): pass
