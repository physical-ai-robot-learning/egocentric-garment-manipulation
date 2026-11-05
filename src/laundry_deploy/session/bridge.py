"""Robot bridge side of the deploy session: the gripper-opening guard for --go, the 25 Hz /observe poller, and the explicit
return to READY (inputs: the MIT bridge HTTP client; outputs: a Ring of (q, g_obs, estop) reads, refusals, a slow joint ramp).
"""

from __future__ import annotations
import threading
import time

import numpy as np

from ..core.contract import SIDES
from ..core.errors import HardwareError
from ..core.kinematics import linkGap
from .paths import say
from .ring import Ring

# 10-27: the 10-23 bridge ran MIT_GRIP_OPEN_FRAC_LEFT=0.5; restarted without it -> left jaw over-opened, holder broke
GRIP_OPEN_FRAC_MAX = {"left": 0.5, "right": 1.0}
GRIP_FRAC_TOL = 1e-6
STATUS_TIMEOUT_S = 2
POLL_HZ = 25.0
POLL_RING = 20
READY_PATH_SAMPLES = 30
READY_MIN_GAP_M = 0.10  # link-origin gap along the joint line to READY
READY_DEG_S = 10.0
CLEAR_ESTOP_TIMEOUT_S = 2.0


def checkGripLimits(url: str) -> None:
    """--go refuses unless the bridge limits each jaw's opening to at most GRIP_OPEN_FRAC_MAX (bridge /status grip_open_frac)."""
    import requests

    try:
        tGot = requests.get(f"{url.rstrip('/')}/status", timeout=STATUS_TIMEOUT_S).json().get("grip_open_frac") or {}
    except Exception as tExc:
        raise HardwareError(f"bridge /status 읽기 실패 ({tExc}) -- 그리퍼 열림 제한을 확인할 수 없어 GO 거부")
    tBad = {
        tSide: tGot.get(tSide)
        for tSide, tMax in GRIP_OPEN_FRAC_MAX.items()
        if tGot.get(tSide) is None or float(tGot[tSide]) > tMax + GRIP_FRAC_TOL
    }
    if tBad:
        raise HardwareError(
            f"그리퍼 열림 제한 불일치 {tBad} (허용 최대 {GRIP_OPEN_FRAC_MAX}) -- bridge 를 MIT_GRIP_OPEN_FRAC_LEFT=0.5 로 다시 띄우세요. GO 거부"
        )


class RobotPoller(threading.Thread):
    """/observe at ~25 Hz -> ring of (q, g_obs, estop); assumeReady: the given q forever (dry-run without a live robot)."""

    def __init__(self, bridge, qReady: dict, assumeReady: bool, hz: float = POLL_HZ) -> None:
        super().__init__(daemon=True)
        self.mBridge, self.mQReady, self.mFake, self.mHz = bridge, qReady, assumeReady, hz
        self.mRing = Ring(POLL_RING)
        self.mStop = False
        self.mError = None
        self.mLat = []

    def read(self):
        if self.mFake:
            return {tSide: self.mQReady[tSide].copy() for tSide in SIDES}, {tSide: None for tSide in SIDES}, False
        tObs = self.mBridge.observe()
        self.mLat.append(tObs.latency_ms)
        return tObs.q, {tSide: tObs.g(tSide) for tSide in SIDES}, tObs.estop

    def run(self) -> None:
        while not self.mStop:
            # 10-30: a bridge restart left the error "/observe: not connected" forever (never cleared after reads recovered) and
            # every later UI run stopped at once with "robot read: ..." -> clear it on the next good read
            try:
                self.mRing.put(time.monotonic_ns(), self.read())
                self.mError = None
            except Exception as tExc:
                self.mError = f"{type(tExc).__name__}: {tExc}"
                self.mRing.put(time.monotonic_ns(), (None, None, True))
            time.sleep(1.0 / self.mHz)


def goReady(bridge, arms: dict, qReady: dict) -> None:
    """Explicit user request only: /clear_estop, link-gap check along the joint line, umi_replay_bi.ramp_both at 10 deg/s."""
    from laundry_shared.visual_teleop.tools.umi_replay_bi import ramp_both

    tObs = bridge.observe()
    tQ = tObs.q
    for tU in np.linspace(0, 1, READY_PATH_SAMPLES):
        tQq = {tSide: tQ[tSide] + tU * (qReady[tSide] - tQ[tSide]) for tSide in SIDES}
        if linkGap(arms, tQq) < READY_MIN_GAP_M:
            print("READY 경로에서 두 팔이 가까움 (< 100 mm) -- 손으로 옮기세요. 이동 안 함", flush=True)
            return
    tReply = bridge.s.post(bridge.base_url + "/clear_estop", timeout=CLEAR_ESTOP_TIMEOUT_S).json()
    if not tReply.get("ok", True):
        print("clear_estop 실패:", tReply)
        return
    tG = {tSide: tObs.g(tSide) for tSide in SIDES}
    say("READY 로 천천히 복귀")
    ramp_both(bridge, tQ, qReady, tG, deg_s=READY_DEG_S)
    bridge.estop()
    say("READY 도착, 홀드")
