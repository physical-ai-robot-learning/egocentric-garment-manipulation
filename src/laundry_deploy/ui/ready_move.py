"""The checked READY move of the UI's [READY로 이동] button: visual_teleop ready_pose's fixed READY_Q_DEG with ready_pose's checks
(TCP >= 0.10 m along the joint line, bridge joint limits) and its smoothstep joint ramp at 15 deg/s through HttpBridge.send; both
arms, one at a time (right, then left), each path also checked against the other arm (capsule clearance). /clear_estop first (the
user confirmed in the UI).
"""

from __future__ import annotations
import time

import numpy as np

from ..core.contract import SIDES
from ..core.errors import HardwareError, RunRefused
from ..core.kinematics import fk

PATH_SAMPLES = 41  # TCP height samples along the joint line
MIN_TCP_Z_M = 0.10
MIN_MOVE_DEG = 2  # a move this small is allowed even below MIN_TCP_Z_M
COLLISION_SAMPLES = 21
UNPARK_TIMEOUT_S = 15.0  # 11-01: a PARKED arm is re-enabled by its first write (~3 s)
CLEAR_ESTOP_TIMEOUT_S = 2.0
RAMP_DEG_S = 15.0
STREAM_HZ = 30.0
ESTOP_CHECK_EVERY = 30  # ticks between /observe E-STOP checks during the ramp
ERROR_EXCERPT = 80
MOVE_ORDER = ("right", "left")


def moveReady(bridge, rig) -> None:
    from laundry_shared.visual_teleop.tools.robot_moves import unpark

    tStatus = bridge.status()
    if tStatus.get("estop"):
        bridge.s.post(bridge.base_url + "/clear_estop", timeout=CLEAR_ESTOP_TIMEOUT_S)
    # 11-01: a PARKED arm is re-enabled by the first /execute_step (~3 s) -> the stream's 0.5 s timeout E-STOPPED the READY move.
    # Unpark first with a long timeout; a refused write (Damiao latch: reads ok, writes ignored) stops here with the reason.
    for tSide in bridge.status().get("parked") or []:
        tReply = bridge.s.post(bridge.base_url + "/torque", json={"arm": tSide, "on": True}, timeout=UNPARK_TIMEOUT_S).json()
        tMsg = str((tReply.get("arms") or {}).get(tSide, ""))
        if tMsg.startswith("ERR"):
            raise HardwareError(f"{tSide} 팔 토크 켜기 실패 ({tMsg[:ERROR_EXCERPT]}) -- 모터 쓰기 무응답: 그 팔 전원을 껐다 켜기")
    for tSide in MOVE_ORDER:
        tObs = bridge.observe()
        if tObs.estop:
            raise HardwareError("E-STOP/읽기 지연")
        tQ0 = tObs.q[tSide].copy()
        tQr = rig.mQReady[tSide]
        tOther = "left" if tSide == "right" else "right"
        tZMin = min(fk(rig.mArms, tSide, tQ0 + tU * (tQr - tQ0))[2, 3] for tU in np.linspace(0, 1, PATH_SAMPLES))
        if tZMin < MIN_TCP_Z_M and np.degrees(np.abs(tQr - tQ0)).max() > MIN_MOVE_DEG:
            raise RunRefused(f"{tSide}: 경로 중 TCP 최저 {tZMin:.3f} m < 0.10 -- 이동 안 함")
        if np.any(tQr < rig.mArms[tSide].mLo) or np.any(tQr > rig.mArms[tSide].mHi):
            raise RunRefused("READY 가 관절 한계 밖")
        for tU in np.linspace(0, 1, COLLISION_SAMPLES):
            tCheck = rig.mCollision.check({tSide: tQ0 + tU * (tQr - tQ0), tOther: tObs.q[tOther]})
            if not tCheck["ok_arm"]:
                raise RunRefused(f"{tSide}: READY 경로에서 두 팔 충돌 위험 ({tCheck['pair']}) -- 이동 안 함")
        unpark(bridge, tSide)
        tObs = bridge.observe()
        tQ0 = tObs.q[tSide].copy()
        tD = tQr - tQ0
        tG = {tX: tObs.g(tX) for tX in SIDES}
        tN = max(2, int(np.ceil(np.degrees(np.abs(tD)).max() / RAMP_DEG_S * STREAM_HZ)))
        tT0 = time.monotonic()
        for tK in range(1, tN + 1):
            tU = tK / tN
            tU = tU * tU * (3 - 2 * tU)
            tReply = bridge.send({tSide: tQ0 + tU * tD, tOther: tObs.q[tOther]}, tG)
            if not tReply.accepted:
                bridge.estop()
                raise HardwareError(f"브리지 거부: {tReply.error} -- E-STOP")
            if tK % ESTOP_CHECK_EVERY == 0 and bridge.observe().estop:
                bridge.estop()
                raise HardwareError("이동 중 E-STOP/읽기 지연")
            time.sleep(max(0.0, tT0 + tK / STREAM_HZ - time.monotonic()))
