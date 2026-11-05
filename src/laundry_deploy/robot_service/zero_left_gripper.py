"""[2026-10-21] LEFT GRIPPER ONLY re-zero (was zero_left_gripper_only.py). The left jaw read -32.6 deg while closed (right: 0.0)
and opened too far on command. Torque is never enabled. Without --go: read only. With --go: disable the gripper motor,
set_zero_position on it only, read back. Close the left jaws FULLY by hand first (the motor is unpowered in this read-only
session). Arm joints untouched. The MIT bridge must not hold this port (REST + stop the bridge first, by PID).
Deploy stage: robot bridge tools.
    python -m laundry_deploy.robot_service.zero_left_gripper <port> [--go]
"""

from __future__ import annotations
import math
import sys
import time

ZERO_OK_DEG = 2.0  # |jaw| below this counts as already / successfully zeroed
DISABLE_SETTLE_S = 0.5
ZERO_SETTLE_S = 1.0
READ_SETTLE_S = 0.5


def main(argv: list[str] | None = None) -> int:
    from lerobot_plugin_rebot.arm_backend.motorbridge_backend import ReBotArmBackend
    from lerobot_plugin_rebot.config import ReBotConfig

    tArgs = sys.argv[1:] if argv is None else argv
    tPort, tGo = tArgs[0], "--go" in tArgs
    tBackend = ReBotArmBackend(ReBotConfig(id="left_grip_zero", port=tPort, can_adapter="damiao_serial", cameras={}))
    tBackend.open_readonly()
    tTransport = tBackend._transport

    def show(tag):
        tG = math.degrees(tTransport.read_gripper_rad())
        tQ = [math.degrees(tV) for tV in tTransport.read_joints_rad()]
        print(f"{tag}: 집게 {tG:7.1f}° | 관절 " + " ".join(f"j{tK + 1}={tV:6.1f}" for tK, tV in enumerate(tQ)), flush=True)
        return tG

    try:
        tG0 = show("전")
        if abs(tG0) < ZERO_OK_DEG:
            print("집게가 이미 0 근처 -- 영점 안 씀")
            sys.exit(0)
        if not tGo:
            print("읽기만 함. 집게를 손으로 끝까지 닫은 상태에서 --go")
            sys.exit(0)
        with tTransport._bus("zero_gripper_only"):
            tMotor = tTransport._motors["gripper"]
            tMotor.disable()
            time.sleep(DISABLE_SETTLE_S)
            tMotor.set_zero_position()
            time.sleep(ZERO_SETTLE_S)
        time.sleep(READ_SETTLE_S)
        tG1 = show("후")
        print(
            "결과:",
            "집게 영점 OK"
            if abs(tG1) < ZERO_OK_DEG
            else f"집게 = {tG1:.1f} -- 영점이 안 써졌을 수 있음 (전원 껐다 켜고 다시 읽기)",
        )
    finally:
        tBackend.disconnect()
        print("연결 해제 (토크는 한 번도 켜지 않음)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
