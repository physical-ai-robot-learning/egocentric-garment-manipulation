"""[2026-10-21] LEFT arm j6 ONLY re-zero (was zero_left_j6_only.py): the encoder zero was half a turn off (reads ~178.5 deg at the
physical zero; user confirmed the wrist matches the right arm). Torque is never enabled. Without --go: read only. With --go:
disable joint_5, set_zero_position on joint_5 only, read back. j1-j5 + gripper untouched (the plugin's set_zero would re-zero
all 7 at the present, slightly-off pose). Deploy stage: robot bridge tools.
    python -m laundry_deploy.robot_service.zero_left_j6 <port> [--go]
"""

from __future__ import annotations
import math
import sys
import time

J6_HALF_TURN_RANGE_DEG = (150.0, 210.0)  # |j6| must read near +-180 (half a turn off) before re-zeroing
REST_TOL_DEG = 15.0  # j1-j5 must be near 0 (rest pose)
ZERO_OK_DEG = 3.0  # |j6| after the re-zero that counts as written
DISABLE_SETTLE_S = 0.5
ZERO_SETTLE_S = 1.0
READ_SETTLE_S = 0.5
J6_INDEX = 5


def main(argv: list[str] | None = None) -> int:
    from lerobot_plugin_rebot.arm_backend.motorbridge_backend import ReBotArmBackend
    from lerobot_plugin_rebot.config import ReBotConfig

    tArgs = sys.argv[1:] if argv is None else argv
    tPort, tGo = tArgs[0], "--go" in tArgs
    tBackend = ReBotArmBackend(ReBotConfig(id="left_j6_zero", port=tPort, can_adapter="damiao_serial", cameras={}))
    tBackend.open_readonly()
    tTransport = tBackend._transport

    def show(tag):
        tQ = [math.degrees(tV) for tV in tTransport.read_joints_rad()]
        print(f"{tag}: " + " ".join(f"j{tK + 1}={tV:7.1f}" for tK, tV in enumerate(tQ)), flush=True)
        return tQ

    try:
        tQ0 = show("전")
        if not (
            J6_HALF_TURN_RANGE_DEG[0] < abs(tQ0[J6_INDEX]) < J6_HALF_TURN_RANGE_DEG[1]
            and all(abs(tV) < REST_TOL_DEG for tV in tQ0[:J6_INDEX])
        ):
            sys.exit("중단: j6 가 ±180 근처가 아니거나 j1-j5 가 쉬는 자세(0 근처)가 아님 -- 영점 안 씀")
        if not tGo:
            print("읽기만 함. 영점을 쓰려면 --go")
            sys.exit(0)
        with tTransport._bus("zero_j6_only"):
            tMotor = tTransport._motors["joint_5"]
            tMotor.disable()
            time.sleep(DISABLE_SETTLE_S)
            tMotor.set_zero_position()
            time.sleep(ZERO_SETTLE_S)
        time.sleep(READ_SETTLE_S)
        tQ1 = show("후")
        print(
            "결과:",
            "j6 영점 OK"
            if abs(tQ1[J6_INDEX]) < ZERO_OK_DEG
            else f"j6 = {tQ1[J6_INDEX]:.1f} -- 영점이 안 써졌을 수 있음 (전원 껐다 켜고 다시 읽기)",
        )
    finally:
        tBackend.disconnect()
        print("연결 해제 (토크는 한 번도 켜지 않음)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
