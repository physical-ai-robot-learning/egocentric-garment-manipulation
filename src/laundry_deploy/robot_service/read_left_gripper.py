"""[2026-10-21] Read-only: print the LEFT gripper motor angle (raw, not clipped) every 0.5 s for N s while the user moves the jaw
by hand. Torque never enabled. Deploy stage: robot bridge tools.
    python -m laundry_deploy.robot_service.read_left_gripper /dev/cu.usbmodem... [seconds]
"""

from __future__ import annotations
import math
import sys
import time

DEFAULT_DURATION_S = 15.0
PRINT_PERIOD_S = 0.5


def main(argv: list[str] | None = None) -> int:
    from lerobot_plugin_rebot.arm_backend.motorbridge_backend import ReBotArmBackend
    from lerobot_plugin_rebot.config import ReBotConfig

    tArgs = sys.argv[1:] if argv is None else argv
    tPort = tArgs[0]
    tDurationS = float(tArgs[1]) if len(tArgs) > 1 else DEFAULT_DURATION_S
    tBackend = ReBotArmBackend(ReBotConfig(id="left_grip_read", port=tPort, can_adapter="damiao_serial", cameras={}))
    tBackend.open_readonly()
    tTransport = tBackend._transport
    try:
        print(
            f"플러그인 설정: 열림 {math.degrees(tBackend._open_rad):.1f}° / 닫힘 {math.degrees(tBackend._close_rad):.1f}°",
            flush=True,
        )
        tT0 = time.time()
        while time.time() - tT0 < tDurationS:
            print(f"  {time.time() - tT0:4.1f}s 집게 {math.degrees(tTransport.read_gripper_rad()):8.1f}°", flush=True)
            time.sleep(PRINT_PERIOD_S)
    finally:
        tBackend.disconnect()
        print("연결 해제 (토크는 한 번도 켜지 않음)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
