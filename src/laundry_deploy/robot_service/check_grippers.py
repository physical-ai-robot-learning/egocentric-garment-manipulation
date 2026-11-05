"""[2026-10-21] Read-only jaw angle per arm port before the MIT bridge opens them (the left jaw zero moved -268 / -100 deg across
power cycles and "close" then drove it open). Prints one line per port; exit 0 always. Deploy stage: robot bridge tools.
    python -m laundry_deploy.robot_service.check_grippers <port> [<port> ...]
"""

from __future__ import annotations
import math
import sys

ERROR_CHARS = 120


def main(argv: list[str] | None = None) -> int:
    from lerobot_plugin_rebot.arm_backend.motorbridge_backend import ReBotArmBackend
    from lerobot_plugin_rebot.config import ReBotConfig

    tPorts = sys.argv[1:] if argv is None else argv
    for tPort in tPorts:
        tBackend = ReBotArmBackend(ReBotConfig(id="grip_check", port=tPort, can_adapter="damiao_serial", cameras={}))
        try:
            tBackend.open_readonly()
            tGripDeg = math.degrees(tBackend._transport.read_gripper_rad())
            print(f"GRIP {tPort} {tGripDeg:.1f}", flush=True)
        except Exception as tErr:
            print(f"GRIP {tPort} ERR {str(tErr)[:ERROR_CHARS]}", flush=True)
        finally:
            try:
                tBackend.disconnect()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
