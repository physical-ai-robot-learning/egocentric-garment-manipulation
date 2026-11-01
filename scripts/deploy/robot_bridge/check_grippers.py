"""[2026-10-21] read-only jaw angle per arm port before the MIT bridge opens them (the left jaw zero moved -268 / -100 deg across power
cycles and "close" then drove it open). Prints one line per port; exit 0 always. Usage: python check_grippers.py <port> [<port> ...]"""
import math, sys
from lerobot_plugin_rebot.config import ReBotConfig
from lerobot_plugin_rebot.arm_backend.motorbridge_backend import ReBotArmBackend
for port in sys.argv[1:]:
    b = ReBotArmBackend(ReBotConfig(id="grip_check", port=port, can_adapter="damiao_serial", cameras={}))
    try:
        b.open_readonly(); g = math.degrees(b._transport.read_gripper_rad())
        print(f"GRIP {port} {g:.1f}", flush=True)
    except Exception as e:
        print(f"GRIP {port} ERR {str(e)[:120]}", flush=True)
    finally:
        try: b.disconnect()
        except Exception: pass
