"""[2026-10-21] read-only: print the LEFT gripper motor angle (raw, not clipped) every 0.5 s for N s while the user moves the jaw by hand.
Torque never enabled. Usage: python read_left_gripper.py /dev/cu.usbmodem... [seconds]"""
import math, sys, time
from lerobot_plugin_rebot.config import ReBotConfig
from lerobot_plugin_rebot.arm_backend.motorbridge_backend import ReBotArmBackend
port = sys.argv[1]; dur = float(sys.argv[2]) if len(sys.argv) > 2 else 15
b = ReBotArmBackend(ReBotConfig(id="left_grip_read", port=port, can_adapter="damiao_serial", cameras={})); b.open_readonly(); T = b._transport
try:
    print(f"플러그인 설정: 열림 {math.degrees(b._open_rad):.1f}° / 닫힘 {math.degrees(b._close_rad):.1f}°", flush=True)
    t0 = time.time()
    while time.time() - t0 < dur:
        print(f"  {time.time() - t0:4.1f}s 집게 {math.degrees(T.read_gripper_rad()):8.1f}°", flush=True); time.sleep(0.5)
finally:
    b.disconnect(); print("연결 해제 (토크는 한 번도 켜지 않음)")
