"""[2026-10-21] LEFT GRIPPER ONLY re-zero. The left jaw read -32.6 deg while closed (right: 0.0) and opened too far on command.
Torque is never enabled. Without --go: read only. With --go: disable the gripper motor, set_zero_position on it only, read back.
Close the left jaws FULLY by hand first (motor is unpowered in this read-only session). Arm joints untouched.
The MIT bridge must not hold this port (REST + stop the bridge first, by PID)."""
import math, sys, time
from lerobot_plugin_rebot.config import ReBotConfig
from lerobot_plugin_rebot.arm_backend.motorbridge_backend import ReBotArmBackend
port, go = sys.argv[1], "--go" in sys.argv
b = ReBotArmBackend(ReBotConfig(id="left_grip_zero", port=port, can_adapter="damiao_serial", cameras={})); b.open_readonly(); T = b._transport
def show(tag):
    g = math.degrees(T.read_gripper_rad()); q = [math.degrees(v) for v in T.read_joints_rad()]
    print(f"{tag}: 집게 {g:7.1f}° | 관절 " + " ".join(f"j{k + 1}={v:6.1f}" for k, v in enumerate(q)), flush=True); return g
try:
    g0 = show("전")
    if abs(g0) < 2: print("집게가 이미 0 근처 -- 영점 안 씀"); sys.exit(0)
    if not go: print("읽기만 함. 집게를 손으로 끝까지 닫은 상태에서 --go"); sys.exit(0)
    with T._bus("zero_gripper_only"):
        m = T._motors["gripper"]; m.disable(); time.sleep(0.5); m.set_zero_position(); time.sleep(1.0)
    time.sleep(0.5); g1 = show("후")
    print("결과:", "집게 영점 OK" if abs(g1) < 2 else f"집게 = {g1:.1f} -- 영점이 안 써졌을 수 있음 (전원 껐다 켜고 다시 읽기)")
finally:
    b.disconnect(); print("연결 해제 (토크는 한 번도 켜지 않음)")
