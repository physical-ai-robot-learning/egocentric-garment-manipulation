"""[2026-10-21] LEFT arm j6 ONLY re-zero (encoder zero half a turn off: reads ~178.5 deg at the physical zero, user confirmed the wrist
matches the right arm). Torque is never enabled. Without --go: read only. With --go: disable joint_5, set_zero_position on joint_5 only,
read back. j1-j5 + gripper untouched (the plugin's set_zero would re-zero all 7 at the present, slightly-off pose)."""
import math, sys, time
from lerobot_plugin_rebot.config import ReBotConfig
from lerobot_plugin_rebot.arm_backend.motorbridge_backend import ReBotArmBackend
port, go = sys.argv[1], "--go" in sys.argv
b = ReBotArmBackend(ReBotConfig(id="left_j6_zero", port=port, can_adapter="damiao_serial", cameras={})); b.open_readonly(); T = b._transport
def show(tag):
    q = [math.degrees(v) for v in T.read_joints_rad()]; print(f"{tag}: " + " ".join(f"j{k + 1}={v:7.1f}" for k, v in enumerate(q)), flush=True); return q
try:
    q0 = show("전")
    if not (150 < abs(q0[5]) < 210 and all(abs(v) < 15 for v in q0[:5])):
        sys.exit(f"중단: j6 가 ±180 근처가 아니거나 j1-j5 가 쉬는 자세(0 근처)가 아님 -- 영점 안 씀")
    if not go: print("읽기만 함. 영점을 쓰려면 --go"); sys.exit(0)
    with T._bus("zero_j6_only"):
        m = T._motors["joint_5"]; m.disable(); time.sleep(0.5); m.set_zero_position(); time.sleep(1.0)
    time.sleep(0.5); q1 = show("후")
    print("결과:", "j6 영점 OK" if abs(q1[5]) < 3 else f"j6 = {q1[5]:.1f} -- 영점이 안 써졌을 수 있음 (전원 껐다 켜고 다시 읽기)")
finally:
    b.disconnect(); print("연결 해제 (토크는 한 번도 켜지 않음)")
