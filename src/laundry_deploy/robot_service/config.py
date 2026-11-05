"""Environment switches and fixed frame constants of the MIT robot bridge (`mit_bridge`).

The switches are read once at import (documented launch environment, see docs/laundry/deploy/robot_bridge.md); the ones
`/set_speed` changes at runtime live in the mutable `LIVE` settings, and the per-side jaw close extra in the mutable
`GRIP_CLOSE_EXTRA` dict (`/jaw_close_test`). Deploy stage: robot bridge (runs on the robot Mac, colleague's env).
"""

from __future__ import annotations
import os

import numpy as np

MOCK = os.environ.get("MIT_MOCK", "0") == "1"
PORT = int(os.environ.get("MIT_PORT", "8021"))
TICK_HZ = float(os.environ.get("MIT_TICK_HZ", "100"))
REST_DEG_S = float(os.environ.get("MIT_REST_DEG_S", "15"))
# 10-02 goal interpolation: a goal within this gap of the previous one becomes a segment (see mit_bridge docstring)
INTERP_MAX_GAP_S = float(os.environ.get("MIT_INTERP_MAX_GAP_S", "1.5"))
CAMERAS = {
    "global": int(os.environ.get("CAM_GLOBAL", 0)),
    "left_wrist": int(os.environ.get("CAM_LEFT", 1)),
    "right_wrist": int(os.environ.get("CAM_RIGHT", 2)),
}
CAM_MAP = {"global": "middle", "left_wrist": "left", "right_wrist": "right"}
CAM_WIDTH_PX = 640
CAM_HEIGHT_PX = 480
CAM_FPS = 30
CAM_WARMUP_S = 4.0
MOTORS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_yaw", "wrist_roll"]
ARM_DOF = 6
SEEED_DIR = np.array([-1.0, -1.0, 1.0, 1.0, 1.0, -1.0])  # Seeed DM follower joint_directions (leader -> raw)
SEEED_LIM_DEG = np.array([(-145, 145), (-170, 0), (-200, 0), (-80, 90), (-90, 90), (-130, 130)], float)  # raw-frame clip
GRIP_OPEN_DEG = -270.0
GRIP_CLOSE_DEG = 0.0
GRIP_CMD_SCALE = -6.0  # raw jaw deg per robot_service jaw command unit (0..45)
# 10-21 user: the left jaw opens too far -> per-side cap of "fully open" as a fraction of the -270 deg range (1.0 = unchanged)
GRIP_OPEN_FRAC = {s: float(os.environ.get(f"MIT_GRIP_OPEN_FRAC_{s.upper()}", "1.0")) for s in ("left", "right")}
# 11-02: per-side jaw close endpoint past 0 deg; mutated live by /jaw_close_test
GRIP_CLOSE_EXTRA = {s: float(os.environ.get(f"MIT_GRIP_CLOSE_DEG_{s.upper()}", "0")) for s in ("left", "right")}
GRIP_SQUEEZE_RAMP_DEG = 30.0  # 11-02: the extra close ramps in over the last 30 deg before closed only
ARM_IDX = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]  # joint (not jaw) entries of the 14-vector
JOINTS_PER_SIDE = 7  # 6 joints + jaw in the 14-vector
N_JOINTS = 14
ALL_SIDES = ("left", "right")
# 10-13: MIT_SIDES=left (or right) connects ONE arm; the absent arm reads as zeros. MIT_NO_CAMERAS=1 opens no camera.
SIDES = tuple(x for x in ALL_SIDES if x in os.environ.get("MIT_SIDES", "left,right").split(","))
NO_CAMERAS = os.environ.get("MIT_NO_CAMERAS", "0") == "1"
# 10-21 user "왼팔이 끊기면 다시 키는 옵션": a "MOTOR FAULT ... disabled" report re-enables that arm in place
MOTOR_REENABLE = os.environ.get("MIT_MOTOR_REENABLE", "1") == "1"
USB_RECONNECT = os.environ.get("MIT_USB_RECONNECT", "1") == "1"  # 10-21: auto re-open an arm whose USB serial vanished
PLAN_MAX_JUMP_DEG = float(os.environ.get("MIT_PLAN_MAX_JUMP_DEG", "90"))


class LiveSettings:
    """Motion settings `/set_speed` changes while the bridge runs (initial values from the launch environment)."""

    def __init__(self) -> None:
        self.mMaxDegS = float(os.environ.get("MIT_MAX_DEG_S", "45"))
        self.mMaxAccDegS2 = float(os.environ.get("MIT_MAX_ACC_DEG_S2", "180"))  # 10-01 colleague's POLICY_MAX_JOINT_ACC_DEG_S2
        self.mGoalInterp = os.environ.get("MIT_GOAL_INTERP", "0") == "1"
        self.mInterpKp = float(os.environ.get("MIT_INTERP_KP", "8.0"))  # 1/s
        # 10-02 user "rest 자세로 다 온 후에는 torque를 꺼줘": park (torque off) once /rest has arrived
        self.mRestTorqueOff = os.environ.get("MIT_REST_TORQUE_OFF", "0") == "1"
        self.mGripDegS = float(os.environ.get("MIT_GRIP_DEG_S", "90"))  # 10-01 user "그리퍼가 너무 빨라" (colleague default 900)


LIVE = LiveSettings()
