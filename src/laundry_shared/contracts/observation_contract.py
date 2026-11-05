"""Canonical observation contract (shared by Agent A and Agent B; the ONLY definition of it).

Observation (what the robot side produces every control tick):
    images            {"global", "left_wrist", "right_wrist"} -> HxWx3 uint8 RGB.  "global" is the fixed scene camera
                      (also the orchestrator's primary view).
    T_W_L, T_W_R      4x4 TCP poses of both arms in ONE common robot frame W (reBot base).  TCP axes = the dataset
                      TCP convention TCP_CONVENTION below; the reBot FK frame differs from it by a fixed rotation and
                      the robot adapter must convert (see action_observation_contract.md).
    gripper_L/R       measured aperture, 0 = closed, 1 = open
    tracking_confidence  0..1 (1 for robot FK; < 1 when poses come from a tracker)

Policy proprioception vector (PROPRIO_DIM = 30), built only by proprio_vector():
    [ L pose9 | R pose9 | g_L | g_R | T_LR pose9 | T_LR_valid ]
      0-8       9-17     18    19     20-28          29
    L/R pose9   per-arm EPISODE-ANCHOR pose inv(T_W_X(t0)) @ T_W_X(t), t0 = episode start (embodiment-normalised;
                same semantics as the ego_cart20 / RELCART20 task-anchor state)
    T_LR        inv(T_W_L) @ T_W_R (xyz + rot6d).  Requires both arms in one frame.  Sources whose L-R extrinsic is
                unknown (e.g. X-VLA Soft-Fold: separate per-arm base frames) set T_LR = identity, T_LR_valid = 0.
"""
from dataclasses import dataclass, field

import numpy as np

from .action_contract import T_to_pose9, inv_T

CAMERA_KEYS = ("global", "left_wrist", "right_wrist")
PROPRIO_DIM = 30
P_L_POSE, P_R_POSE = slice(0, 9), slice(9, 18)
P_GRIP_L, P_GRIP_R = 18, 19
P_T_LR, P_T_LR_VALID = slice(20, 29), 29
PROPRIO_NAMES = ([f"left_anchor_{n}" for n in ("x", "y", "z", *[f"r{i}" for i in range(6)])]
                 + [f"right_anchor_{n}" for n in ("x", "y", "z", *[f"r{i}" for i in range(6)])]
                 + ["gripper_left", "gripper_right"]
                 + [f"T_LR_{n}" for n in ("x", "y", "z", *[f"r{i}" for i in range(6)])] + ["T_LR_valid"])

TCP_CONVENTION = "dataset-TCP axes: x = approach, y = wrist-camera image x, z = wrist-camera image y"

DATA_SOURCES = ("XVLA_CLOTH", "UMI", "VISUAL_TELEOP", "ROBOT_AUTONOMOUS")


@dataclass
class Observation:
    timestamp_s: float
    images: dict
    T_W_L: np.ndarray
    T_W_R: np.ndarray
    gripper_L: float
    gripper_R: float
    tracking_confidence: float = 1.0
    T_LR_valid: bool = True
    meta: dict = field(default_factory=dict)


@dataclass
class EpisodeAnchor:
    """TCP poses at episode start; fixed for the whole episode (skill switches do NOT reset it)."""
    T_W_L0: np.ndarray
    T_W_R0: np.ndarray

    @classmethod
    def from_observation(cls, obs):
        return cls(np.array(obs.T_W_L, np.float64), np.array(obs.T_W_R, np.float64))


def inter_gripper_pose(T_W_L, T_W_R):
    return inv_T(T_W_L) @ np.asarray(T_W_R, np.float64)


def proprio_vector(obs, anchor):
    v = np.zeros(PROPRIO_DIM)
    v[P_L_POSE] = T_to_pose9(inv_T(anchor.T_W_L0) @ obs.T_W_L)
    v[P_R_POSE] = T_to_pose9(inv_T(anchor.T_W_R0) @ obs.T_W_R)
    v[P_GRIP_L], v[P_GRIP_R] = obs.gripper_L, obs.gripper_R
    if obs.T_LR_valid:
        v[P_T_LR] = T_to_pose9(inter_gripper_pose(obs.T_W_L, obs.T_W_R)); v[P_T_LR_VALID] = 1.0
    else:
        v[P_T_LR] = T_to_pose9(np.eye(4)); v[P_T_LR_VALID] = 0.0
    return v.astype(np.float32)


def _is_rigid(T):
    T = np.asarray(T, np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        return False
    R = T[:3, :3]
    return np.allclose(R @ R.T, np.eye(3), atol=1e-4) and abs(np.linalg.det(R) - 1) < 1e-4 and np.allclose(T[3], [0, 0, 0, 1])


def validate_observation(obs, required_cameras=CAMERA_KEYS):
    """-> (ok, reason).  Never raises."""
    for k in required_cameras:
        im = obs.images.get(k)
        if im is None:
            return False, f"missing_camera:{k}"
        im = np.asarray(im)
        if im.ndim != 3 or im.shape[2] != 3 or im.dtype != np.uint8:
            return False, f"bad_image:{k}:{im.shape}:{im.dtype}"
    for name in ("T_W_L", "T_W_R"):
        if not _is_rigid(getattr(obs, name)):
            return False, f"bad_pose:{name}"
    for name in ("gripper_L", "gripper_R"):
        g = getattr(obs, name)
        if not np.isfinite(g) or g < -0.05 or g > 1.05:
            return False, f"bad_gripper:{name}:{g}"
    if not 0.0 <= obs.tracking_confidence <= 1.0:
        return False, f"bad_tracking_confidence:{obs.tracking_confidence}"
    return True, ""
