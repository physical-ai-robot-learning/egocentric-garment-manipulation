"""Robot kinematics of the deploy: robotize_ik arms (= umi_replay_bi settings) and FK of the TCP, the link-origin gap of the two
arms, and rotation helpers.

Inputs: joint vectors (6,) per side in bridge radians. Outputs: 4x4 TCP poses in the reBot base frame, distances (m), angles.
"""

from __future__ import annotations
from types import SimpleNamespace

import numpy as np

from .contract import SIDES

# robotize_ik.Arm settings: IK step bound (deg), orientation cost, table plane z, finger-corner floor, jaw half width, roll
# handling
ROBOTIZE_ARGS = dict(
    ik_step=7.5,
    ori_cost=0.1,
    table_z=0.003,
    tip_clear=0.005,
    tip_clear_left_extra=0.005,
    jaw_half=0.020,
    level_max=0.0,
    roll_thr=115.0,
    roll_relax=1.5,
)


def makeArms(**over):
    """laundry_dataset.Arm per side: Pink solver with the bridge joint limits (0.5 deg inside), joint/base/TCP overrides, orientation
    cost 0.1, 7.5 deg/step bound; cornerLift (finger-corner table floor); link-origin set for the 120 mm gap.
    -> ({side: Arm}, the shared argument namespace)."""
    from laundry_dataset import Arm

    tArgs = SimpleNamespace(**{**ROBOTIZE_ARGS, **over})
    return {tSide: Arm(tSide, tArgs) for tSide in SIDES}, tArgs


def fk(arms: dict, side: str, q) -> np.ndarray:
    return arms[side].mPk.fk(np.asarray(q, np.float64))


def linkGap(arms: dict, q: dict) -> float:
    """umi_replay_bi / robotize_ik link-origin distance (m): min over link3-6, gripper_link, fingers, tcp of both arms."""
    tLeft = arms["left"].links(q["left"])
    tRight = arms["right"].links(q["right"])
    return float(np.linalg.norm(tLeft[:, None] - tRight[None], axis=2).min())


def rotInterp(Ra: np.ndarray, Rb: np.ndarray, s: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return (Rotation.from_matrix(Ra) * Rotation.from_rotvec(s * Rotation.from_matrix(Ra.T @ Rb).as_rotvec())).as_matrix()


def rotAngleDeg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))
