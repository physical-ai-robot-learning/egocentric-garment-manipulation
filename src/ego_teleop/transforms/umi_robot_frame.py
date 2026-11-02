"""THE frame adapter between UMI human TCP poses (reBot DATASET-TCP axes) and the reBot IK/FK frame (2026-10-13).

Two TCP frames are in play. They share the same origin and differ by a FIXED rotation about TCP y:

    dataset-TCP axes (DS)  what every dataset `observation.ee.tcp_state` and every UMI/HandUMI human TCP stream uses.
                           UMI chain (ego_cart20/umi_fusion/run_smoother.py:119, sources/hfold_export.py:46-49, 167):
                               T_W_TCP = T_W_cam @ X_cam_tcp(camera_tcp_v2) @ X_TOOL,   X_TOOL = F @ C @ CT4 = Rx(180 deg)
                           camera_tcp_v2: TCP x = approach (= camera optical z), y = jaw closing (= camera -x), z = x cross y;
                           X_TOOL then flips y and z ("reBot dataset-TCP axes", hfold_export.py:182).
    FK/IK frame (FK)       pyroki rebot_b601 (handumi-sw), = PyrokiIkSolver.fk/ik in robot/rebot_client.py, = eef_kin.

        R_fk = R_ds @ Ry(+90 deg)          p_fk = p_ds            (T_fk = T_ds @ T_DS_TO_FK)

Provenance of that relation:
  * MEASURED 2026-09-22 on 31 recorded robot frames (joints -> eef_kin FK vs the same rows' tcp_state): position 0.00 mm,
    rotation 90.00 deg about [0, 1, 0], std 0.00 (memory rebot-tcp-frame-90deg). Constant in
    ~/softfold_infer/rebot/infer_core_v4.py:274  R_DS_TO_FK = Rot.from_rotvec([0, pi/2, 0]); used at :1045 (IK target
    DS -> FK: R @ R_DS_TO_FK) and :580 (FK -> DS: R @ R_FK_TO_DS). Missing it produced the 577 mm upward runaway.
  * The test tests/teleop/test_umi_robot_frame.py re-reads that line from infer_core_v4.py (when present) and pins it.

Relative teleop (spec):  T_target_fk = T_R0_fk @ map(inv(T_H0_ds) @ T_H_ds(t)).
  With T_R0_ds = T_R0_fk @ inv(T_DS_TO_FK):  T_target_fk = T_R0_fk @ [inv(Y) @ Delta_ds @ Y],  Y = T_DS_TO_FK.
  So the body-frame delta conjugates by Ry(90): R_fk = Ry(90)^T R_ds Ry(90), t_fk = Ry(90)^T t_ds = (-t_z, t_y, t_x),
  i.e. HumanRobotFrameMapper(translation_axis_map=[z, y, x], translation_sign=[-1, 1, 1]) (mapper_config_dataset_to_fk()).
  Robot-world effect: the robot TCP moves by R_R0_ds @ t_ds in its base frame, exactly as if the robot executed the
  human's body-frame delta in its own dataset-TCP frame.

UNVERIFIED (say so wherever this is used):
  U1  That the HandUMI camera_tcp_v2 @ Rx(180) axes are physically the SAME axes as the robot's dataset tcp_state axes is a
      declaration of the export pipeline (hfold_export / handumi_export), not a measurement on a robot. The camera TCP x is
      the OPTICAL axis, pitched against the jaw (camera_tcp_v2 says 20.7 deg; infer_core_v4 V4_UMI_TCP_PITCH_DEG suggests
      33.6-36 deg, default OFF). Any such pitch is a further fixed rotation about TCP y that this adapter does NOT apply.
  U2  The human TCP point (camera_tcp_v2 translation) and the robot FK TCP point (TIP_X = 0.07, ~82 mm ahead of the jaw tip
      per memory hra-shared-base-frame) are not shown to be the same physical point; relative rotations about different
      points give different translations.
  U3  Body-frame relative mapping preserves WORLD directions (human up -> robot up) only if the anchors agree in
      orientation up to a yaw about the vertical: R_W_H0 ~= Rz(psi) @ R_RB_R0_ds. With the robot starting at home and the
      human starting wherever they hold the gripper this is not guaranteed; tools/v0_replay.py reports the anchor tilt
      mismatch and the human-z vs robot-z correlation.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from .frames import FrameMapperConfig, HumanRobotFrameMapper
from .se3 import inv_T, make_T

R_DS_TO_FK = Rotation.from_rotvec([0.0, np.pi / 2, 0.0]).as_matrix()      # == infer_core_v4.py:274
R_FK_TO_DS = R_DS_TO_FK.T
T_DS_TO_FK = make_T(R_DS_TO_FK, np.zeros(3))
T_FK_TO_DS = make_T(R_FK_TO_DS, np.zeros(3))
X_TOOL_EXPECTED = make_T(Rotation.from_rotvec([np.pi, 0.0, 0.0]).as_matrix(), np.zeros(3))   # Rx(180): ego_cart20 hfold_export X_TOOL
TOOL_FRAME_NAME = "reBot dataset-TCP axes (camera_tcp_v2 @ X_TOOL=Rx(180))"


def _arr(T) -> np.ndarray: return np.asarray(T, np.float64)


def dataset_tcp_to_fk(T_ds) -> np.ndarray:
    """Pose(s) in dataset-TCP axes -> the same physical pose(s) in the FK/IK TCP frame. (...,4,4)"""
    return _arr(T_ds) @ T_DS_TO_FK


def fk_to_dataset_tcp(T_fk) -> np.ndarray:
    """FK/IK TCP pose(s) -> dataset-TCP axes (what exports / labels must use). (...,4,4)"""
    return _arr(T_fk) @ T_FK_TO_DS


def delta_dataset_to_fk(D_ds) -> np.ndarray:
    """Body-frame delta inv(T0_ds) @ T_ds -> the equal body-frame delta in the FK TCP frame: inv(Y) @ D @ Y."""
    return T_FK_TO_DS @ _arr(D_ds) @ T_DS_TO_FK


def delta_fk_to_dataset(D_fk) -> np.ndarray:
    return T_DS_TO_FK @ _arr(D_fk) @ T_FK_TO_DS


def mapper_config_dataset_to_fk(translation_scale: float = 1.0, rotation_scale: float = 1.0) -> FrameMapperConfig:
    """The signed-permutation form of delta_dataset_to_fk for HumanRobotFrameMapper (equality pinned by a test)."""
    s = float(translation_scale)
    return FrameMapperConfig(translation_axis_map=("z", "y", "x"), translation_sign=(-1.0, 1.0, 1.0),
                             translation_scale=(s, s, s), rotation_scale=float(rotation_scale))


class UmiRelativeToRobot:
    """Relative teleop from a UMI human TCP stream (dataset-TCP axes, any world) to an FK-frame robot TCP target:

        T_target_fk(t) = T_R0_fk @ scale_t( delta_dataset_to_fk( inv(T_H0_ds) @ T_H_ds(t) ) )

    Translation scale is isotropic (so it commutes with the axis permutation); rotation is never scaled here."""

    def __init__(self, translation_scale: float = 1.0) -> None:
        if translation_scale < 0: raise ValueError("translation_scale must be >= 0")
        self.s = float(translation_scale); self.T_H0: np.ndarray | None = None; self.T_R0_fk: np.ndarray | None = None

    def anchor(self, T_H0_ds, T_R0_fk) -> None:
        self.T_H0 = _arr(T_H0_ds).copy(); self.T_R0_fk = _arr(T_R0_fk).copy()

    def delta_fk(self, T_H_ds) -> np.ndarray:
        if self.T_H0 is None: raise RuntimeError("anchor() first")
        D = delta_dataset_to_fk(inv_T(self.T_H0) @ _arr(T_H_ds)); D = D.copy(); D[:3, 3] *= self.s
        return D

    def target_fk(self, T_H_ds) -> np.ndarray:
        return self.T_R0_fk @ self.delta_fk(T_H_ds)


def anchor_tilt_mismatch_deg(R_W_H0_ds, R_RB_R0_ds) -> float:
    """U3 diagnostic: the smallest rotation between the two anchor orientations after removing the best yaw about the
    vertical (both frames have z = up). 0 => human world directions map 1:1 (up to yaw) onto robot base directions."""
    Ra, Rb = _arr(R_W_H0_ds)[:3, :3], _arr(R_RB_R0_ds)[:3, :3]
    best = 180.0
    for psi in np.radians(np.arange(0.0, 360.0, 0.5)):
        Rz = Rotation.from_rotvec([0, 0, psi]).as_matrix()
        best = min(best, float(np.degrees(Rotation.from_matrix(Rz @ Rb @ Ra.T).magnitude())))
    return best


__all__ = ["R_DS_TO_FK", "R_FK_TO_DS", "T_DS_TO_FK", "T_FK_TO_DS", "X_TOOL_EXPECTED", "TOOL_FRAME_NAME", "dataset_tcp_to_fk",
           "fk_to_dataset_tcp", "delta_dataset_to_fk", "delta_fk_to_dataset", "mapper_config_dataset_to_fk", "UmiRelativeToRobot",
           "anchor_tilt_mismatch_deg", "HumanRobotFrameMapper"]
