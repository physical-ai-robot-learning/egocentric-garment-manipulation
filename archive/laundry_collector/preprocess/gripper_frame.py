"""Human pinch -> virtual parallel-jaw gripper pose, the ONE source for both the rendered gripper and the action labels.

Canonical gripper frame (identical for the left and the right hand; robot TCP should be defined the same way):
  origin  TCP = midpoint of thumb tip and index tip
  +z      palm normal, pointing OUT of the palm (towards what the hand grasps)
  +y      forward = from the wrist towards the fingers (approach direction, projected orthogonal to +z)
  +x      = y x z  -> the jaw axis; for each hand it is thumb->index or index->thumb, whichever makes (x, y, z) right-handed
The jaw axis is EXACTLY the thumb-index direction (jaw tips on the fingertips); +z is the palm normal made orthogonal to it
(Gram-Schmidt), +y = z x x. Left / right are NOT mirrored images of each
other's code path: the palm-normal sign is chosen per hand so +z points out of the palm for both (`palm_normal_sign`).

Degenerate frames (low confidence, thumb-index too short for an axis, collapsed palm) -> valid=False with a reason;
the pose then comes from the smoother (hold / interpolation), never NaN.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hand_pose import INDEX_MCP, INDEX_TIP, PINKY_MCP, THUMB_TIP, WRIST


@dataclass
class GripperConfig:
    human_pinch_min_m: float = 0.015
    human_pinch_max_m: float = 0.100
    robot_jaw_min_m: float = 0.000
    robot_jaw_max_m: float = 0.080
    min_axis_m: float = 0.010             # thumb-index shorter than this: jaw axis undefined -> keep the previous rotation
    min_palm_m2: float = 1e-4             # |v1 x v2| below this: palm plane undefined
    min_confidence: float = 0.5
    # which sign of (index_mcp - wrist) x (pinky_mcp - wrist) points OUT of the palm, per hand (MediaPipe / MANO landmark order)
    palm_normal_sign: dict = field(default_factory=lambda: {"right": 1.0, "left": -1.0})   # right hand, palm to camera, thumb on the viewer's right: v1 x v2 points out of the palm
    alpha_position: float = 0.3
    alpha_jaw: float = 0.3
    alpha_rotation: float = 0.2
    zero_phase: bool = True               # offline: forward + backward pass (no lag)
    # palm_forward (default): z = palm normal, y = wrist->fingers (orthogonalised), x = y x z -- stable when the two fingertips
    #   are close or aligned with the fingers (bring-up 2026-10-08: thumb_index flipped the gripper 180 deg mid-take).
    # thumb_index (spec section 5 order): x = thumb->index exactly, z = palm normal orthogonalised, y = z x x -- jaw tips on the
    #   fingertips, but the frame is only as good as the thumb-index direction.
    orientation_source: str = "palm_forward"
    # pose_mode camera_fixed (default, 2026-10-08 user): the virtual gripper is rigid to the wrist camera like the robot / HandUMI
    #   gripper -- only its two finger ends are in view, at the same place every frame; the human pinch drives the opening only.
    #   T_camera_gripper = mount (below); the action labels use the same mount, so render and labels still share one pose.
    # pose_mode hand: the gripper follows the 3D hand (thumb/index TCP + palm frame) -- diagnostic.
    pose_mode: str = "camera_fixed"
    # view robot_c922 (default with camera_fixed): output in the measured reBot wrist C922 view with the REAL robot tips composited
    #   (robot_view.py); the label pose is then the robot TCP from the hand-eye calibration. view fisheye: render on the raw frame.
    view: str = "robot_c922"
    mount_tcp_cam: list = field(default_factory=lambda: [0.0, 0.06, 0.12])   # TCP (between the finger ends) in the camera frame, m
    mount_pitch_deg: float = 25.0          # fingers tilted up towards the optical axis (rotation about camera x)

    def mount_T(self) -> np.ndarray:
        """Gripper frame in the camera: +y (approach) = camera +z, +x (jaw axis) = camera +x, +z = x * y (camera up), pitched."""
        a = np.radians(self.mount_pitch_deg)
        Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        R0 = np.column_stack([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        T = np.eye(4); T[:3, :3] = Rx @ R0; T[:3, 3] = self.mount_tcp_cam
        return T

    def jaw_width(self, pinch_m):
        g = self.jaw_normalized(pinch_m)
        return self.robot_jaw_min_m + g * (self.robot_jaw_max_m - self.robot_jaw_min_m)

    def jaw_normalized(self, pinch_m):
        return np.clip((np.asarray(pinch_m, float) - self.human_pinch_min_m) / (self.human_pinch_max_m - self.human_pinch_min_m), 0.0, 1.0)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def gripper_from_keypoints(kp: np.ndarray, conf: np.ndarray | None, side: str, cfg: GripperConfig) -> dict:
    """kp (21, 3) in the wrist camera frame -> dict(valid, reason, p (3,), R (3, 3) columns = x y z, pinch_m, jaw_m, jaw_n)."""
    out = dict(valid=False, reason="", p=None, R=None, pinch_m=None, jaw_m=None, jaw_n=None, rot_valid=False)
    need = [WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, PINKY_MCP]
    if kp is None or not np.isfinite(kp[need]).all():
        out["reason"] = "missing_keypoints"; return out
    if conf is not None and min(conf[THUMB_TIP], conf[INDEX_TIP]) < cfg.min_confidence:
        out["reason"] = "low_confidence"; return out
    pt, pi = kp[THUMB_TIP], kp[INDEX_TIP]
    d = float(np.linalg.norm(pi - pt))
    out.update(p=0.5 * (pt + pi), pinch_m=d, jaw_m=float(cfg.jaw_width(d)), jaw_n=float(cfg.jaw_normalized(d)), valid=True)
    v1, v2 = kp[INDEX_MCP] - kp[WRIST], kp[PINKY_MCP] - kp[WRIST]
    n = np.cross(v1, v2)
    if np.linalg.norm(n) < cfg.min_palm_m2:
        out["reason"] = "palm_degenerate"; return out          # position / jaw valid, rotation held by the smoother
    if d < cfg.min_axis_m:
        out["reason"] = "pinch_closed_axis_undefined"; return out
    fwd = 0.5 * (kp[INDEX_MCP] + kp[PINKY_MCP]) - kp[WRIST]
    zr = _unit(n * cfg.palm_normal_sign[side])
    if cfg.orientation_source == "palm_forward":
        z = zr
        y = fwd - np.dot(fwd, z) * z
        if np.linalg.norm(y) < 1e-6:
            out["reason"] = "forward_parallel_to_palm_normal"; return out
        y = _unit(y); xj = np.cross(y, z)
    else:
        # thumb_index: the jaw axis is EXACTLY thumb->index (jaw tips land on the fingertips), palm normal made orthogonal to it
        xj = _unit(pi - pt)
        z = zr - np.dot(zr, xj) * xj
        if np.linalg.norm(z) < 1e-6:
            out["reason"] = "jaw_axis_parallel_to_palm_normal"; return out
        z = _unit(z); y = np.cross(z, xj)
        if np.dot(y, fwd) < 0:                                   # +y towards the fingers for BOTH hands
            xj, y = -xj, -y
    R = np.column_stack([xj, y, z])
    u, _, vt = np.linalg.svd(R); R = u @ vt                      # exact orthonormal, det +1 by construction
    out.update(R=R, rot_valid=True, jaw_axis="index_to_thumb" if np.dot(xj, pi - pt) < 0 else "thumb_to_index")
    return out
