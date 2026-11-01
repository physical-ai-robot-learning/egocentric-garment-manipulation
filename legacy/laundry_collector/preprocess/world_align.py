"""Common TABLE/world frame for both wrists (spec §14, §15, renderer spec §22).

Each MASt3R-SLAM run has its own non-metric world. The initialization hold pins it: during the `initialization` phase the
hand (virtual TCP) of side s sits on marker X_s, whose pose in the table frame T_world_tcp_init(s) is known
(world_config.json: marker position + the orientation convention of how the hand is placed). With the wrist camera ->
TCP extrinsic T_camera_tcp(s) and a metric scale `scale` for the SLAM translations:

    T_world_camera(t) = T_world_tcp_init @ inv(T_camera_tcp) @ inv(S(T_slam_camera(t0))) @ S(T_slam_camera(t))

S() multiplies the translation by `scale`; t0 = the initialization-window frame used as anchor (median of the valid frames
in the hold). The scale is NOT observable from one monocular wrist video: a ScaleEstimator supplies it (global RGB-D
wrist tracking, marker distance, ... -- P2), and the result records which estimator was used.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R


def pose_mat(xyz, qxyzw) -> np.ndarray:
    xyz, q = np.atleast_2d(xyz), np.atleast_2d(qxyzw)
    T = np.tile(np.eye(4), (len(xyz), 1, 1)); T[:, :3, :3] = R.from_quat(q).as_matrix(); T[:, :3, 3] = xyz
    return T


def inv_se3(T: np.ndarray) -> np.ndarray:
    Ti = np.zeros_like(T); Rt = np.swapaxes(T[..., :3, :3], -1, -2)
    Ti[..., :3, :3] = Rt; Ti[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, T[..., :3, 3]); Ti[..., 3, 3] = 1
    return Ti


class ScaleEstimator(ABC):
    name = "base"

    @abstractmethod
    def scale(self, side: str, T_slam_camera: np.ndarray, valid: np.ndarray, context: dict) -> tuple[float, dict]:
        """-> (metres per SLAM unit, diagnostics)."""


class FixedScale(ScaleEstimator):
    name = "fixed"

    def __init__(self, s: float):
        self.s = float(s)

    def scale(self, side, T, valid, context):
        return self.s, dict(source="fixed")


@dataclass
class AlignResult:
    T_world_camera: np.ndarray        # (N, 4, 4), metric, table frame
    valid: np.ndarray                 # (N,)
    anchor_frame: int
    scale: float
    diagnostics: dict


class WorldAligner:
    def __init__(self, world_config: dict, T_camera_tcp: dict, scale_estimator: ScaleEstimator):
        self.world, self.T_cam_tcp, self.scale_est = world_config, T_camera_tcp, scale_estimator

    def T_world_tcp_init(self, side: str) -> np.ndarray:
        m = self.world["init_markers"]["X_L" if side == "left" else "X_R"]
        if any(v is None for v in m["position"]):
            raise ValueError(f"world_config: marker X_{side[0].upper()} position not measured")
        T = np.eye(4); T[:3, 3] = m["position"]
        if m.get("R_world_tcp") is not None:                    # hand orientation on the marker, if specified as a matrix
            T[:3, :3] = np.asarray(m["R_world_tcp"], float)
        return T

    def align(self, side: str, T_slam_camera: np.ndarray, valid: np.ndarray, init_mask: np.ndarray, context: dict | None = None) -> AlignResult:
        if self.T_cam_tcp.get(side) is None:
            raise ValueError(f"T_camera_tcp[{side}] not calibrated (world_config.json)")
        idx = np.flatnonzero(init_mask & valid)
        if len(idx) < 5:
            raise ValueError(f"{side}: only {len(idx)} valid SLAM frames in the initialization hold")
        s, diag = self.scale_est.scale(side, T_slam_camera, valid, context or {})
        Ts = T_slam_camera.copy(); Ts[:, :3, 3] *= s
        t0 = int(idx[len(idx) // 2])
        # hold quality: how still the camera was during the hold (metric, after scaling)
        p = Ts[idx, :3, 3]; diag["init_hold_spread_mm"] = float(np.linalg.norm(p - np.median(p, 0), axis=1).max() * 1000)
        T_wc = self.T_world_tcp_init(side) @ inv_se3(np.asarray(self.T_cam_tcp[side], float)) @ inv_se3(Ts[t0]) @ Ts
        return AlignResult(T_wc, valid.copy(), t0, s, dict(diag, scale_estimator=self.scale_est.name))

    def align_left(self, *a, **k): return self.align("left", *a, **k)
    def align_right(self, *a, **k): return self.align("right", *a, **k)


class HandToWorldTransformer:
    """T_world_gripper(t) = T_world_camera(t) @ T_camera_gripper(t). The SAME T_camera_gripper drives the virtual-gripper
    render and the action labels (renderer spec §21)."""

    @staticmethod
    def transform(world_camera_pose: np.ndarray, camera_gripper_pose: np.ndarray) -> np.ndarray:
        return np.einsum("...ij,...jk->...ik", world_camera_pose, camera_gripper_pose)


def relative_actions(T_world: np.ndarray, ks=range(1, 33)) -> np.ndarray:
    """ΔT(t, k) = T(t)^-1 T(t+k) (renderer spec §23); (N - max k, K, 4, 4)."""
    ks = list(ks); n = len(T_world) - max(ks)
    Ti = inv_se3(T_world[:n])
    return np.stack([np.einsum("nij,njk->nik", Ti, T_world[k:k + n]) for k in ks], axis=1)
