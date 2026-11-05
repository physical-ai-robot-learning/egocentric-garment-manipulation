"""[2026-10-14 Agent B] X-VLA Soft-Fold -> canonical (Agent A contract) per-episode trajectories.  Pure numpy, unit-tested.

Source contract (reports/xvla_contract.md, all PROVEN on data):
  eef_6d per arm [xyz m | rot6d = R[:, :2] row-interleaved (FIRST TWO COLUMNS) | gripper width m], L = 0:10, R = 10:20;
  pose = Piper link6 FLANGE in each arm's own base frame; per-arm clocks eef_left_time / eef_right_time; images on time_stamp.
Decisions 2026-10-14 (user): 20 Hz common timeline (contract ACTION_DT_S); flange kept (offset only in metadata);
  actions = per-sample chunks inv(T(t)) @ T(t+k), k = 1..32 built at TRAINING time from the stored absolute poses.
"""
import pathlib, sys

import numpy as np

from laundry_shared.contracts.action_contract import ACTION_DT_S, T_to_pose9, inv_T, make_T, matrix_to_rot6d  # noqa: E402

# contract TCP axes (x = approach, y = wrist image x, z = wrist image y) from Piper link6 axes (z6 = approach,
# cam X = -y6, cam Y = +x6; measured from wrist optical flow, 10/10 arm-episodes).  R_tcp = R_link6 @ M_LINK6_TO_TCP
M_LINK6_TO_TCP = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])
FLANGE_TO_JAW_ALONG_APPROACH_M = 0.1358          # metadata only (tcp_offset_applied = False)
GRIP_OPEN_REF_M = 0.060                           # width that maps to g = 1 (≈ p95 of the data); g = clip(w / ref, 0, 1)
MAX_SRC_GAP_S = 0.15                              # 2026-10-14: 0.10 split the 15 Hz `_new` folders (EEF gaps to 120 ms) into fragments; at their p50 0.15 m/s a 150 ms linear bridge errs a few mm
MAX_IMG_RESIDUAL_S = 0.06                         # 15 Hz frames with 100 ms clock gaps reach 50 ms; a nearest frame further than this -> invalid
SOURCE_META = dict(data_source="XVLA_CLOTH", source_robot="AGILEX_PIPER_X2", source_pose_frame="link6_flange",
                   tcp_offset_along_approach_m=FLANGE_TO_JAW_ALONG_APPROACH_M, tcp_offset_applied=False,
                   tcp_axes="contract: R_tcp = R_link6 @ M_LINK6_TO_TCP", skill="FOLD", instruction="fold the cloth",
                   T_LR_valid=False, rate_hz=1.0 / ACTION_DT_S)


def xvla_rot6d_to_matrix(v6):
    """X-VLA convention: v = [R00, R01, R10, R11, R20, R21] (first two COLUMNS); Gram-Schmidt, stacked as columns."""
    v6 = np.asarray(v6, np.float64); a1, a2 = v6[..., 0:5:2], v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, -1, keepdims=True) * b1; b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    return np.stack([b1, b2, np.cross(b1, b2)], -1)


def matrix_to_xvla_rot6d(R):
    R = np.asarray(R, np.float64); return R[..., :, :2].reshape(R.shape[:-2] + (6,))


def xvla_arm_to_T(arm10):
    """(N, 10) X-VLA arm block -> (N, 4, 4) pose in contract TCP axes at the flange point, and (N,) gripper width (m)."""
    arm10 = np.asarray(arm10, np.float64)
    return make_T(xvla_rot6d_to_matrix(arm10[:, 3:9]) @ M_LINK6_TO_TCP, arm10[:, 0:3]), arm10[:, 9]


def gripper_canonical(width_m):
    """source width (m, larger = open; proven on 12 video events) -> contract 0 = closed, 1 = open"""
    return np.clip(np.asarray(width_m, np.float64) / GRIP_OPEN_REF_M, 0.0, 1.0)


# ---------------------------------------------------------------- interpolation
def _quat_from_R(R):
    R = np.asarray(R, np.float64)                                # Shepperd: branch on the largest diagonal term
    q = np.zeros(R.shape[:-2] + (4,)); tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    for idx in np.ndindex(R.shape[:-2]):
        m = R[idx]; t = tr[idx]
        if t > 0:
            s = np.sqrt(t + 1) * 2; q[idx] = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4]
        else:
            i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]])); j, k = (i + 1) % 3, (i + 2) % 3
            s = np.sqrt(1 + m[i, i] - m[j, j] - m[k, k]) * 2; v = np.zeros(4)
            v[i] = s / 4; v[j] = (m[j, i] + m[i, j]) / s; v[k] = (m[k, i] + m[i, k]) / s; v[3] = (m[k, j] - m[j, k]) / s; q[idx] = v
    return q                                                     # xyzw


def _R_from_quat(q):
    q = np.asarray(q, np.float64); q = q / np.linalg.norm(q, axis=-1, keepdims=True); x, y, z, w = (q[..., i] for i in range(4))
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def _slerp(q0, q1, u):
    d = np.sum(q0 * q1, -1); q1 = np.where(d[..., None] < 0, -q1, q1); d = np.abs(d)
    th = np.arccos(np.clip(d, -1, 1)); s = np.sin(th); small = s < 1e-8
    a = np.where(small, 1 - u, np.sin((1 - u) * th) / np.where(small, 1, s)); b = np.where(small, u, np.sin(u * th) / np.where(small, 1, s))
    q = a[..., None] * q0 + b[..., None] * q1; return q / np.linalg.norm(q, axis=-1, keepdims=True)


def resample_pose(t_src, T_src, g_src, t_grid, max_gap=MAX_SRC_GAP_S):
    """translation linear, rotation SLERP, gripper linear.  No extrapolation: grid times outside [t_src[0], t_src[-1]] or whose
    bracketing source samples are > max_gap apart are invalid.  -> T (G,4,4), g (G,), valid (G,), bracket_gap (G,)"""
    t_src = np.asarray(t_src, np.float64); t_grid = np.asarray(t_grid, np.float64)
    if np.any(np.diff(t_src) <= 0): raise ValueError("source timestamps not strictly increasing")
    i1 = np.clip(np.searchsorted(t_src, t_grid, side="right"), 1, len(t_src) - 1); i0 = i1 - 1
    gap = t_src[i1] - t_src[i0]; u = np.clip((t_grid - t_src[i0]) / gap, 0, 1)
    inside = (t_grid >= t_src[0] - 1e-9) & (t_grid <= t_src[-1] + 1e-9)
    q = _quat_from_R(T_src[:, :3, :3]); R = _R_from_quat(_slerp(q[i0], q[i1], u))
    p = (1 - u)[:, None] * T_src[i0, :3, 3] + u[:, None] * T_src[i1, :3, 3]
    g = (1 - u) * g_src[i0] + u * g_src[i1]
    return make_T(R, p), g, inside & (gap <= max_gap), gap


def nearest_frame(t_frames, t_grid):
    t_frames = np.asarray(t_frames, np.float64); i = np.clip(np.searchsorted(t_frames, t_grid), 1, len(t_frames) - 1)
    j = np.where(np.abs(t_frames[i - 1] - t_grid) <= np.abs(t_frames[i] - t_grid), i - 1, i)
    return j, np.abs(t_frames[j] - t_grid)


# ---------------------------------------------------------------- episode
def build_episode(e6, t_left, t_right, t_frames, dt=ACTION_DT_S):
    """-> dict with the 20 Hz grid: t, T_L, T_R (contract axes, own base frames, flange), gw_L/gw_R (width m), g_L/g_R (0..1),
    img_idx (frame index per grid time; the three cameras share time_stamp), img_res, valid, per-grid diagnostics."""
    T_L, w_L = xvla_arm_to_T(e6[:, 0:10]); T_R, w_R = xvla_arm_to_T(e6[:, 10:20])
    t0 = max(t_left[0], t_right[0], t_frames[0]); t1 = min(t_left[-1], t_right[-1], t_frames[-1])
    t = t0 + dt * np.arange(int(np.floor((t1 - t0) / dt + 1e-9)) + 1)
    TL, wL, vL, gapL = resample_pose(t_left, T_L, w_L, t); TR, wR, vR, gapR = resample_pose(t_right, T_R, w_R, t)
    idx, res = nearest_frame(t_frames, t)
    valid = vL & vR & (res <= MAX_IMG_RESIDUAL_S)
    return dict(t=t, T_L=TL, T_R=TR, gw_L=wL, gw_R=wR, g_L=gripper_canonical(wL), g_R=gripper_canonical(wR), img_idx=idx, img_res=res,
                valid=valid, gap_L=gapL, gap_R=gapR)


def valid_segments(valid, min_len):
    """contiguous runs of valid grid points with length >= min_len -> list of (start, stop)"""
    segs = []; s = None
    for i, v in enumerate(np.append(valid, False)):
        if v and s is None: s = i
        elif not v and s is not None:
            if i - s >= min_len: segs.append((s, i))
            s = None
    return segs


def abs_action_rows(T_L, T_R, g_L, g_R):
    """stored per-frame 'action' = absolute target = the pose/gripper at the NEXT grid frame, as [L pose9 | R pose9 | gL | gR]
    in each arm's own base frame (contract rot6d = first two ROWS).  With n_obs_steps = 2 the DiT chunk starts at frame t-1,
    whose stored action is exactly the pose at t: chunk row 0 = T(t), rows 1..32 = T(t+1..t+32)."""
    n = len(T_L); nxt = np.minimum(np.arange(n) + 1, n - 1)
    return np.concatenate([T_to_pose9(T_L[nxt]), T_to_pose9(T_R[nxt]), g_L[nxt, None], g_R[nxt, None]], -1).astype(np.float32)


def relative_chunk_from_abs(chunk_abs):
    """(H+1, 20) absolute rows with row 0 = T(t) -> (H, 20) contract chunk: rows k = inv(T(t)) @ T(t+k), grippers absolute."""
    from laundry_shared.contracts.action_contract import pose9_to_T
    c = np.asarray(chunk_abs, np.float64); TL = pose9_to_T(c[:, 0:9]); TR = pose9_to_T(c[:, 9:18])
    out = np.zeros((len(c) - 1, 20))
    out[:, 0:9] = T_to_pose9(inv_T(TL[0])[None] @ TL[1:]); out[:, 9:18] = T_to_pose9(inv_T(TR[0])[None] @ TR[1:])
    out[:, 18] = c[1:, 18]; out[:, 19] = c[1:, 19]
    return out.astype(np.float32)


def proprio_rows(T_L, T_R, g_L, g_R):
    """contract proprio (30D) with the episode anchor = first frame of the (segment) episode; T_LR unknown for this source."""
    from laundry_shared.contracts.observation_contract import PROPRIO_DIM, P_GRIP_L, P_GRIP_R, P_L_POSE, P_R_POSE, P_T_LR, P_T_LR_VALID
    n = len(T_L); v = np.zeros((n, PROPRIO_DIM))
    v[:, P_L_POSE] = T_to_pose9(inv_T(T_L[0])[None] @ T_L); v[:, P_R_POSE] = T_to_pose9(inv_T(T_R[0])[None] @ T_R)
    v[:, P_GRIP_L], v[:, P_GRIP_R] = g_L, g_R; v[:, P_T_LR] = T_to_pose9(np.eye(4))[None]; v[:, P_T_LR_VALID] = 0.0
    return v.astype(np.float32)


__all__ = ["M_LINK6_TO_TCP", "SOURCE_META", "xvla_rot6d_to_matrix", "matrix_to_xvla_rot6d", "xvla_arm_to_T", "gripper_canonical",
           "resample_pose", "nearest_frame", "build_episode", "valid_segments", "abs_action_rows", "relative_chunk_from_abs",
           "proprio_rows", "matrix_to_rot6d"]
