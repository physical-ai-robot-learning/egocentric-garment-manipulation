"""Canonical 20D bimanual action contract (shared by Agent A and Agent B; the ONLY definition of it).

Per chunk step k = 1..H (row k-1 of an (H, 20) chunk), relative to the TCP poses T_L(t), T_R(t) that the policy
observed when it was called:

    dT_L(t, k) = inv(T_L(t)) @ T_L(t + k * ACTION_DT_S)
    dT_R(t, k) = inv(T_R(t)) @ T_R(t + k * ACTION_DT_S)

    row = [ L xyz 3 | L rot6d 6 | R xyz 3 | R rot6d 6 | gripper_L | gripper_R ]      (dims 0-8, 9-17, 18, 19)

    xyz      metres, expressed in the CURRENT TCP frame T(t) (body frame), not the base frame
    rot6d    first two ROWS of the rotation matrix, R[:2, :].reshape(6)  (umi pose_util / ego_cart20 convention);
             decoded with Gram-Schmidt on those rows
    gripper  absolute target aperture, 0 = closed, 1 = open (NOT relative)

The chunk is frame-invariant: any fixed world frame gives the same dT, so the robot side decodes it against its own
base-frame TCP poses.  Executing step k means commanding T(t) @ dT(t, k).
"""
from dataclasses import dataclass

import numpy as np

CONTRACT_VERSION = "laundry-contract/v0.1"

ACTION_DIM = 20
POSE_DIM = 9
L_POS, L_ROT, L_POSE = slice(0, 3), slice(3, 9), slice(0, 9)
R_POS, R_ROT, R_POSE = slice(9, 12), slice(12, 18), slice(9, 18)
GRIP_L, GRIP_R = 18, 19

ACTION_NAMES = ([f"left_d{c}" for c in "xyz"] + [f"left_rot6d_{i}" for i in range(6)]
                + [f"right_d{c}" for c in "xyz"] + [f"right_rot6d_{i}" for i in range(6)]
                + ["gripper_left", "gripper_right"])

GRIPPER_CLOSED, GRIPPER_OPEN = 0.0, 1.0
ROT6D_CONVENTION = "first_two_rows"

DEFAULT_HORIZON = 32                  # predicted steps per chunk
EXECUTE_STEP_CHOICES = (4, 8, 16)     # receding-horizon execution lengths under study; configurable, never hard-coded
ACTION_DT_S = 0.05                    # time between chunk steps (20 Hz).  Agent B's dataset rate must match.

# Plausibility bound of the CONTRACT (not a safety limit): no single relative step may move a TCP further than
# this from T(t).  32 steps * 50 ms = 1.6 s; 0.6 m over 1.6 s is already far beyond cloth-handling speed.
MAX_REL_TRANSLATION_M = 0.6
GRIPPER_TOL = 0.05                    # allowed overshoot outside [0, 1] before a chunk is rejected


# ---------------------------------------------------------------- rotation / pose helpers (numpy, batched)
def matrix_to_rot6d(R):
    R = np.asarray(R, np.float64)
    return R[..., :2, :].reshape(R.shape[:-2] + (6,))


def rot6d_to_matrix(d6):
    d6 = np.asarray(d6, np.float64)
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, -1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    return np.stack([b1, b2, np.cross(b1, b2)], -2)


def make_T(R, p):
    R = np.asarray(R, np.float64); p = np.asarray(p, np.float64)
    T = np.zeros(R.shape[:-2] + (4, 4)); T[..., :3, :3] = R; T[..., :3, 3] = p; T[..., 3, 3] = 1.0
    return T


def inv_T(T):
    T = np.asarray(T, np.float64); Rt = np.swapaxes(T[..., :3, :3], -1, -2)
    return make_T(Rt, -np.einsum("...ij,...j->...i", Rt, T[..., :3, 3]))


def T_to_pose9(T):
    T = np.asarray(T, np.float64)
    return np.concatenate([T[..., :3, 3], matrix_to_rot6d(T[..., :3, :3])], -1)


def pose9_to_T(v):
    v = np.asarray(v, np.float64)
    return make_T(rot6d_to_matrix(v[..., 3:9]), v[..., :3])


# ---------------------------------------------------------------- encode / decode
def encode_action_chunk(T_L_t, T_L_future, T_R_t, T_R_future, g_L, g_R):
    """T_*_future (H,4,4) absolute targets at t + k*ACTION_DT_S, k = 1..H; g_* (H,) absolute -> (H, 20) float32"""
    T_L_future = np.asarray(T_L_future); H = T_L_future.shape[0]
    out = np.zeros((H, ACTION_DIM))
    out[:, L_POSE] = T_to_pose9(inv_T(T_L_t)[None] @ T_L_future)
    out[:, R_POSE] = T_to_pose9(inv_T(T_R_t)[None] @ np.asarray(T_R_future))
    out[:, GRIP_L] = g_L; out[:, GRIP_R] = g_R
    return out.astype(np.float32)


@dataclass
class DecodedChunk:
    T_L: np.ndarray        # (H,4,4) absolute targets in the frame of T_L_t
    T_R: np.ndarray
    g_L: np.ndarray        # (H,) clipped to [0, 1]
    g_R: np.ndarray


def decode_action_chunk(chunk, T_L_t, T_R_t):
    c = np.asarray(chunk, np.float64)
    return DecodedChunk(np.asarray(T_L_t)[None] @ pose9_to_T(c[:, L_POSE]),
                        np.asarray(T_R_t)[None] @ pose9_to_T(c[:, R_POSE]),
                        np.clip(c[:, GRIP_L], 0.0, 1.0), np.clip(c[:, GRIP_R], 0.0, 1.0))


# ---------------------------------------------------------------- validation
@dataclass
class ChunkValidation:
    ok: bool
    reason: str = ""


def validate_action_chunk(chunk, horizon=None, min_steps=1):
    """Syntactic + plausibility check of a policy output.  Never raises; the robot loop rejects on ok=False."""
    try:
        c = np.asarray(chunk)
    except Exception as e:  # noqa: BLE001 - anything unconvertible is a reject
        return ChunkValidation(False, f"not_array:{type(e).__name__}")
    if c.dtype.kind not in "fiu":
        return ChunkValidation(False, f"dtype:{c.dtype}")
    if c.ndim != 2 or c.shape[1] != ACTION_DIM:
        return ChunkValidation(False, f"shape:{tuple(c.shape)} expected (H,{ACTION_DIM})")
    if horizon is not None and c.shape[0] != horizon:
        return ChunkValidation(False, f"horizon:{c.shape[0]} expected {horizon}")
    if c.shape[0] < min_steps:
        return ChunkValidation(False, f"too_short:{c.shape[0]} < {min_steps}")
    c = c.astype(np.float64)
    if not np.isfinite(c).all():
        return ChunkValidation(False, f"non_finite:{int((~np.isfinite(c)).sum())}")
    g = c[:, [GRIP_L, GRIP_R]]
    if (g < -GRIPPER_TOL).any() or (g > 1 + GRIPPER_TOL).any():
        return ChunkValidation(False, f"gripper_range:[{g.min():.3f},{g.max():.3f}]")
    for name, rot in (("left", L_ROT), ("right", R_ROT)):
        a1, a2 = c[:, rot][:, :3], c[:, rot][:, 3:]
        n1, n2 = np.linalg.norm(a1, axis=1), np.linalg.norm(a2, axis=1)
        if (n1 < 1e-3).any() or (n2 < 1e-3).any():
            return ChunkValidation(False, f"rot6d_degenerate:{name}")
        cos = np.abs(np.sum(a1 * a2, 1)) / (n1 * n2)
        if (cos > 0.99).any():
            return ChunkValidation(False, f"rot6d_collinear:{name}")
    for name, pos in (("left", L_POS), ("right", R_POS)):
        d = np.linalg.norm(c[:, pos], axis=1).max()
        if d > MAX_REL_TRANSLATION_M:
            return ChunkValidation(False, f"translation_implausible:{name}:{d:.3f}m")
    return ChunkValidation(True)


def identity_chunk(horizon=DEFAULT_HORIZON, g_L=1.0, g_R=1.0):
    """'stay where you are' chunk (useful for mocks and tests)"""
    c = np.zeros((horizon, ACTION_DIM), np.float32)
    eye6 = matrix_to_rot6d(np.eye(3))
    c[:, L_ROT] = eye6; c[:, R_ROT] = eye6; c[:, GRIP_L] = g_L; c[:, GRIP_R] = g_R
    return c
