"""Temporal smoothing of the gripper trajectory (renderer spec §7, §19). Offline: gaps are filled first (linear / SLERP between
the nearest valid frames, hold at the ends), then an EMA runs forward and, with zero_phase, again backward (no lag).
Rotations are never averaged as matrices: quaternions, sign-aligned, SLERP-based EMA. Output never contains NaN.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


def _fill_vec(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(valid); out = x.copy()
    if len(idx) == 0:
        return np.zeros_like(x)
    for c in range(x.shape[1]):
        out[:, c] = np.interp(np.arange(len(x)), idx, x[idx, c])
    return out


def _fill_rot(q: np.ndarray, valid: np.ndarray) -> np.ndarray:
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return np.tile([0, 0, 0, 1.0], (len(q), 1))
    if len(idx) == 1:
        return np.tile(q[idx[0]], (len(q), 1))
    t = np.clip(np.arange(len(q)), idx[0], idx[-1])
    return Slerp(idx, R.from_quat(q[idx]))(t).as_quat()


def _ema_vec(x, a):
    y = x.copy()
    for i in range(1, len(x)):
        y[i] = a * x[i] + (1 - a) * y[i - 1]
    return y


def _ema_rot(q, a):
    y = q.copy()
    for i in range(1, len(q)):
        prev = R.from_quat(y[i - 1]); cur = R.from_quat(q[i])
        y[i] = Slerp([0, 1], R.concatenate([prev, cur]))([a]).as_quat()[0]   # move a fraction `a` towards the new sample
    return y


def smooth_trajectory(p, q, jaw, pos_valid, rot_valid, *, alpha_position=0.3, alpha_rotation=0.2, alpha_jaw=0.3, zero_phase=True):
    """p (N,3), q (N,4 xyzw), jaw (N,) -> smoothed copies. pos_valid covers p and jaw; rot_valid covers q."""
    p = _fill_vec(np.asarray(p, float), pos_valid)
    jaw = _fill_vec(np.asarray(jaw, float)[:, None], pos_valid)[:, 0]
    q = np.asarray(q, float).copy(); q[~rot_valid] = [0, 0, 0, 1]
    q = _fill_rot(q, rot_valid)
    for i in range(1, len(q)):                                    # hemisphere continuity
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    ps, js, qs = _ema_vec(p, alpha_position), _ema_vec(jaw[:, None], alpha_jaw)[:, 0], _ema_rot(q, alpha_rotation)
    if zero_phase:
        ps = _ema_vec(ps[::-1], alpha_position)[::-1]
        js = _ema_vec(js[::-1, None], alpha_jaw)[::-1, 0]
        qs = _ema_rot(qs[::-1], alpha_rotation)[::-1]
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)
    return ps, qs, js
