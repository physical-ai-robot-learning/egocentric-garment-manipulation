"""10-24 shared helpers for tracker v32 (copied from refine_v4.py, which stays frozen): per-arm AHRS rotation on the Orbbec grid
(rest-period bias, accel tilt, one yaw anchor per demo window at its first >= 2-tag raw R_tag frame or the nearest other demo's,
IMU clock lag per demo from a reference track) and the translation-only rigid-model fit."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
import core as C  # noqa
import imu_motion as IM  # noqa
import build_candidate_v0 as V0  # noqa
FD = C.FD
HERE = Path(__file__).resolve().parent


def arm_rotations(take, t, track_c, arm, anchor_ok=None):
    """(n,3,3) AHRS R_B_tcp on the Orbbec grid t (NaN outside demo windows); track_c = a reference centre track for the sync lag;
    anchor_ok (bool on the take's tag frames) restricts the yaw-anchor frames (hidden-tag runs: visible tags only, no GT leakage)"""
    D = FD.load_arm(take, arm, with_slam=False); C.FC.prep(D); D["b_rest"] = IM.rest_bias(D, take, FD)
    Rb = D["R_tag"]; tg2 = (D["ntags"] >= 2) & (D["n_tagrot"] >= 2) & np.isfinite(Rb[:, 0, 0]); n = len(t); Rm = np.full((n, 3, 3), np.nan)
    if anchor_ok is not None: tg2 &= anchor_ok
    for Wd in FD.trims()[take]:
        k = np.flatnonzero((t >= Wd["a0"]) & (t <= Wd["w1"] + 2e9))
        if not len(k): continue
        anc = np.flatnonzero((t >= Wd["a0"]) & (t <= Wd["w1"] + 2e9) & tg2)
        if not len(anc) and tg2.any(): j_ = np.flatnonzero(tg2); anc = j_[[np.argmin(np.abs(t[j_] - t[k[0]]))]]
        if not len(anc): continue
        lag, _ = V0.sync_lag(D, t, track_c, k, t[anc[0]], R.from_matrix(Rb[anc[0]])) if np.isfinite(track_c[k, 0]).sum() > 30 else (0, 0.0)
        Rm[k] = IM.ahrs_base(D, t[k] + lag, t[anc[0]] + lag, R.from_matrix(Rb[anc[0]])).as_matrix()
    return Rm


def rigid_fit(pts, Rm, M, c0, steps=(0.04, 0.025, 0.015, 0.012, 0.012), min_inl=40, fit_r=0.09):
    """translation-only fit of model M (body frame) rotated by Rm, starting at c0 -> (T, n_inl, fit, rmse) or None"""
    T = c0.copy(); Mr = M @ Rm.T
    for r_ in steps:
        d, j = cKDTree(Mr + T).query(pts, distance_upper_bound=r_); inl = np.isfinite(d)
        if inl.sum() < min_inl // 2: return None
        T = T + np.median(pts[inl] - (Mr[j[inl]] + T), axis=0)
    d, _ = cKDTree(Mr + T).query(pts); near = np.linalg.norm(pts - T, axis=1) < fit_r; inl = d < steps[-1]
    if near.sum() == 0: return None
    return T, int(inl.sum()), float((inl & near).sum() / near.sum()), float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else np.nan
