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


PRE_TF = 1.0


def arm_rotations(take, t, track_c, arm, anchor_ok=None, t_need=None):
    """(n,3,3) AHRS R_B_tcp on the Orbbec grid t (NaN outside demo windows); track_c = a reference centre track for the sync lag;
    anchor_ok (bool on the take's tag frames) restricts the yaw-anchor frames (hidden-tag runs: visible tags only, no GT leakage)"""
    D = FD.load_arm(take, arm, with_slam=False); C.FC.prep(D); D["b_rest"] = IM.rest_bias(D, take, FD)
    Rb = D["R_tag"]; tg2 = (D["ntags"] >= 2) & (D["n_tagrot"] >= 2) & np.isfinite(Rb[:, 0, 0]); n = len(t); Rm = np.full((n, 3, 3), np.nan)
    if anchor_ok is not None: tg2 &= anchor_ok
    for Wd in FD.trims()[take]:
        # 11-11: t_need = (lo, hi) of the caller's frames -> skip the other demo windows (each window is solved independently, so the
        # result on [lo, hi] is unchanged; solve2 on a 20-demo take recomputed all 20 windows per demo = O(n^2), ~3.6 min per demo-arm)
        if t_need is not None and (Wd["w1"] + 2e9 < t_need[0] or Wd["a0"] > t_need[1]): continue
        # 11-11 tag-free: the CoTracker pack starts TF_PRE_S (0.8 s) before a0 with the units resting at their L and solve2 seeds there;
        # with Rm NaN before a0 every seed got q = NaN -> no trusted points -> the first pose came from re-acquisition ~2.6 s into the
        # demo (shirt / 3-6 cm off, validation clone 20261105_164114).  The AHRS is defined at any time -> start the window PRE_TF earlier.
        pre = PRE_TF * 1e9 if D.get("tagfree") else 0
        k = np.flatnonzero((t >= Wd["a0"] - pre) & (t <= Wd["w1"] + 2e9))
        if not len(k): continue
        anc = np.flatnonzero((t >= Wd["a0"]) & (t <= Wd["w1"] + 2e9) & tg2)
        if not len(anc) and tg2.any(): j_ = np.flatnonzero(tg2); anc = j_[[np.argmin(np.abs(t[j_] - t[k[0]]))]]
        # 10-30 takes 110211 / 111136 / 111615: the active (left-hand) rig never showed >= 2 of its tags at once in the whole take ->
        # no anchor -> Rm NaN -> CoTracker coverage 0 for every demo.  Same single-tag fallback as cotrack/gate.py (10-29 take 192644).
        if not len(anc):
            tg1 = (D["ntags"] >= 1) & (D["n_tagrot"] >= 1) & np.isfinite(Rb[:, 0, 0])
            if anchor_ok is not None: tg1 &= anchor_ok
            a1 = np.flatnonzero((t >= Wd["a0"]) & (t <= Wd["w1"] + 2e9) & tg1)
            if len(a1): anc = a1
            elif tg1.any(): j_ = np.flatnonzero(tg1); anc = j_[[np.argmin(np.abs(t[j_] - t[k[0]]))]]
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


def tf_rest_centre(take, arm, R_rest):
    """11-11 tag-free start seed: the jaw tip (TCP) rests ON the arm's L tape (protocol) -> body centre = L + R_rest o (o = TCP->body
    offset of the arm's rig, body_offset_by_rig_v1).  Replaces the fixed REST_XYZ (median of dark unit points = 4-8 cm outward of the
    true body origin: validation clone 20261105_164114 solved c0 == seed, tag body 20-84 mm away, constant over the demo)."""
    L = json.loads((HERE / f"tagfree_L_{take}.json").read_text())["L"][arm]["xyz"]
    rig = json.loads((FD.TAG / take / "tagfree.json").read_text())["rig_of_arm"][arm]
    o = np.array(json.loads((HERE / "body_offset_by_rig_v1.json").read_text())["offsets_by_rig"][rig]) / 1000
    return np.array(L, float) + np.asarray(R_rest) @ o
