"""IMU motion-consistency score for Orbbec 3D tracks (tracker v2 building block; 10-23).

Not a hard acceleration equality: both signals are low-passed (gaussian SIG_S) and compared over WIN_S windows by
  ncc   = sum(a_o . a_i) / sqrt(sum|a_o|^2 sum|a_i|^2)      direction + timing agreement of the 3D linear acceleration
  energy gate: windows where neither moves (rms < E_MIN) are 'still' -> no evidence (score NaN), never a reject.
IMU linear acceleration in the base frame = R_B_tcp(t) f_tcp(t) - g, R_B_tcp from the gyro propagated from an anchor rotation;
lever arm ignored (low-pass keeps the omega^2 r / alpha x r terms small; the score is a correlation, not a difference).
    imu_lin_acc(D, t_grid, anchor_t, anchor_R) -> (n,3);  track_acc(t, c) -> (t_grid, a);  window_ncc(a_o, a_i, ...) -> (n,)"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

SIG_S = 0.08; WIN_S = 0.6; E_MIN = 0.35; HZ = 30.0


def grid(t0, t1):
    return np.arange(t0, t1 + 1, int(1e9 / HZ), dtype=np.int64)


def imu_lin_acc(D, C, g, anchor_t, anchor_R):
    """specific force -> base frame (gyro-propagated rotation) -> minus gravity (estimated as the median over the window) -> low-pass"""
    ti = D["imu_t"]; f = np.stack([np.interp(g, ti, D["acc"][:, k]) for k in range(3)], 1)
    Rt = anchor_R * C.gyro_rel(D, np.full(len(g), anchor_t), g); fB = Rt.apply(f)
    gB = np.median(fB, 0); a = fB - gB
    return gaussian_filter1d(a, SIG_S * HZ, axis=0, mode="nearest")


def track_acc(t, c, g, max_gap_s=0.25):
    """Orbbec centre (valid samples) -> linear on the grid inside stretches (gaps <= max_gap_s) -> low-pass -> 2nd derivative; NaN elsewhere"""
    v = np.isfinite(c[:, 0]); tv, cv = t[v], c[v]; out = np.full((len(g), 3), np.nan)
    if len(tv) < 5: return out
    j = np.clip(np.searchsorted(tv, g, side="right"), 1, len(tv) - 1); ok = ((tv[j] - tv[j - 1]) <= max_gap_s * 1e9) & (g >= tv[0]) & (g <= tv[-1])
    f = ((g - tv[j - 1]) / np.maximum(tv[j] - tv[j - 1], 1))[:, None]; P = cv[j - 1] * (1 - f) + cv[j] * f
    x = np.r_[0, ok.astype(int), 0]; s_, e_ = np.flatnonzero(np.diff(x) == 1), np.flatnonzero(np.diff(x) == -1)
    for a, b in zip(s_, e_):
        if b - a < 8: continue
        Ps = gaussian_filter1d(P[a:b], SIG_S * HZ, axis=0, mode="nearest"); out[a:b] = np.gradient(np.gradient(Ps, 1 / HZ, axis=0), 1 / HZ, axis=0)
    return out


def window_ncc(ao, ai, win_s=WIN_S, e_min=E_MIN):
    """per grid sample: ncc over the centred window; NaN if the window has < 70 % finite Orbbec samples or both sides are still"""
    n = len(ao); h = int(win_s * HZ / 2); out = np.full(n, np.nan)
    fin = np.isfinite(ao[:, 0])
    for k in range(h, n - h):
        sl = slice(k - h, k + h + 1); m = fin[sl]
        if m.mean() < 0.7: continue
        A, B = ao[sl][m], ai[sl][m]; ea, eb = np.sqrt((A * A).sum(1).mean()), np.sqrt((B * B).sum(1).mean())
        if max(ea, eb) < e_min: continue
        out[k] = float((A * B).sum() / np.sqrt((A * A).sum() * (B * B).sum() + 1e-12))
    return out


AHRS = dict(KP=1.0, KI=0.005, ACC_TOL=0.8, GYRO_MAX=3.0)   # scripts/handumi_canonical_track.py CANON_V1 Mahony gains


def mahony(ti, gyro, acc, bias, P=AHRS):
    """R_g_body (body -> gravity world, +Z up, yaw arbitrary) at every IMU sample; accelerometer corrects roll/pitch only
    (same filter as scripts/handumi_canonical_track.mahony, bias given)."""
    t = (ti - ti[0]) / 1e9; n = len(t); g_rest = float(np.median(np.linalg.norm(acc, axis=1)))
    a0 = acc[:50].mean(0); a0 /= np.linalg.norm(a0); Rm = R.align_vectors([[0, 0, 1.0]], [a0])[0].as_matrix()
    Rs = np.empty((n, 3, 3)); integ = np.zeros(3)
    for k in range(n):
        Rs[k] = Rm
        if k == n - 1: break
        dt = min(max(float(t[k + 1] - t[k]), 0.0), 0.05); w = gyro[k] - bias; an = np.linalg.norm(acc[k])
        if abs(an - g_rest) < P["ACC_TOL"] and np.linalg.norm(w) < P["GYRO_MAX"]:
            e = np.cross(acc[k] / an, Rm[2]); integ += P["KI"] * e * dt; w = w + P["KP"] * e + integ
        Rm = Rm @ R.from_rotvec(w * dt).as_matrix()
        if k % 200 == 0: U, _, Vt = np.linalg.svd(Rm); Rm = U @ Vt
    return Rs


def ahrs_base(D, g, anchor_t, anchor_R):
    """R_B_tcp on grid g: Mahony tilt + one yaw aligned to anchor_R at anchor_t (rig frame = TCP frame in fgo_data)"""
    if "Rg" not in D: D["Rg"] = mahony(D["imu_t"], D["gyro"], D["acc"], D.get("b_rest", D["b0"]))
    from scipy.spatial.transform import Slerp
    S = Slerp(D["imu_t"] / 1e9, R.from_matrix(D["Rg"])); ti = D["imu_t"]
    Ra = S(np.clip(anchor_t, ti[0], ti[-1]) / 1e9); M = anchor_R * Ra.inv()          # world_B <- gravity world
    if D.get("tagfree"): M = anchor_R          # 11-11 tag-free take: the anchor IS the world yaw of the Mahony gravity frame (fgo_data._load_arm_tagfree)
    yaw = np.arctan2(M.as_matrix()[1, 0], M.as_matrix()[0, 0]); Rz = R.from_euler("z", yaw)
    return Rz * S(np.clip(g, ti[0], ti[-1]) / 1e9)


def rest_bias(D, take, FD):
    """gyro bias from the rest periods between demo windows (samples within 1.5 deg/s of the period median).  fgo_core.prep's
    |omega| < 3 deg/s stillness test misses the true bias when |bias| itself is ~3-4 deg/s (123747: 1.5 deg/s error)."""
    ti, g = D["imu_t"], D["gyro"]; T = FD.trims()[take]; S = []
    for i in range(len(T) - 1):
        m = (ti > T[i]["w1"]) & (ti < T[i + 1]["w0"]); w = g[m]
        if m.sum() < 50: continue
        st = np.linalg.norm(w - np.median(w, 0), axis=1) < np.radians(1.5)
        if st.sum() > 200: S.append(w[st])
    return np.median(np.concatenate(S), 0) if S else D["b0"]
