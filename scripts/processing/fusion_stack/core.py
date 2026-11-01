"""10-23 marker-free SLAM<->IMU self-consistency for UMI wrist MASt3R trajectories.
Reads (never writes) other agents' data: raw MASt3R per-window segs, IMU via _fgo_1023/tools/fgo_data.load_arm, trims.
All times on the orbbec clock (fgo_data convention: SLAM t = wrist t - 20 ms, IMU = host + calibrated offset); an extra
per-arm SLAM time offset is estimated here by |omega| cross-correlation."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path.home() / "ego_collector/datasets/m3_runs/_fgo_1023/tools"))
import fgo_data as FD  # noqa
import fgo_core as FC  # noqa

M3 = FD.M3; FR = FD.FR
HZ = (0.1, 0.4, 0.8, 1.6)


# ---------------------------------------------------------------- MASt3R raw segments (per window dirs, works while windows are landing)
def slam_tag(take, cam):
    return f"umi_{take}" + ("" if cam == "wrist" else "_left")


def load_segs_raw(take, cam):
    tag = slam_tag(take, cam); man = json.loads((M3 / tag / "segs/manifest.json").read_text())
    fr = {f["i"]: f["t_ns"] for f in json.loads((FR / tag / "frames.json").read_text())["frames"]}
    segs, avail = [], []
    for w in man["windows"]:
        d = M3 / tag / "windows" / w["sub"] / "segs"
        csvs = sorted(d.glob("*.csv")) if d.exists() else []
        if not csvs: continue
        avail.append((w["t0_ns"] - int(FD.SLAM_DT), w["t1_ns"] - int(FD.SLAM_DT), w["w"]))
        for c in csvs:
            s0 = int(c.stem.split("_")[0]) + int(w["i0"])
            df = pd.read_csv(c)
            if len(df) == 0: continue
            bad = df["is_lost"].astype(str).str.lower().eq("true").to_numpy() | ~df["state"].isin(["INIT", "TRACKING"]).to_numpy()
            cut = int(np.argmax(bad)) if bad.any() else len(df); lost_end = bool(bad.any())
            df = df.iloc[:cut]
            q = df[["q_x", "q_y", "q_z", "q_w"]].to_numpy(float); p = df[["x", "y", "z"]].to_numpy(float)
            gi = s0 + df["frame_idx"].to_numpy(int)
            ok = np.isfinite(q).all(1) & np.isfinite(p).all(1) & (np.linalg.norm(q, axis=1) > 1e-3) & np.isin(gi, list(fr.keys()))
            if ok.sum() < 10: continue
            t = np.array([fr[i] for i in gi[ok]], np.int64) - int(FD.SLAM_DT)
            o = np.argsort(t, kind="stable"); t, u = np.unique(t[o], return_index=True); o = o[u]
            segs.append(dict(t=t, p=p[ok][o], R=R.from_quat(q[ok][o]), sim3=df["sim3_scale"].to_numpy(float)[ok][o],
                             kf=df["is_keyframe"].astype(str).str.lower().eq("true").to_numpy()[ok][o], lost_end=lost_end, gi0=s0, win=w["w"]))
    return segs, avail


# ---------------------------------------------------------------- IMU helpers
def gyro_rel(D, ta, tb):
    """bias-corrected gyro rotation from ta to tb (TCP frame), arrays"""
    return FC.Qat(D, ta).inv() * FC.Qat(D, tb)


def omega_mag_corr(D, segs, lags_ms=np.arange(-200, 201, 2)):
    """extra SLAM time offset (ns, added to SLAM t) maximising corr(|omega_slam|, |omega_gyro|)"""
    ti = D["imu_t"]; gm = np.linalg.norm(D["gyro"] - D.get("b0", 0), axis=1)
    dt_imu = np.median(np.diff(ti)) / 1e9; gms = gaussian_filter1d(gm, max(1, 0.02 / dt_imu))
    T, W = [], []
    for s in segs:
        if len(s["t"]) < 5: continue
        dt = np.diff(s["t"]) / 1e9; ok = dt < 0.1
        w = (s["R"][:-1].inv() * s["R"][1:]).magnitude() / np.maximum(dt, 1e-3)
        T.append(((s["t"][:-1] + s["t"][1:]) // 2)[ok]); W.append(w[ok])
    T = np.concatenate(T); W = np.concatenate(W); m = W < np.percentile(W, 99.5)
    best = (-9, 0, None); cs = []
    for L in lags_ms:
        g = np.interp(T[m] + L * 1e6, ti, gms); c = np.corrcoef(g, W[m])[0, 1]; cs.append(c)
        if c > best[0]: best = (c, L)
    return int(best[1] * 1e6), float(best[0]), np.array(cs)


def kabsch_rot(a, b, w=None):
    """RX minimising sum w |a - RX b|^2"""
    w = np.ones(len(a)) if w is None else w
    H = (b * w[:, None]).T @ a; U, S, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T @ U.T))
    return R.from_matrix(Vt.T @ np.diag([1, 1, d]) @ U.T)


def handeye_rot(D, segs, horizon=0.4, stride=3):
    """rotation-only hand-eye from SLAM-vs-gyro: rotvec(cam rel) = RX . rotvec(gyro rel)  (fgo convention M = RX^-1 A RX = B)"""
    A, B, TA, TB = [], [], [], []
    for s in segs:
        t = s["t"]; k = int(round(horizon * 30))
        if len(t) <= k: continue
        i = np.arange(0, len(t) - k, stride); j = i + k; ok = (t[j] - t[i]) < (horizon + 0.1) * 1e9
        i, j = i[ok], j[ok]
        if not len(i): continue
        A.append((s["R"][i].inv() * s["R"][j]).as_rotvec()); TA.append(t[i]); TB.append(t[j])
    A = np.concatenate(A); TA = np.concatenate(TA); TB = np.concatenate(TB)
    B = gyro_rel(D, TA, TB).as_rotvec(); m = (np.linalg.norm(B, axis=1) > np.radians(5))
    keep = m.copy()
    for _ in range(6):
        RX = kabsch_rot(A[keep], B[keep]); r = np.linalg.norm(A - RX.apply(B), axis=1)
        thr = max(3 * np.median(r[m]), np.radians(1.0)); keep = m & (r < thr)
    return RX, int(keep.sum()), float(np.degrees(np.median(r[keep]))), (A, B, TA, TB)


def refine_bias(D, RX, pairs):
    """constant gyro bias correction (TCP frame) so that gyro matches SLAM rotation rate; robust median"""
    A, B, TA, TB = pairs; dt = (TB - TA) / 1e9
    Bm = RX.inv().apply(A)                 # SLAM rotation expressed in TCP frame
    db = (B - Bm) / dt[:, None]; r = np.linalg.norm(B - Bm, axis=1); m = r < np.percentile(r, 50)
    return np.median(db[m], 0)


def integrate_q(D, bias):
    ti = D["imu_t"]; dt = np.diff(ti) / 1e9; inc = R.from_rotvec((D["gyro"][:-1] - bias) * dt[:, None])
    q = np.empty((len(ti), 4)); cur = R.identity(); q[0] = cur.as_quat()
    for k in range(len(inc)): cur = cur * inc[k]; q[k + 1] = cur.as_quat()
    D["Q"] = R.from_quat(q); D["Qs"] = Slerp(ti / 1e9, D["Q"]); D["bias_total"] = bias


def calibrate(D, segs, RX_fixed=None):
    """per arm: extra SLAM time offset, rotation-only hand-eye (joint with constant gyro bias). Mutates segs t and D gyro/Q."""
    FC.prep(D); off, c, cs = omega_mag_corr(D, segs)
    for s in segs: s["t"] = s["t"] + off
    RXe, n, med, pairs = handeye_rot(D, segs); A, B, TA, TB = pairs; dt = (TB - TA) / 1e9
    db = np.zeros(3); mb = np.linalg.norm(B, axis=1) > np.radians(2)
    for _ in range(8):
        Bc = B - db * dt[:, None]
        r = np.linalg.norm(A - RXe.apply(Bc), axis=1); thr = max(3 * np.median(r[mb]), np.radians(1.0)); keep = mb & (r < thr)
        RXe = kabsch_rot(A[keep], Bc[keep]) if RX_fixed is None else RX_fixed
        db = np.median(((B - RXe.inv().apply(A)) / dt[:, None])[keep], 0)
    res = {}
    for nm, X in (("est", RXe), ("harness", D["RX"])):
        res[nm] = float(np.degrees(np.median(np.linalg.norm(A - X.apply(B - db * dt[:, None]), axis=1))))
    integrate_q(D, D["b0"] + db)
    RXa = handeye_rot(D, segs)[0]
    cal = dict(RX_arm_only=RXa, he_arm_vs_used_deg=float(np.degrees((RXa.inv() * RXe).magnitude())), offset_ms=off / 1e6, corr=c, corr0=float(cs[len(cs) // 2]), RX_est=RXe, RX_harness=D["RX"], he_diff_deg=float(np.degrees((RXe.inv() * D["RX"]).magnitude())),
               dbias_dps=np.degrees(db).tolist(), b0_dps=np.degrees(D["b0"]).tolist(), res04_est_deg=res["est"], res04_harness_deg=res["harness"], n_pairs=int(keep.sum()))
    return cal


# ---------------------------------------------------------------- per-segment precompute
def prep_seg(D, s, RX):
    t = s["t"]; s["Rs"] = Slerp(t / 1e9, s["R"]); s["t0"] = int(t[0]); s["t1"] = int(t[-1]) - (int(0.2e9) if s["lost_end"] else 0)
    st = np.linalg.norm(np.diff(s["p"], axis=0), axis=1); dtt = np.diff(t) / 1e9
    med = np.median(st) + 1e-9; s["v_med"] = float(np.median(st / np.maximum(dtt, 1e-3)))
    # pose jump: a single-frame step far above the segment's typical step and its neighbours
    nb = np.maximum(np.r_[st[1:], st[-1]], np.r_[st[0], st[:-1]])
    jump = (st > 8 * med) & (st > 3 * nb); s["jump_t"] = t[1:][jump]
    s["gap_t"] = t[1:][dtt > 0.1]
    # continuity: |second difference| normalised by the segment median
    if len(t) > 3:
        a2 = np.linalg.norm(s["p"][2:] - 2 * s["p"][1:-1] + s["p"][:-2], axis=1); s["acc_z"] = a2 / (np.median(a2) + 1e-9); s["acc_zt"] = t[1:-1]
    else: s["acc_z"] = np.zeros(0); s["acc_zt"] = np.zeros(0, np.int64)
    ls = np.log(np.clip(s["sim3"], 1e-4, None)); s["dls"] = np.abs(np.diff(ls)); s["dls_t"] = t[1:]
    # IMU accelerometer plausibility (soft): SLAM camera acceleration vs IMU linear acceleration in the SLAM world
    s["acc_ok"] = False
    if (t[-1] - t[0]) > 3e9:
        g = np.arange(t[0], t[-1], int(1e9 / 30)); P = np.stack([np.interp(g, t, s["p"][:, a]) for a in range(3)], 1)
        sig = 0.1 * 30; Ps = gaussian_filter1d(P, sig, axis=0, mode="nearest"); aS = np.gradient(np.gradient(Ps, 1 / 30, axis=0), 1 / 30, axis=0)
        ti = D["imu_t"]; dti = np.median(np.diff(ti)) / 1e9; fa = gaussian_filter1d(D["acc"], 0.1 / dti, axis=0, mode="nearest")
        f = np.stack([np.interp(g, ti, fa[:, a]) for a in range(3)], 1); Rw = s["Rs"](g / 1e9) * RX; fW = Rw.apply(f)
        aI = fW - fW.mean(0); e = int(0.3 * 30)                         # gravity in the segment world = mean specific force
        g, aS, aI = g[e:-e], aS[e:-e], aI[e:-e]
        if len(g) > 30:
            s["acc_g"] = g; s["aS"] = aS; s["aI"] = aI; s["s_seg"] = float((aS * aI).sum() / max((aS * aS).sum(), 1e-12)); s["acc_ok"] = True
    return s


def overlap_k(sp, so, RX, cache, key):
    if key in cache: return cache[key]
    a, b = max(sp["t0"], so["t0"]), min(sp["t1"], so["t1"]); k = np.nan
    if b - a > 1.2e9:
        ta = np.arange(a, b - int(0.8e9), int(0.1e9)); tb = ta + int(0.8e9)
        dp = disp_cam(sp, ta, tb); do = disp_cam(so, ta, tb); den = (do * do).sum()
        if den > 0: k = float((dp * do).sum() / den)
    cache[key] = k; return k


def pos(s, t):
    return np.stack([np.interp(t, s["t"], s["p"][:, a]) for a in range(3)], 1)


def disp_cam(s, ta, tb):
    """camera displacement ta->tb in the camera frame at ta (segment units)"""
    return s["Rs"](ta / 1e9).inv().apply(pos(s, tb) - pos(s, ta))


def win_in_seg(s, tc, h):
    ta, tb = tc - h / 2 * 1e9, tc + h / 2 * 1e9
    if s["t1"] - s["t0"] < h * 1e9: return None
    if ta < s["t0"]: ta, tb = s["t0"], s["t0"] + h * 1e9
    if tb > s["t1"]: ta, tb = s["t1"] - h * 1e9, s["t1"]
    return int(ta), int(tb)


def rot_res(D, s, ta, tb, RX, with_mag=False):
    A = s["Rs"](np.array([ta, tb]) / 1e9); M = RX.inv() * (A[0].inv() * A[1]) * RX; B = gyro_rel(D, np.array([ta]), np.array([tb]))[0]
    e = float(np.degrees((M.inv() * B).magnitude()))
    return (e, float(np.degrees(B.magnitude()))) if with_mag else e


def evaluate_arm(take, arm, demos=None):
    D = FD.load_arm(take, arm, with_slam=False); segs, avail = load_segs_raw(take, D["cam"])
    pq = Path(__file__).parent / f"out/RX_pooled_{D['cam']}.npy"
    cal = calibrate(D, segs, R.from_quat(np.load(pq)) if pq.exists() else None); RX = cal["RX_est"]
    for s in segs: prep_seg(D, s, RX)
    D["tX_val"] = FD.handeye_1022(D["cam"])[1]; val_prep(D, segs, RX)
    rows = []; cache = {}
    for W in FD.trims()[take]:
        if demos is not None and W["demo"] not in demos: continue
        tn = np.arange(W["w0"], W["w1"], int(0.05e9))
        cov = np.zeros(len(tn), bool)
        for a0, a1, _ in avail: cov |= (tn >= a0) & (tn <= a1)
        if cov.mean() < 0.95: continue                     # window not processed yet
        for t in tn:
            r = dict(take=take, arm=arm, demo=W["demo"], t=int(t))
            cand = [i for i, s in enumerate(segs) if s["t0"] <= t <= s["t1"]]
            r["n_cover"] = len(cand)
            if not cand:
                r["reason_track"] = "no_segment"; rows.append(r); continue
            pi = max(cand, key=lambda i: min(t - segs[i]["t0"], segs[i]["t1"] - t)); s = segs[pi]; r["seg"] = pi
            r["jump"] = bool(np.any(np.abs(s["jump_t"] - t) < 0.1e9)) or bool(np.any(np.abs(s["gap_t"] - t) < 0.1e9))
            r["seg_len_s"] = (s["t1"] - s["t0"]) / 1e9; r["edge_s"] = min(t - s["t0"], s["t1"] - t) / 1e9
            for h in HZ:
                w = win_in_seg(s, t, h)
                r[f"e{h}"], r[f"g{h}"] = rot_res(D, s, w[0], w[1], RX, True) if w else (np.nan, np.nan)
                if h in (0.4, 1.6): r[f"eH{h}"] = rot_res(D, s, w[0], w[1], cal["RX_harness"]) if w else np.nan
            if t + 1.6e9 <= s["t1"]:                       # forward action-chunk window [t, t+1.6 s] and its halves
                r["f1.6"], r["fg1.6"] = rot_res(D, s, t, t + int(1.6e9), RX, True)
                r["f0.8a"], r["fg0.8a"] = rot_res(D, s, t, t + int(0.8e9), RX, True); r["f0.8b"], r["fg0.8b"] = rot_res(D, s, t + int(0.8e9), t + int(1.6e9), RX, True)
            m = np.abs(s["acc_zt"] - t) <= 0.25e9; r["cont_z"] = float(s["acc_z"][m].max()) if m.any() else np.nan
            m = np.abs(s["dls_t"] - t) <= 0.25e9; r["dlogsim3"] = float(s["dls"][m].max()) if m.any() else np.nan
            # overlap (independent re-tracking of the same frames by another segment): relative displacement discrepancy over 0.8 s
            ov = []
            for j in cand:
                if j == pi: continue
                o = segs[j]; a, b = max(s["t0"], o["t0"]), min(s["t1"], o["t1"])
                if b - a < 0.8e9: continue
                k = overlap_k(s, o, RX, cache, (pi, j))
                if not np.isfinite(k): continue
                ta = int(np.clip(t - 0.4e9, a, b - 0.8e9)); tb = ta + int(0.8e9)
                dp = disp_cam(s, np.array([ta]), np.array([tb]))[0]; do = disp_cam(o, np.array([ta]), np.array([tb]))[0]
                floor = 0.3 * s["v_med"] * 0.8
                ov.append(np.linalg.norm(dp - k * do) / max(np.linalg.norm(dp), floor, 1e-9))
            r["ov_rel"] = float(np.min(ov)) if ov else np.nan
            if s["acc_ok"]:
                m = np.abs(s["acc_g"] - t) <= 0.8e9
                if m.sum() > 10:
                    aS, aI = s["aS"][m], s["aI"][m]
                    r["acc_exc"] = float(np.sqrt((aI ** 2).sum(1).mean())); r["acc_cos"] = float((aS * aI).sum() / np.sqrt((aS ** 2).sum() * (aI ** 2).sum() + 1e-12))
                    sl = (aS * aI).sum() / max((aS * aS).sum(), 1e-12); r["acc_sratio"] = float(sl / s["s_seg"]) if s["s_seg"] > 0 else np.nan
            val_chunk(D, segs, pi, t, RX, r)
            rows.append(r)
    return pd.DataFrame(rows), cal, D, segs


# ---------------------------------------------------------------- marker side check ONLY (held-out validation; never used by the confidence)
K32 = int(1.6e9)


def tag_frame(D, t, tol=25e6):
    i = int(np.clip(np.searchsorted(D["t"], t), 1, len(D["t"]) - 1)); i = i if abs(D["t"][i] - t) < abs(D["t"][i - 1] - t) else i - 1
    return i if abs(D["t"][i] - t) <= tol and D["ntags"][i] >= 2 and D["n_tagrot"][i] >= 2 else -1


def chunk_xly(D, s, RX, ta, tb):
    """SLAM chunk in the TCP frame at ta: x = R_tcp^T dC (segment units), l = metric lever term"""
    ta, tb = np.clip([ta, tb], s["t"][0], s["t"][-1]); Ra, Rb = s["Rs"](np.array([ta, tb]) / 1e9); Rt = Ra * RX
    x = Rt.inv().apply(pos(s, np.array([tb]))[0] - pos(s, np.array([ta]))[0]); l = Rt.inv().apply(Rb.apply(D["tX_val"]) - Ra.apply(D["tX_val"]))
    return x, l, RX.inv() * (Ra.inv() * Rb) * RX


def val_prep(D, segs, RX):
    """per segment: tag-referenced 0.8 s pairs for the held-out scale"""
    ok = (D["ntags"] >= 2) & (D["n_tagrot"] >= 2); t = D["t"]
    for s in segs:
        idx = np.where(ok & (t >= s["t0"]) & (t <= s["t1"] - 0.8e9))[0]; P = []
        for i in idx[::2]:
            j = tag_frame(D, t[i] + 0.8e9, 40e6)
            if j < 0 or t[j] > s["t1"]: continue
            x, l, _ = chunk_xly(D, s, RX, t[i], t[j]); y = R.from_matrix(D["R_tag"][i]).inv().apply(D["p_tag"][j] - D["p_tag"][i])
            P.append((t[i], t[j], x, l, y))
        s["vp"] = P


def val_chunk(D, segs, pi, t, RX, r):
    s = segs[pi]; tb = t + K32
    if tb > s["t1"]: return
    i, j = tag_frame(D, t), tag_frame(D, tb)
    if i < 0 or j < 0: return
    P = [p for p in s["vp"] if p[1] < t - 3e9 or p[0] > tb + 3e9]
    x, l, M = chunk_xly(D, s, RX, D["t"][i], D["t"][j])
    Ri = R.from_matrix(D["R_tag"][i]); y = Ri.inv().apply(D["p_tag"][j] - D["p_tag"][i]); Rref = Ri.inv() * R.from_matrix(D["R_tag"][j])
    r["v_disp_mm"] = float(np.linalg.norm(y) * 1000); r["v_rot_deg"] = float(np.degrees((M.inv() * Rref).magnitude()))
    r["v_rot_gyro_deg"] = float(np.degrees((gyro_rel(D, np.array([D["t"][i]]), np.array([D["t"][j]]))[0].inv() * Rref).magnitude()))
    xx = float(x @ x)
    if s.get("acc_ok") and s["s_seg"] > 0: r["v_trans_imuscale_mm"] = float(np.linalg.norm(s["s_seg"] * x + l - y) * 1000)   # diagnostic only
    if xx > 0:
        so = float(x @ (y - l)) / xx; r["v_trans_oracle_mm"] = float(np.linalg.norm(so * x + l - y) * 1000)
    P1 = [p for p in s["vp"] if p[1] < t - 1e9 or p[0] > tb + 1e9]
    if len(P1) >= 5:
        X = np.array([p[2] for p in P1]); Y = np.array([p[4] - p[3] for p in P1]); sh = float((X * Y).sum() / max((X * X).sum(), 1e-12))
        if sh > 0: r["v_trans_m1_mm"] = float(np.linalg.norm(sh * x + l - y) * 1000)
    if len(P) >= 8:
        X = np.array([p[2] for p in P]); Y = np.array([p[4] - p[3] for p in P]); sh = float((X * Y).sum() / max((X * X).sum(), 1e-12))
        if sh > 0:
            r["v_scale_n"] = len(P); r["v_trans_mm"] = float(np.linalg.norm(sh * x + l - y) * 1000)
