"""10-24 CoTracker3 points -> UMI body origin with a point LIFECYCLE (pack v2: initial seeds + reserve candidates every 1 s).
States: RESERVE (tracked by CoTracker, not used) -> PROBATION -> TRUSTED -> (OUTLIER streak) -> DROPPED.
  pose      AHRS rotation fixed; votes of visible TRUSTED points c_i = P_i - R q_i, weight 0.3 + 0.7 min(age/AGE_FULL, 1); weighted median,
            inliers within INL (2 rounds); pose valid if >= MIN_INL inliers and inlier fraction >= MIN_FRAC.  conf HIGH if >= HIGH_INL inliers,
            fraction >= 0.6 and 3D spread >= SPREAD_MIN.
  outliers  a TRUSTED point outside INL for OUT_N consecutive visible frames -> DROPPED (also PROBATION after 2 misses)
  promotion a PROBATION point within INL for PROB_N consecutive visible frames -> TRUSTED (q fixed at activation)
  refill    only while pose is HIGH and (alive TRUSTED < REFILL_FRAC x initial, or spread < SPREAD_MIN, or residual > RES_HI), at most every
            REFILL_GAP_S: up to N_ADD visible RESERVE points whose 3D point lies within BODY_R of the current origin -> PROBATION
  lost      no valid pose: candidate = RESERVE points started within the last 1 s, rigid-model fit around their centroid (fit >= FIT_MIN),
            within min(GATE0 + GATE_V dt, GATE_MAX) of the last pose -> those points become PROBATION with q from that fit; the pose is
            output again once >= MIN_INL of them are TRUSTED (self-consistent for PROB_N frames)
Metrics vs tag GT (fixed per-rig offset) + vs the v31speed (+v4) raw baseline: coverage, longest gap, abs error, far10, e32, longest
wrong burst (> 10 cm, unknown frames joined only between wrong frames), trusted points min/median, refill+reacq count, longest no-trusted gap.
    python solve2.py <pack_dir> [hide_off]   (hide_off 0/1/2: yaw anchor from visible tags only -> solve2_<arm>_h<off>.npz)"""
import json, sys
from pathlib import Path
import cv2, numpy as np
HERE = Path(__file__).resolve().parent
from garment_pipeline.fusion_stack import core as C  # noqa
from garment_pipeline.fusion_stack.orbbec.v2b import v32_common as V  # noqa
from garment_pipeline.fusion_stack import camcal  # noqa
from egogarment import paths as egopaths  # noqa: E402
FD = C.FD
P = dict(INL=0.025, MIN_INL=6, MIN_FRAC=0.4, HIGH_INL=10, SPREAD_MIN=0.012, OUT_N=3, PROB_N=4, AGE_FULL=30, REFILL_FRAC=0.5, RES_HI=0.012,
         REFILL_GAP_S=0.5, N_ADD=15, BODY_R=0.08, FIT_MIN=0.3, GATE0=0.05, GATE_V=1.0, GATE_MAX=0.4)
cal = json.loads((FD.EGO / "configs/calibration/robot_cam_v001.json").read_text()); TBC = np.array(cal["T_B_C"])
OFF = json.loads((HERE.parent / "body_offset_by_rig_v1.json").read_text())["offsets_by_rig"]


def lift_all(depth, TR, I):
    """(T, N, 3) base-frame points from a 3x3 median of depth at every track pixel (NaN where invalid)"""
    T, N, _ = TR.shape; out = np.full((T, N, 3), np.nan)
    for j in range(T):
        d = depth[j]; H, W = d.shape; u = TR[j, :, 0]; v = TR[j, :, 1]; ok = np.isfinite(u) & (u >= 1) & (u < W - 1) & (v >= 1) & (v < H - 1)
        ui, vi = u[ok].astype(int), v[ok].astype(int); st = np.stack([d[vi + dy, ui + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], 1).astype(float)
        st[st <= 0] = np.nan; z = np.nanmedian(st, 1) / 1000 if st.size else np.zeros(0); idx = np.flatnonzero(ok)
        good = np.isfinite(z); idx, z, uu, vv = idx[good], z[good], u[ok][good], v[ok][good]
        out[j, idx] = np.stack([(uu - I["cx"]) / I["fx"] * z, (vv - I["cy"]) / I["fy"] * z, z], 1) @ TBC[:3, :3].T + TBC[:3, 3]
    return out


def wmed(x, w):
    o = np.argsort(x); cw = np.cumsum(w[o]); return x[o[np.searchsorted(cw, cw[-1] / 2)]]


def runs(b, t):
    x = np.r_[0, b.astype(int), 0]; s, e = np.flatnonzero(np.diff(x) == 1), np.flatnonzero(np.diff(x) == -1)
    return [(t[min(e_ - 1, len(t) - 1)] - t[s_]) / 1e9 + 0.067 for s_, e_ in zip(s, e)]


def metrics(c, t, gt, okgt, okgt2):
    v = np.isfinite(c[:, 0]); m = v & okgt; ae = np.linalg.norm(c - gt, axis=1) * 1000
    lab = np.full(len(t), np.nan); lab[m] = (ae[m] > 100).astype(float); w = np.zeros(len(t), bool)
    kn = np.flatnonzero(np.isfinite(lab))
    for a_, b_ in zip(kn[:-1], kn[1:]):
        if lab[a_] == 1 and lab[b_] == 1 and v[a_:b_ + 1].all(): w[a_:b_ + 1] = True
    w[kn[lab[kn] == 1]] = True
    e = []
    for i in range(len(t)):
        j = np.searchsorted(t, t[i] + int(1.6e9))
        if j < len(t) and abs(t[j] - t[i] - 1.6e9) < 70e6 and v[i] and v[j] and okgt2[i] and okgt2[j]: e.append(np.linalg.norm((c[j] - c[i]) - (gt[j] - gt[i])) * 1000)
    return dict(cov=round(float(v.mean()), 3), gap_max_s=round(max(runs(~v, t) or [0.0]), 2), abs_med=round(float(np.nanmedian(ae[m])), 1) if m.any() else np.nan,
                far10=round(float(np.nanmean(ae[m] > 100)), 3) if m.any() else np.nan, wrong_burst_s=round(max(runs(w, t) or [0.0]), 2),
                e32_med=round(float(np.median(e)), 1) if e else np.nan, e32_p90=round(float(np.percentile(e, 90)), 1) if e else np.nan, n_e32=len(e))


def solve_arm(PT, VIS, qstart, kind, Rm, t, j0, T0, M, fit_fn):
    T_, N = VIS.shape; st = np.zeros(N, int)                                  # 0 reserve, 1 probation, 2 trusted, -1 dropped
    q = np.full((N, 3), np.nan); age = np.zeros(N); out_s = np.zeros(N, int); good = np.zeros(N, int)
    seed = np.flatnonzero((kind == 0) & np.isfinite(PT[j0, :, 0])); q[seed] = (PT[j0, seed] - T0) @ Rm[j0]; st[seed] = 2; n_init = len(seed)
    c = np.full((T_, 3), np.nan); conf = np.zeros(T_, int); ntr = np.zeros(T_, int); events = 0; last_refill = -1e18; last_c = T0.copy(); last_t = t[j0]
    for j in range(j0, T_):
        R = Rm[j]
        if not np.isfinite(R[0, 0]): continue
        vis = (VIS[j] > 0.5) & np.isfinite(PT[j, :, 0]); tr = np.flatnonzero((st == 2) & vis); ntr[j] = int((st == 2).sum())
        pose = None
        if len(tr) >= P["MIN_INL"]:
            votes = PT[j, tr] - q[tr] @ R.T; wt = 0.3 + 0.7 * np.minimum(age[tr] / P["AGE_FULL"], 1)
            med = np.array([wmed(votes[:, k], wt) for k in range(3)])
            for _ in range(2):
                inl = np.linalg.norm(votes - med, axis=1) < P["INL"]
                if not inl.any(): break
                med = (votes[inl] * wt[inl, None]).sum(0) / wt[inl].sum()
            if inl.sum() >= P["MIN_INL"] and inl.mean() >= P["MIN_FRAC"]:
                pose = med; res = float(np.median(np.linalg.norm(votes[inl] - med, axis=1))); spread = float(np.sqrt(PT[j, tr[inl]].var(0).sum()))
                conf[j] = 2 if (inl.sum() >= P["HIGH_INL"] and inl.mean() >= 0.6 and spread >= P["SPREAD_MIN"]) else 1
                age[tr[inl]] += 1; out_s[tr[inl]] = 0; out_s[tr[~inl]] += 1; st[tr[~inl][out_s[tr[~inl]] >= P["OUT_N"]]] = -1
        if pose is not None:
            c[j] = pose; last_c, last_t = pose, t[j]
            pr = np.flatnonzero((st == 1) & vis)
            if len(pr):
                ok = np.linalg.norm(PT[j, pr] - q[pr] @ R.T - pose, axis=1) < P["INL"]; good[pr[ok]] += 1; out_s[pr[~ok]] += 1
                st[pr[ok & (good[pr] >= P["PROB_N"])]] = 2; st[pr[~ok & (out_s[pr] >= 2)]] = -1
            alive = int((st == 2).sum())
            if conf[j] == 2 and (alive < P["REFILL_FRAC"] * n_init or spread < P["SPREAD_MIN"] or res > P["RES_HI"]) and t[j] - last_refill >= P["REFILL_GAP_S"] * 1e9:
                rv = np.flatnonzero((st == 0) & vis & (qstart <= j) & (kind == 1))
                rv = rv[np.linalg.norm(PT[j, rv] - pose, axis=1) < P["BODY_R"]][:P["N_ADD"]]
                if len(rv): q[rv] = (PT[j, rv] - pose) @ R; st[rv] = 1; good[rv] = 0; out_s[rv] = 0; last_refill = t[j]; events += 1
        else:                                                                # lost: re-acquire from recent reserve points
            pr = np.flatnonzero((st == 1) & vis)
            if len(pr) and np.isfinite(q[pr, 0]).all():
                votes = PT[j, pr] - q[pr] @ R.T; med = np.median(votes, 0); ok = np.linalg.norm(votes - med, axis=1) < P["INL"]
                good[pr[ok]] += 1; st[pr[~ok]] = -1; st[pr[ok & (good[pr] >= P["PROB_N"])]] = 2
            if (st == 1).sum() < P["MIN_INL"]:
                rv = np.flatnonzero((st == 0) & vis & (qstart <= j) & (qstart >= j - 15) & (kind == 1))
                if len(rv) >= P["MIN_INL"]:
                    cen = np.median(PT[j, rv], 0); fr = fit_fn(j, cen)
                    if fr is not None and fr[2] >= P["FIT_MIN"]:
                        gate = min(P["GATE0"] + P["GATE_V"] * (t[j] - last_t) / 1e9, P["GATE_MAX"])
                        if np.linalg.norm(fr[0] - last_c) <= gate:
                            sel = rv[np.linalg.norm(PT[j, rv] - fr[0], axis=1) < P["BODY_R"]]
                            q[sel] = (PT[j, sel] - fr[0]) @ R; st[sel] = 1; good[sel] = 0; events += 1
    return c, conf, ntr, events


from garment_pipeline.fusion_stack.stitch.wdo import hidden_mask  # noqa
HOFF = int(sys.argv[2]) if len(sys.argv) > 2 else None      # hidden-tag run offset: yaw anchor from visible tags only -> solve2_<arm>_h<off>.npz
for d in sorted(Path(sys.argv[1]).iterdir()):
    if not (d / "tracks.npz").exists(): continue
    meta = json.loads((d / "meta.json").read_text()); take = meta["take"]; TBC = camcal.T_B_C(take); ks = np.array(meta["ks"]); trz = np.load(d / "tracks.npz"); TR, VIS = trz["tracks"], trz["vis"].astype(float)
    tk = FD.TAG / take; I = json.loads((tk / "meta.json").read_text())["intrinsics"]; aq = np.array(meta["arm_of_query"]); kind = np.array(meta["query_kind"]); qs = np.array(meta["queries"])[:, 0].astype(int)
    ref = np.load(HERE.parent / f"out/trackv31speed_{take}.npz"); tfull = ref["t_ns"]; z_tab = float(ref["z_table"]); t = tfull[ks]
    asg = json.loads((egopaths.DERIVED / take / "assignment.json").read_text())
    base_name = "trackv4speed" if (HERE.parent / f"out/trackv4speed_{take}.npz").exists() else "trackv31speed"; base = np.load(HERE.parent / f"out/{base_name}_{take}.npz")
    depth = [cv2.imread(str(tk / f"depth_{k:05d}.png"), -1) for k in ks]; PT = lift_all(depth, TR, I)
    gray = {}
    for a in ("left", "right"):
        Rm = V.arm_rotations(take, tfull, ref[f"c_{a}"], a, anchor_ok=None if HOFF is None else ~hidden_mask(tfull, HOFF))[ks]; sel = np.flatnonzero(aq == a); j0 = meta["seeds"][a]["frame"]
        M = np.load(HERE.parent / f"out/rig_model_{asg['rig_of_arm'][a]}.npz")["pts"]
        def fit_fn(j, cen, a=a, Rm=Rm, M=M):
            if not np.isfinite(Rm[j, 0, 0]): return None
            g = cv2.imread(str(tk / f"color_{ks[j]:05d}.jpg"), cv2.IMREAD_GRAYSCALE); dd = depth[j].astype(np.float32) / 1000; vv, uu = np.mgrid[0:dd.shape[0], 0:dd.shape[1]]
            X = np.stack([(uu - I["cx"]) / I["fx"] * dd, (vv - I["cy"]) / I["fy"] * dd, dd], -1) @ TBC[:3, :3].T + TBC[:3, 3]
            pts = X[(dd > 0.2) & (g < 40) & (X[..., 2] - z_tab > 0.02) & (np.linalg.norm(X - cen, axis=-1) < 0.13)]
            return V.rigid_fit(pts, Rm[j], M, cen) if len(pts) > 40 else None
        c0 = np.array(meta["seeds"][a]["centre"]); f0 = fit_fn(j0, c0); T0 = f0[0] if f0 is not None else np.nanmean(PT[j0, sel[kind[sel] == 0]], 0)
        c, conf, ntr, ev = solve_arm(PT[:, sel], VIS[:, sel], qs[sel], kind[sel], Rm, t, j0, T0, M, fit_fn)
        D = FD.load_arm(take, a, with_slam=False); o = np.array(OFF[asg["rig_of_arm"][a]]) / 1000
        gt = D["p_tag"][ks] + np.einsum("nij,j->ni", Rm, o); okgt = (D["ntags"][ks] >= 1) & np.isfinite(Rm[:, 0, 0]); okgt2 = (D["ntags"][ks] >= 2) & np.isfinite(Rm[:, 0, 0])
        mc = metrics(c, t, gt, okgt, okgt2); mb = metrics(base[f"c_{a}"][ks], t, gt, okgt, okgt2)
        live = ntr[j0:]; notr = runs(ntr[j0:] < P["MIN_INL"], t[j0:])
        mc.update(trusted_min=int(live.min()), trusted_med=int(np.median(live)), events=ev, no_trusted_gap_s=round(max(notr or [0.0]), 2), high=round(float((conf[j0:] == 2).mean()), 2))
        print(f"{d.name} {a[0]} | CT-v2 {mc}\n{'':22s}| base[{base_name}] {mb}", flush=True)
        np.savez(d / (f"solve2_{a}.npz" if HOFF is None else f"solve2_{a}_h{HOFF}.npz"), c=c, conf=conf, ntr=ntr, t=t, gt=gt)
