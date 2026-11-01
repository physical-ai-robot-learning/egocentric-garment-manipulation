"""10-23 UMI pose smoother (method c) + baselines (a) current fused, (b) endpoint-anchored SLAM bridge.
Two-stage factor graph on a 20 Hz node grid per trimmed demo window and arm:
  stage 1 ROTATION  : nodes R_n = R0_n Exp(d_n), gyro bias db (const). Factors: gyro relative rotation (strong, L2-ish), raw AprilTag PnP rotation
                      (soft, 1 tag weaker, Cauchy), MASt3R relative rotation over ~0.5 s (soft, Cauchy).
  stage 2 POSITION  : nodes P_n, MASt3R scale per segment chunk (3 s, random walk between chunks). Factors: tag position (soft, Cauchy, 1 tag weaker),
                      MASt3R TCP displacement between consecutive wrist frames = s_chunk * R_B_cam(t_i) dC_i + lever term (Cauchy) -> all segments act on ONE
                      trajectory (no segment jumps), weak acceleration prior (sigma_acc large so fast grasps are not blunted), optional weak accel factor.
                      Linear in (P, s) -> IRLS with sparse normal equations; posterior position std from the inverse information matrix.
All times on the orbbec clock (SLAM -20 ms, IMU host+calibrated offset as tag_rig_fuse)."""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R, Slerp

DT = 0.05
OPT = dict(sig_gyro_deg=0.15, sig_tagrot2_deg=2.0, sig_tagrot1_deg=6.0, sig_slamrot_deg=1.5, slam_rot=True,
           sig_tag2=0.004, sig_tag1=0.008, sig_slam=0.003, slam_pos=True, sig_acc_prior=30.0, chunk_s=2.0, sig_scale_rw=0.05, sig_rot_rw_deg=2.0,
           imu_acc=False, sig_acc=2.0, cauchy=3.0, tag_lag_ms=0.0, max_nfev=200, gn_iters=15)


def prep(D):
    """cumulative gyro rotation Q(t) on the IMU samples (TCP frame), bias = median over still samples"""
    if "Q" in D: return D
    ti = D["imu_t"]; g = D["gyro"]; still = np.linalg.norm(g, axis=1) < np.radians(3)
    D["b0"] = np.median(g[still], 0) if still.sum() > 200 else np.zeros(3)
    dt = np.diff(ti) / 1e9; inc = R.from_rotvec((g[:-1] - D["b0"]) * dt[:, None])
    q = np.empty((len(ti), 4)); cur = R.identity(); q[0] = cur.as_quat()
    for k in range(len(inc)): cur = cur * inc[k]; q[k + 1] = cur.as_quat()
    D["Q"] = R.from_quat(q); D["Qs"] = Slerp(ti / 1e9, D["Q"])
    return D


def Qat(D, t):
    ti = D["imu_t"]; return D["Qs"](np.clip(t, ti[0], ti[-1]) / 1e9)


def node_of(tn, t):
    return np.clip(np.round((t - tn[0]) / (DT * 1e9)).astype(int), 0, len(tn) - 1)


def rot_at(D, tn, Rn, t):
    n = node_of(tn, t); return Rn[n] * (Qat(D, tn[n]).inv() * Qat(D, t))


def slam_in(D, w0, w1):
    out = []
    for k, sg in enumerate(D["segs"]):
        m = (sg["t"] >= w0) & (sg["t"] <= w1)
        if m.sum() >= 3: out.append(dict(k=k, t=sg["t"][m], p=sg["p"][m], R=sg["R"][m], t0=sg["t"][0]))
    return out


# ------------------------------------------------------------------ stage 1: rotation
def solve_rot(D, tn, hide, o):
    N = len(tn); R0 = Slerp(D["t"] / 1e9, R.from_matrix(D["R_fused"]))(np.clip(tn, D["t"][0], D["t"][-1]) / 1e9)
    G = Qat(D, tn[:-1]).inv() * Qat(D, tn[1:]); dtn = np.diff(tn) / 1e9
    # tag rotation frames in window, not hidden
    t = D["t"]; m = (t >= tn[0]) & (t <= tn[-1]) & (D["n_tagrot"] > 0) & ~hide_mask(t, hide)
    tf = t[m]; Rt = R.from_matrix(D["R_tag"][m]); nf = node_of(tn, tf); Cf = Qat(D, tn[nf]).inv() * Qat(D, tf)
    sf = np.radians(np.where(D["n_tagrot"][m] >= 2, o["sig_tagrot2_deg"], o["sig_tagrot1_deg"]))
    # SLAM relative rotation pairs (~0.5 s apart)
    pi, pj, Mrel = [], [], []
    if o["slam_rot"] and "RX" in D:
        for sg in slam_in(D, tn[0], tn[-1]):
            k = 15
            if len(sg["t"]) <= k: continue
            a = np.arange(0, len(sg["t"]) - k, 5); b = a + k
            pi.append(sg["t"][a]); pj.append(sg["t"][b]); Mrel.append(D["RX"].inv() * (sg["R"][a].inv() * sg["R"][b]) * D["RX"])
    if pi:
        pi = np.concatenate(pi); pj = np.concatenate(pj); Mrel = R.concatenate(Mrel); ni = node_of(tn, pi); nj = node_of(tn, pj)
        Ci = Qat(D, tn[ni]).inv() * Qat(D, pi); Cj = Qat(D, tn[nj]).inv() * Qat(D, pj)
    else: ni = nj = np.zeros(0, int)
    sg_ = np.radians(o["sig_gyro_deg"]); ss = np.radians(o["sig_slamrot_deg"])

    def fun(x):
        d = x[:3 * N].reshape(N, 3); db = x[3 * N:]; Rn = R0 * R.from_rotvec(d)
        e1 = ((G.inv() * (Rn[:-1].inv() * Rn[1:])).as_rotvec() + db[None] * dtn[:, None]) / sg_
        e2 = (Rt.inv() * (Rn[nf] * Cf)).as_rotvec() / sf[:, None] if len(nf) else np.zeros((0, 3))
        e3 = (Mrel.inv() * ((Rn[ni] * Ci).inv() * (Rn[nj] * Cj))).as_rotvec() / ss if len(ni) else np.zeros((0, 3))
        return np.concatenate([e1.ravel(), e2.ravel(), e3.ravel(), x[3 * N:] / np.radians(0.5)])
    rows, cols = [], []
    def add(r0, nodes):
        for n in nodes:
            for a in range(3):
                rows.extend([r0 + a] * 3); cols.extend([3 * n + 0, 3 * n + 1, 3 * n + 2])
    r = 0
    for n in range(N - 1):
        add(r, [n, n + 1]);
        for a in range(3): rows.extend([r + a] * 3); cols.extend([3 * N, 3 * N + 1, 3 * N + 2])
        r += 3
    for n in nf: add(r, [n]); r += 3
    for a_, b_ in zip(ni, nj): add(r, [a_, b_] if a_ != b_ else [a_]); r += 3
    for a in range(3): rows.append(r + a); cols.append(3 * N + a)
    r += 3
    S = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(r, 3 * N + 3)).tocsr(); S.data[:] = 1
    sol = least_squares(fun, np.zeros(3 * N + 3), jac_sparsity=S, loss="cauchy", f_scale=o["cauchy"], method="trf", x_scale=1.0, max_nfev=30, tr_solver="lsmr")
    Rn = R0 * R.from_rotvec(sol.x[:3 * N].reshape(N, 3))
    return Rn, dict(rot_cost=float(sol.cost), n_tagrot=int(len(nf)), n_slamrot=int(len(ni)))


def hide_mask(t, hide):
    if hide is None: return np.zeros(len(t), bool)
    return (t >= hide[0]) & (t <= hide[1])


# ------------------------------------------------------------------ stage 2: position (Gauss-Newton, per-chunk Sim3 of each SLAM segment)
def _skew(v):
    z = np.zeros(len(v)); return np.stack([np.stack([z, -v[:, 2], v[:, 1]], 1), np.stack([v[:, 2], z, -v[:, 0]], 1), np.stack([-v[:, 1], v[:, 0], z], 1)], 1)


def solve_pos(D, tn, Rn, hide, o, want_cov=True):
    """x = [P_n (3N) | per SLAM chunk c: R_c (left increment), log s_c]. SLAM factor: P(t_j) - P(t_i) - lever = s_c R_c (pc_j - pc_i)
    (R_c, s_c free per chunk_s chunk, random walk between chunks of one segment -> SLAM translation SHAPE, one global trajectory, no jumps).
    Solved by damped Gauss-Newton with sparse direct solves; phase 1 plain L2, phase 2 Cauchy IRLS (tags + SLAM)."""
    N = len(tn); dtn = DT * 1e9
    def P_at(t):
        f = np.clip((t - tn[0]) / dtn, 0, N - 1 - 1e-9); i = np.floor(f).astype(int); return i, f - i
    t = D["t"] + int(o["tag_lag_ms"] * 1e6); msk = (t >= tn[0]) & (t <= tn[-1]) & (D["ntags"] > 0) & ~hide_mask(D["t"], hide)
    ti_, ta_ = P_at(t[msk]); pt = D["p_tag"][msk]; s_t = np.where(D["ntags"][msk] >= 2, o["sig_tag2"], o["sig_tag1"])
    tt = t[msk]
    P0 = np.stack([np.interp(tn, tt, pt[:, a]) for a in range(3)], 1) if len(tt) >= 2 else np.repeat(D["p_tag"][np.argmin(np.abs(D["t"] - tn[0]))][None], N, 0)
    I0, A0, I1, A1, U, Wl, CH, rw = [], [], [], [], [], [], [], []; C = 0; Rinit, ls0 = [], []
    if o["slam_pos"] and "RX" in D:
        RXi = D["RX"].inv(); tX = D["tX"]
        for sg in slam_in(D, tn[0], tn[-1]):
            ts = sg["t"]; ok = np.where(np.diff(ts) < 0.1e9)[0]
            if len(ok) < 3: continue
            Rbc = rot_at(D, tn, Rn, ts) * RXi
            u = sg["p"][1:] - sg["p"][:-1]; w = Rbc[1:].apply(tX) - Rbc[:-1].apply(tX)
            ch = ((ts[:-1] - sg["t0"]) // int(o["chunk_s"] * 1e9)).astype(int); uch = np.unique(ch[ok]); cmap = {c: C + j for j, c in enumerate(uch)}
            for c in uch:
                kk = ok[ch[ok] == c]; Rinit.append((Rbc[kk] * sg["R"][kk].inv()).mean().as_matrix())
                m2 = msk & (t >= ts[kk[0]]) & (t <= ts[kk[-1] + 1]); sl = np.linalg.norm(u[kk], axis=1).sum()
                tp = np.linalg.norm(np.diff(D["p_tag"][m2], axis=0), axis=1).sum() if m2.sum() > 3 else 0
                ls0.append(np.log(np.clip(tp / sl, 0.03, 2.0)) if (sl > 0 and tp > 0.02) else np.log(0.2))
            rw += [(cmap[a_], cmap[b_]) for a_, b_ in zip(uch[:-1], uch[1:])]
            i0, a0 = P_at(ts[:-1][ok]); i1, a1 = P_at(ts[1:][ok])
            I0.append(i0); A0.append(a0); I1.append(i1); A1.append(a1); U.append(u[ok]); Wl.append(w[ok]); CH.append(np.array([cmap[c] for c in ch[ok]])); C += len(uch)
    if I0: I0, A0, I1, A1, U, Wl, CH = map(np.concatenate, (I0, A0, I1, A1, U, Wl, CH))
    else: I0 = I1 = CH = np.zeros(0, int); A0 = A1 = np.zeros(0); U = Wl = np.zeros((0, 3))
    rw = np.array(rw, int).reshape(-1, 2); ls0 = np.array(ls0); Rc = np.array(Rinit).reshape(-1, 3, 3); ls = ls0.copy(); P = P0.copy()
    nv = 3 * N + 4 * C; sq = o["sig_slam"]; sa = o["sig_acc_prior"] * DT * DT; sr = np.radians(o["sig_rot_rw_deg"]); ss = o["sig_scale_rw"]
    oR, oL = 3 * N, 3 * N + 3 * C; ax = np.arange(3)
    # constant parts of the Jacobian (P columns)
    nT, nS, nA = len(ti_), len(I0), max(N - 2, 0)
    rT = 3 * np.arange(nT)[:, None] + ax                                  # rows (nT,3)
    rS = 3 * nT + 3 * np.arange(nS)[:, None] + ax
    rA = 3 * (nT + nS) + 3 * np.arange(nA)[:, None] + ax
    rRW = 3 * (nT + nS + nA) + 3 * np.arange(len(rw))[:, None] + ax
    rLW = 3 * (nT + nS + nA + len(rw)) + np.arange(len(rw)); rLP = rLW[-1] + 1 + np.arange(C) if len(rw) else 3 * (nT + nS + nA) + np.arange(C)
    nr = 3 * (nT + nS + nA + len(rw)) + len(rw) + C
    def jac_res(P, Rc, ls):
        R_ = []; C_ = []; V_ = []
        def add(r, c, v): R_.append(np.broadcast_to(r, np.shape(v)).ravel()); C_.append(np.broadcast_to(c, np.shape(v)).ravel()); V_.append(np.ravel(v))
        e1 = ((1 - ta_)[:, None] * P[ti_] + ta_[:, None] * P[ti_ + 1] - pt) / s_t[:, None]
        add(rT, 3 * ti_[:, None] + ax, ((1 - ta_) / s_t)[:, None] * np.ones(3)); add(rT, 3 * (ti_ + 1)[:, None] + ax, (ta_ / s_t)[:, None] * np.ones(3))
        if nS:
            Ru = np.einsum("nij,nj->ni", Rc[CH], U); sc = np.exp(ls[CH])[:, None]
            Pj = (1 - A1)[:, None] * P[I1] + A1[:, None] * P[I1 + 1]; Pi = (1 - A0)[:, None] * P[I0] + A0[:, None] * P[I0 + 1]
            e2 = (Pj - Pi - Wl - sc * Ru) / sq
            for idx, coef in ((I1, (1 - A1)), (I1 + 1, A1), (I0, -(1 - A0)), (I0 + 1, -A0)):
                add(rS, 3 * idx[:, None] + ax, (coef / sq)[:, None] * np.ones(3))
            Jd = sc[:, :, None] * _skew(Ru) / sq                                # d e2 / d delta_c (left increment)
            add(rS[:, :, None], (oR + 3 * CH)[:, None, None] + ax[None, None, :], Jd)
            add(rS, (oL + CH)[:, None] * np.ones(3, int), -sc * Ru / sq)
        else: e2 = np.zeros((0, 3))
        e3 = (P[2:] - 2 * P[1:-1] + P[:-2]) / sa; n_ = np.arange(1, N - 1)
        for off, coef in ((-1, 1.0), (0, -2.0), (1, 1.0)): add(rA, 3 * (n_ + off)[:, None] + ax, np.full((nA, 3), coef / sa))
        if len(rw):
            e4 = R.from_matrix(np.einsum("nij,nkj->nik", Rc[rw[:, 1]], Rc[rw[:, 0]])).as_rotvec() / sr
            add(rRW, (oR + 3 * rw[:, 1])[:, None] + ax, np.full((len(rw), 3), 1 / sr)); add(rRW, (oR + 3 * rw[:, 0])[:, None] + ax, np.full((len(rw), 3), -1 / sr))
            e5 = (ls[rw[:, 1]] - ls[rw[:, 0]]) / ss; add(rLW, oL + rw[:, 1], np.full(len(rw), 1 / ss)); add(rLW, oL + rw[:, 0], np.full(len(rw), -1 / ss))
        else: e4 = np.zeros((0, 3)); e5 = np.zeros(0)
        e6 = (ls - ls0) / 1.5; add(rLP, oL + np.arange(C), np.full(C, 1 / 1.5))
        r = np.concatenate([e1.ravel(), e2.ravel(), e3.ravel(), e4.ravel(), e5, e6])
        J = sp.csr_matrix((np.concatenate(V_), (np.concatenate(R_), np.concatenate(C_))), shape=(nr, nv))
        return r, J
    rob = np.zeros(nr, bool); rob[:3 * (nT + nS)] = True
    def weights(r):
        w = np.ones(nr); b = r[:3 * (nT + nS)].reshape(-1, 3); e = np.linalg.norm(b, axis=1); w[:3 * (nT + nS)] = np.repeat(1.0 / (1.0 + (e / o["cauchy"]) ** 2), 3); return w
    lam = 1e-5; w = np.ones(nr); cost = None
    for it in range(o["gn_iters"]):
        r, J = jac_res(P, Rc, ls)
        if it >= o["gn_iters"] // 3: w = weights(r)
        H = (J.T @ sp.diags(w) @ J).tocsc(); g = J.T @ (w * r)
        dx = None
        for lm in (lam, 1e-3, 1e-1):
            try:
                dx = -splu((H + sp.identity(nv) * lm).tocsc()).solve(g)
                if np.all(np.isfinite(dx)): break
            except RuntimeError: dx = None
        if dx is None or not np.all(np.isfinite(dx)): break
        dx[oL:] = np.clip(dx[oL:], -0.5, 0.5); dx[oR:oL] = np.clip(dx[oR:oL], -0.3, 0.3)
        P = P + dx[:3 * N].reshape(N, 3)
        if C:
            Rc = R.from_matrix(np.einsum("nij,njk->nik", R.from_rotvec(dx[oR:oL].reshape(C, 3)).as_matrix(), Rc)).as_matrix(); ls = np.clip(ls + dx[oL:], np.log(0.01), np.log(3.0))
        if np.abs(dx[:3 * N]).max() < 1e-5: break
    r, J = jac_res(P, Rc, ls); w = weights(r); std = None
    if want_cov:
        H = (J.T @ sp.diags(w) @ J).toarray() + np.eye(nv) * 1e-9; Cv = np.linalg.inv(H)
        std = np.sqrt(np.maximum(np.diag(Cv)[:3 * N].reshape(N, 3).sum(1), 0))
    nt3, ns3 = 3 * nT, 3 * nS
    info = dict(n_tag=int(nT), n_slam=int(nS), nchunk=int(C), scales=np.exp(ls), tag_w_med=float(np.median(w[:nt3])) if nt3 else 1.0,
                slam_w_med=float(np.median(w[nt3:nt3 + ns3])) if ns3 else 1.0, nfev=it + 1)
    cov = np.zeros(N, bool)
    if ns3:
        ws = w[nt3:nt3 + ns3].reshape(-1, 3).min(1) > 0.5; cov[I0[ws]] = True; cov[np.minimum(I1[ws] + 1, N - 1)] = True
    info["slam_cov"] = cov
    return P, std, info


def smooth(D, w0, w1, hide=None, o=None, want_cov=True):
    o = dict(OPT, **(o or {})); prep(D)
    tn = np.arange(w0, w1 + 1, int(DT * 1e9), dtype=np.int64)
    Rn, i1 = solve_rot(D, tn, hide, o)
    Pn, std, i2 = solve_pos(D, tn, Rn, hide, o, want_cov)
    return dict(tn=tn, Rn=Rn, Pn=Pn, std=std, info={**i1, **i2})


def eval_at(D, S, t):
    tn = S["tn"]; P = np.stack([np.interp(t, tn, S["Pn"][:, a]) for a in range(3)], 1)
    Rr = rot_at(D, tn, S["Rn"], t); sd = None if S["std"] is None else np.interp(t, tn, S["std"])
    return P, Rr, sd


# ------------------------------------------------------------------ (a) current fused (no new info)
def method_a(D, t_eval, hide):
    """the current file: gated SLAM where slam_ok, else linear interpolation between the remaining fused_valid samples; rotation = stored R_B_tcp"""
    t = D["t"]; hid = hide_mask(t, hide); valid = D["fused_valid"] & ~(hid & ~D["slam_ok"])
    P = np.stack([np.interp(t_eval, t[valid], D["p_fused"][valid, a]) for a in range(3)], 1)
    k = np.searchsorted(t, t_eval); k = np.clip(k, 0, len(t) - 1)
    sl = D["slam_ok"][k] & (t[k] == t_eval); P[sl] = D["p_fused"][k[sl]]
    return P, R.from_matrix(D["R_fused"][k])


# ------------------------------------------------------------------ (b) endpoint-anchored SLAM bridge
def gyro_bridge(D, ta, tb, Ra, Rb, t):
    """rotation in a gap from the gyro, endpoint error distributed by SLERP"""
    prep(D); Qa, Qb = Qat(D, np.array([ta]))[0], Qat(D, np.array([tb]))[0]
    Rg = Ra * (Qa.inv() * Qat(D, t)); Rgb = Ra * (Qa.inv() * Qb); err = (Rgb.inv() * Rb).as_rotvec()
    wgt = np.clip((t - ta) / max(tb - ta, 1), 0, 1); return Rg * R.from_rotvec(wgt[:, None] * err[None])


def method_b(D, t_eval, hide, ctx_s=1.0, max_res=0.02):
    """hidden window [h0,h1]: anchors = tag frames outside. Every raw SLAM segment spanning [h0-ctx, h1+ctx] with >=4 anchors each side:
    rotation RA from the (fused) anchor rotations, scale+translation by LS, residual at both ends removed linearly. Best context residual wins;
    if none (or residual > max_res) -> linear fill. Rotation = gyro bridge between the anchor rotations."""
    t = D["t"]; h0, h1 = hide; anc = (D["ntags"] > 0) & ~hide_mask(t, hide)
    ia = np.where(anc & (t < h0))[0]; ib = np.where(anc & (t > h1))[0]
    ta, tb = t[ia[-1]], t[ib[0]]; pa, pb = D["p_tag"][ia[-1]], D["p_tag"][ib[0]]
    Rf = R.from_matrix(D["R_fused"]); Rr = gyro_bridge(D, ta, tb, Rf[ia[-1]], Rf[ib[0]], t_eval)
    P = pa + (pb - pa) * np.clip((t_eval - ta) / (tb - ta), 0, 1)[:, None]; best = None
    if "RX" in D:
        ca = ia[t[ia] >= ta - ctx_s * 1e9]; cb = ib[t[ib] <= tb + ctx_s * 1e9]
        for sg in D["segs"]:
            if sg["t"][0] > t[ca[0]] or sg["t"][-1] < t[cb[-1]] or len(ca) < 4 or len(cb) < 4: continue
            c = np.r_[ca, cb]; tc = t[c]
            if np.diff(sg["t"][(sg["t"] >= tc[0]) & (sg["t"] <= tc[-1])]).max(initial=0) > 0.3e9: continue
            pc = np.stack([np.interp(tc, sg["t"], sg["p"][:, a]) for a in range(3)], 1); Rc = Slerp(sg["t"] / 1e9, sg["R"])(tc / 1e9)
            RA = (Rf[c] * D["RX"].inv() * Rc.inv()).mean()
            y = D["p_tag"][c] - (RA * Rc).apply(D["tX"]); xx = RA.apply(pc)
            Am = np.zeros((3 * len(c), 4)); Am[:, 0] = xx.ravel(); Am[0::3, 1] = 1; Am[1::3, 2] = 1; Am[2::3, 3] = 1
            sol = np.linalg.lstsq(Am, y.ravel(), rcond=None)[0]; s, tt = sol[0], sol[1:]
            if s <= 0: continue
            pred = s * xx + (RA * Rc).apply(D["tX"]) + tt; r = pred - D["p_tag"][c]; rms = float(np.sqrt(np.mean(np.sum(r ** 2, 1))))
            if best is None or rms < best[0]: best = (rms, sg, RA, s, tt, r, c)
        if best is not None and best[0] < max_res:
            rms, sg, RA, s, tt, r, c = best; m = (t_eval >= sg["t"][0]) & (t_eval <= sg["t"][-1])
            pe = np.stack([np.interp(t_eval, sg["t"], sg["p"][:, a]) for a in range(3)], 1); Re = Slerp(sg["t"] / 1e9, sg["R"])(np.clip(t_eval, sg["t"][0], sg["t"][-1]) / 1e9)
            pred = s * RA.apply(pe) + (RA * Re).apply(D["tX"]) + tt
            ea = r[np.isin(c, ia)][-3:].mean(0); eb = r[np.isin(c, ib)][:3].mean(0); wgt = np.clip((t_eval - ta) / (tb - ta), 0, 1)[:, None]
            Pb = pred - ((1 - wgt) * ea + wgt * eb); P[m] = Pb[m]
    return P, Rr, (best[0] if best is not None else None)


# ------------------------------------------------------------------ (a') current pipeline, leak-free: whole-segment Sim3 (hand-eye fixed) on the
# non-hidden tags, gated like chain_v1 (own-fit p90 <= 30 mm, max dev from the tag baseline <= 0.15 m); else linear fill
def method_a_clean(D, t_eval, hide, p90_mm=30.0):
    t = D["t"]; anc = (D["ntags"] > 0) & ~hide_mask(t, hide)
    P = np.stack([np.interp(t_eval, t[anc], D["p_tag"][anc, a]) for a in range(3)], 1); Rf = R.from_matrix(D["R_fused"])
    k = np.clip(np.searchsorted(t, t_eval), 0, len(t) - 1); best = None
    if "RX" in D:
        for sg in D["segs"]:
            ev = (t_eval >= sg["t"][0]) & (t_eval <= sg["t"][-1])
            if not ev.any(): continue
            c = np.where(anc & (t >= sg["t"][0]) & (t <= sg["t"][-1]))[0]
            if len(c) < 8: continue
            pc = np.stack([np.interp(t[c], sg["t"], sg["p"][:, a]) for a in range(3)], 1); Rc = Slerp(sg["t"] / 1e9, sg["R"])(t[c] / 1e9)
            RA = (Rf[c] * D["RX"].inv() * Rc.inv()).mean(); y = D["p_tag"][c] - (RA * Rc).apply(D["tX"]); xx = RA.apply(pc)
            Am = np.zeros((3 * len(c), 4)); Am[:, 0] = xx.ravel(); Am[0::3, 1] = 1; Am[1::3, 2] = 1; Am[2::3, 3] = 1
            sol = np.linalg.lstsq(Am, y.ravel(), rcond=None)[0]; s, tt = sol[0], sol[1:]
            if not (0.05 < s < 3.0): continue
            r = np.linalg.norm((Am @ sol - y.ravel()).reshape(-1, 3), axis=1)
            if np.percentile(r, 90) * 1000 > p90_mm: continue
            te = t_eval[ev]; pe = np.stack([np.interp(te, sg["t"], sg["p"][:, a]) for a in range(3)], 1); Re = Slerp(sg["t"] / 1e9, sg["R"])(te / 1e9)
            pred = s * RA.apply(pe) + (RA * Re).apply(D["tX"]) + tt
            if np.linalg.norm(pred - P[ev], axis=1).max() > 0.15: continue
            if best is None or len(c) > best[0]: best = (len(c), ev, pred)
    if best is not None: P[best[1]] = best[2]
    return P, Rf[k], best is not None


def oracle_slam(D, t_eval):
    """CHEATING upper bound: best single raw segment, Sim3 fitted directly on the evaluated (hidden) tag frames"""
    best = None; y = D["p_tag"][np.searchsorted(D["t"], t_eval)]
    for sg in D["segs"]:
        if sg["t"][0] > t_eval[0] or sg["t"][-1] < t_eval[-1] or len(t_eval) < 4: continue
        pc = np.stack([np.interp(t_eval, sg["t"], sg["p"][:, a]) for a in range(3)], 1)
        A = pc - pc.mean(0); B = y - y.mean(0); U, S_, Vt = np.linalg.svd(B.T @ A / len(A)); Dg = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        Rm = U @ Dg @ Vt; s = (S_ * np.diag(Dg)).sum() / A.var(0).sum(); P = s * pc @ Rm.T + (y.mean(0) - s * Rm @ pc.mean(0))
        e = np.linalg.norm(P - y, axis=1)
        if best is None or e.mean() < best[0]: best = (e.mean(), P)
    return None if best is None else best[1]
