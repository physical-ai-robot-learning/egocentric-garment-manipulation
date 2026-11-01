"""10-23 whole-demo stitch optimizer (v2 / v3) on EXISTING raw visual segments (MASt3R now; any source with t/p/R per segment, v4 = A2).
Builds on _fgo_1023 (fgo_core.solve_rot for the gyro-led rotation stage). Differences vs fgo_v0:
  * camera<->IMU: marker-free pooled hand-eye RX + per-arm camera time offset + gyro bias from SLAM<->gyro (fgo_v0 used the per-arm tag
    hand-eye, 150 deg / 34-39 deg wrong on 124754 R / 144321 L,R, which also rotated every SLAM displacement into the wrong direction)
  * position: per visual segment ONE Sim(3) (R_i fixed from the gyro-led rotation, s_i free; t_i eliminated by displacement factors), so each
    segment keeps its relative shape (fgo_v0: free rotation + scale random walk per 2 s chunk)          [v2]
    v3: scale knots every KNOT_S inside a segment, strong random-walk smoothness between knots (+ optional weak Cauchy accel factor)
  * segments cut at pose jumps / tracking loss; SLAM steps whose 0.4 s rotation disagrees with the gyro are down-weighted (x0.1)
  * smoothness = second difference (velocity-continuous, Hermite-like across gaps; no straight-line fill), weak where a visual or tag
    measurement touches the node, strong only on no-information nodes
  * tags: soft robust (Cauchy) absolute anchors only; hidden-tag QA removes them entirely (fit + init) in hidden blocks
Linear in (P, s) given the rotations -> sparse IRLS.
    python wdo.py TAKE ARM VARIANT(v2|v3on|v3off) [--offsets 0,1,2,-1] [--demos 0,4]  -> qa/<variant>/<take>_<arm>_d<NN>_h<off>.npz"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.spatial.transform import Rotation as R, Slerp

HERE = Path(__file__).parent
from garment_pipeline.fusion_stack.core import FD, FC, load_segs_raw, calibrate, prep_seg, gyro_rel  # noqa: E402

DT = 0.05; KNOT_S = 1.5
SIG = dict(abs_prior=1e6, slam=0.003, tag2=0.004, tag1=0.008, acc_info=30.0, acc_gap=3.0, scale_prior=0.5, knot_rw=0.05, imu_acc=2.0, cauchy=3.0, rot_bad_w=0.1)


def hidden_mask(t, off, block_s=4.0, mod=3):
    return np.zeros(len(t), bool) if off < 0 else ((((t - t[0]) // int(block_s * 1e9)) % mod) == off)


def load_visual(npz):
    """generic frontend trajectory (one camera): t_ns (n,) wrist-camera clock (= m3_frames/<tag>/frames.json t_ns), p (n,3) camera centre in the
    segment world (any unit), q_xyzw (n,4) camera->world (OpenCV camera axes), segment_id (n,) (restart / new map = new id), optional conf (n,)"""
    z = np.load(npz); t = z["t_ns"].astype(np.int64) - int(FD.SLAM_DT); sid = z["segment_id"]; segs = []
    keep = np.isfinite(z["p"]).all(1) & np.isfinite(z["q_xyzw"]).all(1) & (np.linalg.norm(z["q_xyzw"], axis=1) > 1e-3)
    for k in np.unique(sid):
        m = keep & (sid == k)
        if m.sum() < 10: continue
        o = np.argsort(t[m], kind="stable"); tt, u = np.unique(t[m][o], return_index=True); o = o[u]
        segs.append(dict(t=tt, p=z["p"][m][o], R=R.from_quat(z["q_xyzw"][m][o]), sim3=np.ones(len(tt)), kf=np.zeros(len(tt), bool), lost_end=True, gi0=int(k), win=-1))   # last 0.2 s of a segment dropped, as for MASt3R
    return segs


def prepare(take, arm, visual=None):
    D = FD.load_arm(take, arm, with_slam=False)
    if visual: segs = load_visual(str(visual).format(take=take, arm=arm, cam=D["cam"]))
    else: segs, _ = load_segs_raw(take, D["cam"])
    cal = calibrate(D, segs, R.from_quat(np.load(HERE.parent / f"out/RX_pooled_{D['cam']}.npy")))
    D["RX"] = cal["RX_est"]; D["tX"] = FD.handeye_1022(D["cam"])[1]
    out = []
    for s in segs:                                           # no splitting (one map = one Sim3); steps touching a pose jump get no factor
        prep_seg(D, s, D["RX"]); bad = np.zeros(len(s["t"]), bool)
        for jt in s["jump_t"]: bad |= np.abs(s["t"] - jt) <= 0.05e9
        m = s["t"] <= s["t1"]
        if m.sum() >= 10: out.append(dict(t=s["t"][m], p=s["p"][m], R=s["R"][m], bad=bad[m], src=(s["gi0"], int(s["t"][0]))))
    D["segs"] = out; D["cal"] = cal; return D


def mask_tags(D, hid):
    D2 = dict(D); D2["ntags"] = np.where(hid, 0, D["ntags"]); D2["n_tagrot"] = np.where(hid, 0, D["n_tagrot"])
    Rf = R.from_matrix(D["R_fused"]); ok = ~hid
    if hid.any() and ok.any():                                # init rotation in hidden frames from the gyro, not the tag
        idx = np.where(ok)[0]; k = idx[np.clip(np.searchsorted(idx, np.arange(len(hid))) - 1, 0, len(idx) - 1)]
        Rg = Rf[k] * (FC.Qat(D, D["t"][k]).inv() * FC.Qat(D, D["t"])); M = Rf.as_matrix().copy(); M[hid] = Rg.as_matrix()[hid]; D2["R_fused"] = M
    return D2


def solve_demo(D, w0, w1, variant, hid):
    S = _solve(D, w0, w1, variant, hid, set())
    drop = set()
    for (src, t0, t1, nk), off_ in zip(S["seginfo"], np.cumsum([0] + [x[3] for x in S["seginfo"]])[:-1]):
        sc = S["scales"][off_:off_ + nk]
        if np.any(sc < 0.25 * S["s0"]) or np.any(sc > 4 * S["s0"]): drop.add(src)
    if drop: S = _solve(D, w0, w1, variant, hid, drop); S["dropped"] = len(drop)
    return S


def _solve(D, w0, w1, variant, hid, drop):
    D2 = mask_tags(D, hid); o = dict(FC.OPT)
    tn = np.arange(w0, w1 + 1, int(DT * 1e9), dtype=np.int64); N = len(tn)
    Rn, _ = FC.solve_rot(D2, tn, None, o)                    # gyro-led rotation (TCP), soft tag + SLAM relative rotation factors
    RXi = D["RX"].inv(); tX = D["tX"]
    def Pw(t):
        f = np.clip((t - tn[0]) / (DT * 1e9), 0, N - 1 - 1e-9); i = np.floor(f).astype(int); return i, f - i
    # unknowns: P (3N) then scales
    nS = 0; info = np.zeros(N, bool); seginfo = []
    # tags
    t = D2["t"]; m = (t >= tn[0]) & (t <= tn[-1]) & (D2["ntags"] > 0)
    tags_used = m.sum()
    # visual segments
    for sg in D["segs"]:
        k = (sg["t"] >= tn[0]) & (sg["t"] <= tn[-1])
        if k.sum() < 10 or sg["src"] in drop: continue
        ts, ps, Rs = sg["t"][k], sg["p"][k], sg["R"][k]
        Rbc = FC.rot_at(D, tn, Rn, ts) * RXi
        # rotation consistency per step (0.4 s window), SLAM vs gyro-led rotation
        h = 6; a_ = np.clip(np.arange(len(ts)) - h, 0, len(ts) - 1); b_ = np.clip(np.arange(len(ts)) + h, 0, len(ts) - 1)
        e = np.degrees(((Rs[a_].inv() * Rs[b_]).inv() * (Rbc[a_].inv() * Rbc[b_])).magnitude())
        g = np.degrees((Rbc[a_].inv() * Rbc[b_]).magnitude()); good = e <= 2.5 + 0.1 * g
        Ri = (Rbc[good] * Rs[good].inv()).mean() if good.sum() >= 5 else (Rbc * Rs.inv()).mean()
        u = Ri.apply(ps[1:] - ps[:-1]); lev = Rbc[1:].apply(tX) - Rbc[:-1].apply(tX); w = np.where(good[1:] & good[:-1], 1.0, SIG["rot_bad_w"])
        bad = sg["bad"][k]; ok = (np.diff(ts) < 0.1e9) & ~bad[1:] & ~bad[:-1]
        if variant == "v2": knots = np.array([ts[0]]); kid = np.zeros(len(ts) - 1, int); kw = np.zeros(len(ts) - 1)
        else:
            nk = max(1, int(np.ceil((ts[-1] - ts[0]) / (KNOT_S * 1e9)))) + 1; knots = np.linspace(ts[0], ts[-1], nk)
            f = np.clip((ts[:-1] - ts[0]) / max(ts[-1] - ts[0], 1) * (nk - 1), 0, nk - 1 - 1e-9); kid = np.floor(f).astype(int); kw = f - kid
        seginfo.append(dict(src=sg["src"], t0=int(ts[0]), t1=int(ts[-1]), s_off=nS, nk=len(knots), ts=ts, u=u, lev=lev, w=w, ok=ok, kid=kid, kw=kw))
        nS += len(knots)
    nv = 3 * N + nS
    # initial scale prior s0: per segment tag LS (non-hidden tags), median over segments
    s_est = []
    for si in seginfo:
        mm = (t >= si["t0"]) & (t <= si["t1"]) & (D2["ntags"] > 0)
        if mm.sum() >= 10:
            idx = np.where(mm)[0]; ia, ib = idx[:-5], idx[5:]; dpt = D["p_tag"][ib] - D["p_tag"][ia]
            sg_ = next(s for s in D["segs"] if s["src"] == si["src"]); pc = lambda tt: np.stack([np.interp(tt, sg_["t"], sg_["p"][:, q]) for q in range(3)], 1)
            Rbc = FC.rot_at(D, tn, Rn, t[ia]) * RXi; Ri_ = (Rbc * Slerp(sg_["t"] / 1e9, sg_["R"])(np.clip(t[ia], sg_["t"][0], sg_["t"][-1]) / 1e9).inv()).mean()
            x = Ri_.apply(pc(t[ib]) - pc(t[ia])); ss = (x * dpt).sum() / max((x * x).sum(), 1e-12)
            if ss > 0: s_est.append(ss)
    s0 = float(np.median(s_est)) if s_est else 0.1
    # assemble (as triplets with explicit column indices)
    R_, C_, V_, B_, S_, RB_ = [], [], [], [], [], []; nr = 0
    def addf(colmat, valmat, b, s, robust):
        """one 3-vector factor: colmat list of (col_base or scalar col, coef or vec) applied per axis"""
        nonlocal nr
        for a in range(3):
            for kind, c, v in colmat:
                R_.append(nr + a); C_.append(3 * c + a if kind == "P" else c); V_.append(v if np.ndim(v) == 0 else v[a])
        B_.extend(b); S_.extend([s] * 3); RB_.extend([robust] * 3); nr += 3
    ti, ta = Pw(t[m])
    for i_, a_, p_, n_ in zip(ti, ta, D["p_tag"][m], D2["ntags"][m]):
        addf([("P", i_, 1 - a_), ("P", i_ + 1, a_)], None, p_, SIG["tag2"] if n_ >= 2 else SIG["tag1"], True); info[i_] = info[min(i_ + 1, N - 1)] = True
    for si in seginfo:
        ts = si["ts"]; i0, a0 = Pw(ts[:-1]); i1, a1 = Pw(ts[1:])
        for k in np.where(si["ok"])[0]:
            cs = [("P", i1[k], 1 - a1[k]), ("P", i1[k] + 1, a1[k]), ("P", i0[k], -(1 - a0[k])), ("P", i0[k] + 1, -a0[k])]
            sc = 3 * N + si["s_off"] + si["kid"][k]
            if si["nk"] == 1: cs.append(("S", sc, -si["u"][k]))
            else: cs += [("S", sc, -(1 - si["kw"][k]) * si["u"][k]), ("S", sc + 1, -si["kw"][k] * si["u"][k])]
            addf(cs, None, si["lev"][k], SIG["slam"] / np.sqrt(si["w"][k]), True); info[[i0[k], i1[k]]] = True
    # weak absolute drift guard: every node toward the tag-only linear fill of the NON-hidden tags (held flat beyond the last tag), L2 sigma 0.15 m
    tk_ = np.where((D2["ntags"] > 0))[0]
    if len(tk_) > 1 and SIG["abs_prior"] < 1e5:
        base = np.stack([np.interp(tn, t[tk_], D["p_tag"][tk_, q]) for q in range(3)], 1)
        for n in range(N): addf([("P", n, 1.0)], None, base[n], SIG["abs_prior"], False)
    # scalar rows: scale prior + knot random walk
    srow = []
    for si in seginfo:
        for q in range(si["nk"]): srow.append(([(3 * N + si["s_off"] + q, 1.0)], s0, SIG["scale_prior"] * s0))
        for q in range(si["nk"] - 1): srow.append(([(3 * N + si["s_off"] + q + 1, 1.0), (3 * N + si["s_off"] + q, -1.0)], 0.0, SIG["knot_rw"] * s0))
    # smoothness (second difference), weak on info nodes, strong on no-information nodes
    acc_target = np.zeros((N, 3))
    if variant == "v3on":
        ti_ = D["imu_t"]; f = np.stack([np.interp(tn, ti_, D["acc"][:, a]) for a in range(3)], 1)       # TCP frame specific force (20 Hz avg-ish)
        k5 = np.ones(5) / 5; f = np.stack([np.convolve(f[:, a], k5, mode="same") for a in range(3)], 1); fB = Rn.apply(f); acc_target = fB - fB.mean(0)
    for n in range(1, N - 1):
        sa = (SIG["acc_info"] if (info[n - 1] or info[n] or info[n + 1]) else SIG["acc_gap"]) * DT * DT
        addf([("P", n - 1, 1.0), ("P", n, -2.0), ("P", n + 1, 1.0)], None, np.zeros(3), sa, False)
        if variant == "v3on": addf([("P", n - 1, 1.0), ("P", n, -2.0), ("P", n + 1, 1.0)], None, acc_target[n] * DT * DT, SIG["imu_acc"] * DT * DT, True)
    for cs, b, s in srow:
        for c, v in cs: R_.append(nr); C_.append(c); V_.append(v)
        B_.append(b); S_.append(s); RB_.append(False); nr += 1
    A = sp.csr_matrix((V_, (R_, C_)), shape=(nr, nv)); b = np.array(B_); s = np.array(S_); rb = np.array(RB_)
    w = 1.0 / s ** 2; x = None
    for it in range(8):
        H = (A.T @ sp.diags(w) @ A).tocsc() + sp.identity(nv) * 1e-9; x = splu(H).solve(A.T @ (w * b))
        e = (A @ x - b) / s
        cw = np.where(rb, 1.0 / (1.0 + (np.abs(e) / SIG["cauchy"]) ** 2), 1.0); w = cw / s ** 2
    P = x[:3 * N].reshape(N, 3); scales = x[3 * N:]
    return dict(tn=tn, Rn=Rn, P=P, scales=scales, s0=s0, seginfo=[(si["src"], si["t0"], si["t1"], si["nk"]) for si in seginfo], n_tags=int(tags_used))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("take"); ap.add_argument("arm"); ap.add_argument("variant", choices=["v2", "v3on", "v3off"])
    ap.add_argument("--offsets", default="0,1,2,-1"); ap.add_argument("--demos", default="")
    ap.add_argument("--visual", default="", help="generic frontend npz (pattern with {take} {arm} {cam}); default = raw MASt3R")
    ap.add_argument("--mode", default="B", choices=["A", "B", "auto"], help="B = fragmented input (one Sim3 per segment, default); A = continuous/global "
                    "input: all samples treated as ONE world (one Sim3 for the demo, shape preserved, tags soft + light smoothing only); auto = A if <= 2 segments")
    ap.add_argument("--no-abs-prior", action="store_true", help="(default now) no drift guard")
    ap.add_argument("--abs-prior", action="store_true", help="ablation: 0.15 m L2 pull toward the tag-only linear fill (REJECTED 10-23, kept for reproduction)")
    ap.add_argument("--name", default="", help="output variant name (default = VARIANT); e.g. v2_megasam"); a = ap.parse_args()
    if a.abs_prior and not a.no_abs_prior: SIG["abs_prior"] = 0.15
    D = prepare(a.take, a.arm, a.visual or None)
    if a.mode == "A" or (a.mode == "auto" and len(D["segs"]) <= 2):
        S_ = sorted(D["segs"], key=lambda g: g["t"][0]); t_ = np.concatenate([g["t"] for g in S_]); o_ = np.argsort(t_, kind="stable"); t_, u_ = np.unique(t_[o_], return_index=True); o_ = o_[u_]
        D["segs"] = [dict(t=t_, p=np.concatenate([g["p"] for g in S_])[o_], R=R.concatenate([g["R"] for g in S_])[o_], bad=np.concatenate([g["bad"] for g in S_])[o_], src=(0, 0))]
        print("mode A (one world for the whole input)", flush=True); var = a.variant.replace("v3off", "v3off")
    vr = "v2" if a.variant == "v2" else a.variant
    for W in FD.trims()[a.take]:
        if a.demos and W["demo"] not in [int(x) for x in a.demos.split(",")]: continue
        w0, w1 = W["w0"] - int(2e9), W["w1"] + int(2e9)
        for off in [int(x) for x in a.offsets.split(",")]:
            hid = hidden_mask(D["t"], off)
            S = solve_demo(D, w0, w1, "v2" if vr == "v2" else vr, hid)
            m = (D["t"] >= S["tn"][0]) & (D["t"] <= S["tn"][-1]); tt = D["t"][m]
            P = np.stack([np.interp(tt, S["tn"], S["P"][:, q]) for q in range(3)], 1); Rr = FC.rot_at(D, S["tn"], S["Rn"], tt)
            out = HERE / f"qa/{a.name or vr}/{a.take}_{a.arm}_d{W['demo']:02d}_h{off}.npz"; out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, t_ns=tt, p_B=P, R_B_tcp=Rr.as_matrix(), qa_hidden=hid[m], tn=S["tn"], Pn=S["P"], scales=S["scales"], s0=S["s0"],
                     seginfo=np.array(S["seginfo"], dtype=object), variant=vr, cal=str({k: v for k, v in D["cal"].items() if not hasattr(v, "as_quat")}))
            print(out.name, "s0 %.3f scales" % S["s0"], np.round(np.percentile(S["scales"], [10, 50, 90]), 3) if len(S["scales"]) else "-", flush=True)


if __name__ == "__main__":
    main()
