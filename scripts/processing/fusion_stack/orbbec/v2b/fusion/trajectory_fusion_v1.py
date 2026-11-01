"""10-24 trajectory_fusion_v1: whole-demo batch factor graph for the tag-point trajectory p(t) (= qa "TCP": body centre - R o) on a 20 Hz grid.
Rotation (all variants): Mahony AHRS (rest-period gyro bias), one yaw anchor at the demo's first >= 2-tag frame, per-demo IMU<->Orbbec lag
(build_candidate_v0.sync_lag on the CoTracker track).  Position = sparse linear IRLS (per-axis systems, shared Cauchy weights on the 3D residual):
  CT    CoTracker v2 body centre (cotrack/solve2 output) -> y = c - R o;  sigma by conf (2: SIG ct_hi, 1: ct_lo), Cauchy, + Orbbec bias b
  IMU   second difference = (R f(t+lag) - mean) DT^2, sigma imu_acc, Cauchy  (gravity removed by the demo mean, as stitch/wdo v3on)
  TAG   p = p_tag (non-hidden frames, >= 1 tag), sigma tag2/tag1 inflated by the tag-vs-AHRS rotation disagreement (quality), Cauchy
  SMOOTH second difference prior, acc_info near observations, acc_gap in unobserved spans (bridges gaps)
  SLAM  (variant D) wrist MASt3R relative displacement, per-segment scale -- needs SLAM for the take (not run for 155004)
Output is cropped to [first, last] observed node (no extrapolation).  Suffix g (e.g. Cg) = IMU factor only on unsupported nodes.
Variants: R = raw CoTracker y (no fusion) | A = CT + smooth (b = 0) | B = A + IMU | C = B + TAG (+ constant 3D Orbbec bias) | D = C + SLAM
          E = C + tag rotation correction (world-frame rotvec knots every ROT_KNOT_S, fit to non-hidden R_tag) + random-walk bias knots.
Hidden-tag protocol: runs h0/h1/h2 drop every tag in blocks block%3 == off (stitch/wdo.hidden_mask on the take's tag grid), h-1 = production.
--strict (no leakage): nothing in run h<off> is estimated from hidden tags -- yaw anchor (here and in solve2_<arm>_h<off>.npz from
cotrack/solve2.py <pack> <off>), tag position / rotation factors, bias; fixed body offset from dev take 123747.
Outputs stitch/qa/fus_<V>/{take}_{arm}_dNN_h{off}.npz (t_ns, p_B, R_B_tcp, + n_ct per node, bias) -> stitch/qa_eval.py; prints fusion metrics.
    python trajectory_fusion_v1.py <ctpack dir> <variants e.g. A,B,C,E> [--demos 0,1] [--strict --suffix _s] [--ct solve3]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import splu
from scipy.spatial.transform import Rotation as R
HERE = Path(__file__).resolve().parent; V2B = HERE.parent; ROOT = V2B.parents[1]
sys.path[:0] = [str(V2B), str(ROOT), str(ROOT / "stitch")]
import core as C  # noqa
import imu_motion as IM  # noqa
import build_candidate_v0 as V0  # noqa
from wdo import hidden_mask  # noqa
FD = C.FD; DT = 0.05; STRICT = False; SUFFIX = ""; CT = "solve2"
SIG = dict(ct_hi=0.010, ct_lo=0.025, tag2=0.004, tag1=0.008, tag_rot_deg=5.0, imu_acc=2.0, acc_info=30.0, acc_gap=3.0, cauchy=3.0,
           bias_prior=0.10, bias_rw=0.010, bias_knot_s=4.0, rot_knot_s=2.0, rot_tag_deg=3.0, rot_rw_deg=1.0)
OFF = json.loads((V2B / "body_offset_by_rig_v1.json").read_text())["offsets_by_rig"]


def irls(rows, nv, iters=8):
    """rows: list of (cols, coefs, target(3,), sigma, robust) -> x (nv,3); one sparse system per axis, shared weights on the 3D residual"""
    ri, ci, vi = [], [], []
    for r, (cols, cf, _, _, _) in enumerate(rows): ri += [r] * len(cols); ci += list(cols); vi += list(cf)
    A = sp.csr_matrix((vi, (ri, ci)), shape=(len(rows), nv)); B = np.array([r[2] for r in rows]); s = np.array([r[3] for r in rows]); rb = np.array([r[4] for r in rows])
    w = 1 / s ** 2
    for _ in range(iters):
        lu = splu((A.T @ sp.diags(w) @ A).tocsc() + sp.identity(nv) * 1e-10); X = np.stack([lu.solve(A.T @ (w * B[:, q])) for q in range(3)], 1)
        e = np.linalg.norm(A @ X - B, axis=1) / s; w = np.where(rb, 1 / (1 + (e / SIG["cauchy"]) ** 2), 1.0) / s ** 2
    return X, np.where(rb, w * s ** 2, 1.0)


def lin(tn, tq):
    """node index i and weight a such that p(tq) = (1-a) P[i] + a P[i+1]"""
    u = np.clip((tq - tn[0]) / (DT * 1e9), 0, len(tn) - 1 - 1e-9); i = np.floor(u).astype(int); return i, u - i


def rot_correction(tn, Ra, tq, Rq_tag, Rq_ahrs, sig_q):
    """world-frame rotvec correction knots (piecewise linear) fit to R_tag R_ahrs^-1 at non-hidden tag frames -> corrected R on tn"""
    tk = np.arange(tn[0], tn[-1] + SIG["rot_knot_s"] * 1e9, SIG["rot_knot_s"] * 1e9); K = len(tk); rows = []
    if len(tq) >= 5:
        d = (Rq_tag * Rq_ahrs.inv()).as_rotvec(); u = np.clip((tq - tk[0]) / (tk[1] - tk[0] if K > 1 else 1), 0, K - 1 - 1e-9); i = np.floor(u).astype(int); a = u - i
        for k in range(len(tq)): rows.append(([i[k], min(i[k] + 1, K - 1)], [1 - a[k], a[k]], d[k], sig_q[k], True))
    for k in range(K - 1): rows.append(([k, k + 1], [1.0, -1.0], np.zeros(3), np.radians(SIG["rot_rw_deg"]), False))
    rows.append(([0], [1.0], np.zeros(3), 1.0, False))
    X, _ = irls(rows, K); dn = np.stack([np.interp(tn, tk, X[:, q]) for q in range(3)], 1) if K > 1 else np.repeat(X, len(tn), 0)
    return R.from_rotvec(dn) * Ra


def fuse(var, D, tn, ct, Rn, fB, tag, hid_t):
    """one run -> P (N,3), bias, info; ct = dict(t, y, conf), tag = dict(t, p, n, q_deg) for this arm, hid_t = hidden mask on tag frames"""
    N = len(tn); vb = var[0]; use_bias = vb in ("C", "D", "E"); rw_bias = vb == "E"; imu_gap = var.endswith("g")
    tb = np.arange(tn[0], tn[-1] + SIG["bias_knot_s"] * 1e9, SIG["bias_knot_s"] * 1e9) if rw_bias else tn[:1]; K = len(tb) if use_bias else 0
    rows = []; info = np.zeros(N, bool)
    m = ct["conf"] > 0; i, a = lin(tn, ct["t"][m])
    if K > 1: ub = np.clip((ct["t"][m] - tb[0]) / (tb[1] - tb[0]), 0, K - 1 - 1e-9); ib = np.floor(ub).astype(int); ab = ub - ib
    for k in range(m.sum()):
        cols, cf = [i[k], i[k] + 1], [1 - a[k], a[k]]
        if K == 1: cols, cf = cols + [N], cf + [1.0]
        elif K > 1: cols, cf = cols + [N + ib[k], N + min(ib[k] + 1, K - 1)], cf + [1 - ab[k], ab[k]]
        rows.append((cols, cf, ct["y"][m][k], SIG["ct_hi"] if ct["conf"][m][k] == 2 else SIG["ct_lo"], True)); info[i[k]] = info[i[k] + 1] = True
    if vb in ("C", "D", "E"):
        mt = (tag["n"] >= 1) & ~hid_t; i, a = lin(tn, tag["t"][mt])
        s0 = np.where(tag["n"][mt] >= 2, SIG["tag2"], SIG["tag1"]) * (1 + (np.nan_to_num(tag["q_deg"][mt], nan=0.0) / SIG["tag_rot_deg"]) ** 2)
        for k in range(mt.sum()): rows.append(([i[k], i[k] + 1], [1 - a[k], a[k]], tag["p"][mt][k], s0[k], True)); info[i[k]] = info[i[k] + 1] = True
    acc = fB - fB.mean(0)
    for n in range(1, N - 1):
        sa = (SIG["acc_info"] if info[n - 1:n + 2].any() else SIG["acc_gap"]) * DT * DT
        rows.append(([n - 1, n, n + 1], [1.0, -2.0, 1.0], np.zeros(3), sa, False))
        if vb != "A" and not (imu_gap and info[n - 1:n + 2].any()): rows.append(([n - 1, n, n + 1], [1.0, -2.0, 1.0], acc[n] * DT * DT, SIG["imu_acc"] * DT * DT, True))
    if K: rows.append(([N], [1.0], np.zeros(3), SIG["bias_prior"], False))
    for k in range(K - 1): rows.append(([N + k, N + k + 1], [1.0, -1.0], np.zeros(3), SIG["bias_rw"], False))
    if not use_bias and not m.any(): return None
    X, wr = irls(rows, N + K)
    return X[:N], (X[N:] if K else np.zeros((0, 3))), info, np.flatnonzero(info)[[0, -1]]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pack"); ap.add_argument("variants"); ap.add_argument("--demos", default=""); ap.add_argument("--strict", action="store_true"); ap.add_argument("--suffix", default=""); ap.add_argument("--ct", default="solve2", help="CoTracker input: solve2 | solve3 (gated, cotrack/gate.py)"); A_ = ap.parse_args()
    global STRICT, SUFFIX, CT; STRICT = A_.strict; SUFFIX = A_.suffix; CT = A_.ct
    variants = A_.variants.split(","); demos = {int(x) for x in A_.demos.split(",") if x}
    if "D" in variants: raise SystemExit("variant D needs wrist SLAM segments for the take (none for 155004 yet)")
    for d in sorted(Path(A_.pack).iterdir()):
        if not (d / f"{CT}_left.npz").exists(): continue
        meta = json.loads((d / "meta.json").read_text()); take, demo = meta["take"], int(meta["demo"])
        if demos and demo not in demos: continue
        W = {w["demo"]: w for w in FD.trims()[take]}[demo]; asg = json.loads((Path.home() / "laundry_dit_training/data/tag_rig_derived" / take / "assignment.json").read_text())
        for arm in ("left", "right"):
            D = FD.load_arm(take, arm, with_slam=False); C.FC.prep(D); D["b_rest"] = IM.rest_bias(D, take, FD); t = D["t"]
            o = np.array(OFF[asg["rig_of_arm"][arm]]) / 1000
            for off in (-1, 0, 1, 2):
                # leakage guard: in hidden run h<off> nothing is estimated from hidden-block tags -- CoTracker input from the solve2 run whose
                # yaw anchor was visible-only (solve2_<arm>_h<off>.npz), our own yaw anchor visible-only, tag factors / rot correction visible-only
                hid_full = hidden_mask(t, off) if off >= 0 else np.zeros(len(t), bool)
                fz = d / (f"{CT}_{arm}.npz" if off < 0 else f"{CT}_{arm}_h{off}.npz")
                if not fz.exists():
                    if STRICT: raise SystemExit(f"missing {fz} (run cotrack/solve2.py / gate.py <pack> {off})")
                    fz = d / f"{CT}_{arm}.npz"
                z = np.load(fz); tct, cct, conf = z["t"].astype(np.int64), z["c"], z["conf"].astype(int)
                tg2 = (D["ntags"] >= 2) & (D["n_tagrot"] >= 2) & np.isfinite(D["R_tag"][:, 0, 0]) & ~hid_full; win = (t >= W["a0"]) & (t <= W["w1"] + 2e9)
                anc = np.flatnonzero(win & tg2)
                if not len(anc): j_ = np.flatnonzero(tg2); anc = j_[[np.argmin(np.abs(t[j_] - W["a0"]))]]
                ta, Ranc = t[anc[0]], R.from_matrix(D["R_tag"][anc[0]])
                lag, ncc = V0.sync_lag(D, tct, cct, np.arange(len(tct)), ta, Ranc) if np.isfinite(cct[:, 0]).sum() > 30 else (0, 0.0)
                t0, t1 = max(tct[0], W["w0"] - int(2e9)), min(tct[-1], W["w1"] + int(2e9)); tn = np.arange(t0, t1 + 1, int(DT * 1e9))
                Ra = IM.ahrs_base(D, tn + lag, ta + lag, Ranc)
                f = np.stack([np.interp(tn + lag, D["imu_t"], D["acc"][:, q]) for q in range(3)], 1); k5 = np.ones(5) / 5
                f = np.stack([np.convolve(f[:, q], k5, mode="same") for q in range(3)], 1)
                mt = (t >= tn[0]) & (t <= tn[-1]); Rq = IM.ahrs_base(D, t[mt] + lag, ta + lag, Ranc); okr = (D["n_tagrot"][mt] >= 1) & np.isfinite(D["R_tag"][mt][:, 0, 0])
                qd = np.full(mt.sum(), np.nan)
                if okr.any(): qd[okr] = np.degrees((R.from_matrix(D["R_tag"][mt][okr]).inv() * Rq[okr]).magnitude())
                tag = dict(t=t[mt], p=D["p_tag"][mt], n=D["ntags"][mt], q_deg=qd); hid_t = hid_full[mt]
                for var in variants:
                    od = ROOT / "stitch/qa" / f"fus_{var}{SUFFIX}"; od.mkdir(parents=True, exist_ok=True)
                    Rn = Ra
                    if var[0] == "E":
                        ok = okr & ~hid_t & (tag["n"] >= 1)
                        sq = np.radians(SIG["rot_tag_deg"]) * np.where(D["n_tagrot"][mt][ok] >= 2, 1.0, 2.0)
                        Rn = rot_correction(tn, Ra, t[mt][ok], R.from_matrix(D["R_tag"][mt][ok]), Rq[ok], sq)
                    Rct = Rn[np.clip(np.round((tct - tn[0]) / (DT * 1e9)).astype(int), 0, len(tn) - 1)]
                    inr = (tct >= tn[0]) & (tct <= tn[-1]) & np.isfinite(cct[:, 0])
                    ct = dict(t=tct[inr], y=cct[inr] - Rct[inr].apply(o), conf=np.where(np.isfinite(cct[inr, 0]), conf[inr], 0))
                    if var == "R":                         # raw CoTracker (no fusion; qa_eval interpolates linearly across gaps)
                        mm = ct["conf"] > 0; np.savez(od / f"{take}_{arm}_d{demo:02d}_h{off}.npz", t_ns=ct["t"][mm], p_B=ct["y"][mm], R_B_tcp=Rct[inr][mm].as_matrix()); continue
                    r_ = fuse(var, D, tn, ct, Rn, Rn.apply(f), tag, hid_t)
                    if r_ is None: continue
                    P, bias, info, (lo, hi) = r_; sl = slice(lo, hi + 1)   # no extrapolation beyond the first / last observation
                    np.savez(od / f"{take}_{arm}_d{demo:02d}_h{off}.npz", t_ns=tn[sl], p_B=P[sl], R_B_tcp=Rn.as_matrix()[sl], info=info[sl], bias=bias, lag_ns=lag)
                if off < 0: print(f"{take} d{demo:02d} {arm[0]} lag {lag / 1e6:.0f} ms ncc {ncc:.2f} | nodes {len(tn)} ct {int((ct['conf'] > 0).sum())} tags {int((tag['n'] >= 1).sum())}", flush=True)


if __name__ == "__main__":
    main()
