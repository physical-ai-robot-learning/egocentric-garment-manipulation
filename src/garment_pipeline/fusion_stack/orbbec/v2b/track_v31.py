"""10-23 Orbbec UMI tracker v3.1 = v3.0 (track_v3.py, frozen official) + four dev-take-only fixes:
  (1) seed lock: an unseeded arm only accepts a candidate within SEED_R of its start marker (no side-prior Hungarian on the first frame);
      after SEED_TIMEOUT_S without a seed it falls back to the v3.0 lost-arm rule
  (2) EXT_MAX re-checked on 123747 (correct-UMI vs false global candidates; see ext_check in the log)
  (3) W_FLAT swept on 123747 (0.05 / 0.1 / 0.2 / 0.3)
  (4) velocity: updated only for 0 < dt <= VEL_DT_MAX (else reset to 0); dt in the update floored at VEL_DT_MIN (frame drops)
  (5) table ROI (found on 123747 R d1: the global search locked onto dark equipment beside the table): points must lie inside the
      table-plane xy bounds (1st..99th percentile of plane inliers) + ROI_PAD
  (6) [tried, reverted: side prior only while both arms are lost -- no gain]
  (7) static-lock check (123747 R d1: locked onto desk equipment / hair): over the last STATIC_WIN_S the tracked centre stays within
      STATIC_D while the arm's IMU |omega| (rest-period bias removed, low-passed) has median > STATIC_W_DPS -> inconsistency; after
      STATIC_N consecutive inconsistent frames the arm is unlocked (lost) and that position is blacklisted (BLACK_R) for BLACK_S
  (8) workspace ROI from the protocol start markers (tag-free): x, y within the markers' range +- WS_R (intersected with (5))
Held-out for v3.1 (chosen before running it): 20261022_161756 (beige) and 20261022_130007 (black shirt).
--- v3.0 docstring ---
10-23 Orbbec UMI tracker v3 (targets the v1 failure analysis: search_miss, cloth_overlap, swap / false object).

Points per frame (stride 2, base frame): dark (gray < DARK) AND H_MIN..H_MAX above the table AND not in an image component touching the
bottom edge (operator).  NO size filter (v1 dropped components > MAX_AREA -> a unit lying on a dark shirt vanished).
Weights: w = 1 where the height top-hat (height minus its K_OPEN opening) > TH (locally raised: the rigid body), W_FLAT elsewhere
(flat dark cloth), so mean-shift prefers the raised body inside a big dark blob.
Candidates
  local   per tracked arm: weighted mean-shift (bandwidth BW) from the constant-velocity prediction
  global  when an arm is lost (no accepted centre for > LOST_S): modes of the whole frame (weighted mean-shift from seeds on every
          component, merged within MERGE), each with npts >= MIN_PTS and 3D extent <= EXT_MAX
Joint assignment (scipy linear_sum_assignment) of {left, right} to candidates, cost = distance to the arm's prediction / its gate
(gate = R_GATE + V_GATE * time since its last accepted centre, cap GATE_LOST for lost arms) + SIDE_W * side-prior violation
(lost arm only: base-y sign of the protocol start marker) ; pairs above cost 1 are not assigned; two arms never share a candidate;
a lost arm's candidate must be >= MIN_SEP from the other arm's current centre.
Identity: seeded once per demo window at the start markers (start_seed_v1.json, tag-free), then only this assignment.
Output out/trackv31_<take>.npz (+ global_cands: k, x, y, z, ext, npts of every global mode before the EXT_MAX test): t_ns, z_table, c_<arm>, npts_<arm>, valid_<arm> (same keys as v1/v2b) + per-arm logs
mode_<arm> (0 none, 1 local, 2 global), dpred_<arm> [m], cost_<arm>, ext_<arm>, ncand (global candidates), params.
Parameters set on 123747 only, then frozen (V3_P env overrides for the record).
    python track_v3.py <take>"""
import json, os, sys
from pathlib import Path
import cv2, numpy as np
from scipy.optimize import linear_sum_assignment
HERE = Path(__file__).resolve().parent
from garment_pipeline.fgo_tools import fgo_data as FD  # noqa
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
SEEDS = json.loads((HERE / "start_seed_v1.json").read_text())   # per-take override start_seed_<take>.json (10-26: fixed tags / start spots moved)
from garment_pipeline.fusion_stack import camcal  # noqa
P = dict(DARK=40, H_MIN=0.020, H_MAX=0.65, STRIDE=2, K_OPEN=15, TH=0.025, W_FLAT=0.3, BW=0.07, MIN_PTS=60, EXT_MAX=0.09, MERGE=0.06,
         R_GATE=0.07, V_GATE=0.5, GATE_LOST=0.6, LOST_S=0.3, SIDE_W=1.0, MIN_SEP=0.10, JUMP=0.06, N_SEEDS=4,
         SEED_R=0.15, SEED_TIMEOUT_S=3.0, VEL_DT_MIN=0.05, VEL_DT_MAX=0.25, ROI_PAD=0.02,
         STATIC_WIN_S=0.8, STATIC_D=0.015, STATIC_W_DPS=30.0, STATIC_N=5, BLACK_R=0.08, BLACK_S=5.0, WS_R=0.45)
P.update(json.loads(os.environ.get("V31_P", "{}")))


def wshift(pts, w, c, bw, it=6):
    for _ in range(it):
        sel = np.linalg.norm(pts - c, axis=1) < bw
        if w[sel].sum() < 1e-6: return None, sel
        c = (pts[sel] * w[sel, None]).sum(0) / w[sel].sum()
    return c, np.linalg.norm(pts - c, axis=1) < bw


def main(take):
    tk = FD.TAG / take; I = json.loads((tk / "meta.json").read_text())["intrinsics"]
    TBC = camcal.T_B_C(take)                                       # 10-26: per-take camera calibration
    global SEEDS
    if (HERE / f"start_seed_{take}.json").exists(): SEEDS = json.loads((HERE / f"start_seed_{take}.json").read_text())
    t = np.load(tk / "t_ns.npy").astype(np.int64); n = len(t); s = P["STRIDE"]
    H, W = cv2.imread(str(tk / "color_00000.jpg")).shape[:2]; vv, uu = np.mgrid[0:H:s, 0:W:s]; xn = (uu - I["cx"]) / I["fx"]; yn = (vv - I["cy"]) / I["fy"]
    fz = OUT / f"trackv2b_{take}.npz"
    if fz.exists(): z_tab = float(np.load(fz)["z_table"])
    else:                                                                  # same estimate as orbbec_track.py: mode of bright-pixel base z
        zs_ = []
        for k in np.linspace(0, n - 1, 12).astype(int):
            c0 = cv2.imread(str(tk / f"color_{k:05d}.jpg"), cv2.IMREAD_GRAYSCALE); d0 = cv2.imread(str(tk / f"depth_{k:05d}.png"), -1)
            if c0 is None or d0 is None: continue
            z0 = d0[::s, ::s].astype(np.float32) / 1000.0; ok0 = (z0 > 0.2) & (z0 < 2.0)
            X0 = np.stack([xn * z0, yn * z0, z0], -1) @ TBC[:3, :3].T + TBC[:3, 3]; zs_.append(X0[ok0 & (c0[::s, ::s] > 150)][:, 2])
        hist_, ed_ = np.histogram(np.concatenate(zs_), bins=400, range=(-0.3, 0.3)); z_tab = float(ed_[np.argmax(hist_)] + 0.00075)
    KER = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (P["K_OPEN"], P["K_OPEN"]))
    tb = []
    for k in np.linspace(0, n - 1, 6).astype(int):
        d0 = cv2.imread(str(tk / f"depth_{k:05d}.png"), -1)
        if d0 is None: continue
        z0 = d0[::s, ::s].astype(np.float32) / 1000.0; ok0 = (z0 > 0.2) & (z0 < 2.0)
        X0 = (np.stack([xn * z0, yn * z0, z0], -1) @ TBC[:3, :3].T + TBC[:3, 3])[ok0]; tb.append(X0[np.abs(X0[:, 2] - z_tab) < 0.01])
    tb = np.concatenate(tb); roi = (np.percentile(tb[:, 0], 1) - P["ROI_PAD"], np.percentile(tb[:, 0], 99) + P["ROI_PAD"], np.percentile(tb[:, 1], 1) - P["ROI_PAD"], np.percentile(tb[:, 1], 99) + P["ROI_PAD"])
    sx = [SEEDS[a][0] for a in ("left", "right")]; sy = [SEEDS[a][1] for a in ("left", "right")]
    roi = (max(roi[0], min(sx) - P["WS_R"]), min(roi[1], max(sx) + P["WS_R"]), max(roi[2], min(sy) - P["WS_R"]), min(roi[3], max(sy) + P["WS_R"]))   # (8)
    from garment_pipeline.fusion_stack.orbbec.v2b import imu_motion as IM  # noqa
    from scipy.ndimage import gaussian_filter1d
    omega = {}
    for a in ("left", "right"):
        D = FD.load_arm(take, a, with_slam=False); D["b0"] = np.zeros(3); b = IM.rest_bias(D, take, FD); w = np.degrees(np.linalg.norm(D["gyro"] - b, axis=1))
        omega[a] = (D["imu_t"], gaussian_filter1d(w, 0.05 * 420))
    arms = ("left", "right"); side = {a: np.sign(SEEDS[a][1]) for a in arms}
    hist = {a: [] for a in arms}; black = []; res_static = {a: np.zeros(n, bool) for a in arms}
    res = {f"c_{a}": np.full((n, 3), np.nan) for a in arms}
    for a in arms:
        res[f"npts_{a}"] = np.zeros(n, int); res[f"mode_{a}"] = np.zeros(n, int); res[f"dpred_{a}"] = np.full(n, np.nan)
        res[f"cost_{a}"] = np.full(n, np.nan); res[f"ext_{a}"] = np.full(n, np.nan)
    res["ncand"] = np.zeros(n, int); glog = []
    m = json.loads((tk / "meta.json").read_text()); t0 = int(m["t_start_monotonic_ns"])
    wins = [(t0 + int(p["t0"] * 1e9), t0 + int((p["t0"] + p["dur"]) * 1e9)) for p in m["protocol"] if p["text"].startswith("자유:")]
    for w0, w1 in wins:
        ks = np.flatnonzero((t >= w0 - 2e9) & (t <= w1 + 2e9))
        if not len(ks): continue                                         # protocol window without Orbbec frames
        st = {a: dict(c=np.array(SEEDS[a]), v=np.zeros(3), tl=t[ks[0]] - int(10e9), seeded=False) for a in arms}   # starts 'lost' at the marker
        for k in ks:
            col = cv2.imread(str(tk / f"color_{k:05d}.jpg"), cv2.IMREAD_GRAYSCALE); d = cv2.imread(str(tk / f"depth_{k:05d}.png"), -1)
            if col is None or d is None: continue
            zz = d[::s, ::s].astype(np.float32) / 1000.0; ok = (zz > 0.2) & (zz < 2.0)
            XG = np.stack([xn * zz, yn * zz, zz], -1) @ TBC[:3, :3].T + TBC[:3, 3]; h = XG[..., 2] - z_tab; g = col[::s, ::s]
            inroi = (XG[..., 0] > roi[0]) & (XG[..., 0] < roi[1]) & (XG[..., 1] > roi[2]) & (XG[..., 1] < roi[3])
            mk = (ok & (g < P["DARK"]) & (h > P["H_MIN"]) & (h < P["H_MAX"]) & inroi).astype(np.uint8)
            nc, lab, stt, _ = cv2.connectedComponentsWithStats(mk, connectivity=8)
            drop = [j for j in range(1, nc) if stt[j, cv2.CC_STAT_TOP] + stt[j, cv2.CC_STAT_HEIGHT] >= mk.shape[0] - 1]
            if drop: mk[np.isin(lab, drop)] = 0
            hz = np.where(ok, h, 0).astype(np.float32); top = hz - cv2.morphologyEx(hz, cv2.MORPH_OPEN, KER)
            sel_m = mk > 0; pts = XG[sel_m]; wts = np.where(top[sel_m] > P["TH"], 1.0, P["W_FLAT"]); labs = lab[sel_m]
            if len(pts) < P["MIN_PTS"]: continue
            # ---- candidates
            cands = []                                                    # (centre, npts, extent, mode, owner)
            pred = {}; lost = {}
            for a in arms:
                S = st[a]; dt = (t[k] - S["tl"]) / 1e9; lost[a] = (not S["seeded"]) or dt > P["LOST_S"]
                pred[a] = S["c"] + (S["v"] * min(dt, 0.5) if not lost[a] else 0)
                if not lost[a]:
                    c, sel = wshift(pts, wts, pred[a], P["BW"])
                    if c is not None and sel.sum() >= P["MIN_PTS"]: cands.append((c, int(sel.sum()), float(np.sqrt(pts[sel].var(0).sum())), 1))
            if any(lost.values()):
                rng = np.random.default_rng(k)
                for j in np.unique(labs):
                    idx = np.flatnonzero(labs == j)
                    if len(idx) < P["MIN_PTS"]: continue
                    for sd in pts[rng.choice(idx, min(P["N_SEEDS"] + len(idx) // 400, len(idx)), replace=False)]:
                        c, sel = wshift(pts, wts, sd, P["BW"])
                        if c is None or sel.sum() < P["MIN_PTS"]: continue
                        ext = float(np.sqrt(pts[sel].var(0).sum()))
                        if any(np.linalg.norm(c - q[0]) < P["MERGE"] for q in cands): continue
                        glog.append((k, *c, ext, int(sel.sum())))
                        if ext > P["EXT_MAX"]: continue
                        cands.append((c, int(sel.sum()), ext, 2))
                res["ncand"][k] = sum(1 for q in cands if q[3] == 2)
            if not cands: continue
            # ---- joint assignment
            Cm = np.full((2, len(cands)), 9.0)
            for i, a in enumerate(arms):
                S = st[a]; dt = (t[k] - S["tl"]) / 1e9; gate = P["GATE_LOST"] if lost[a] else min(P["R_GATE"] + P["V_GATE"] * dt, P["GATE_LOST"])
                for j, (c, npt, ext, md) in enumerate(cands):
                    cost = np.linalg.norm(c - pred[a]) / gate
                    if any(np.linalg.norm(c - bp) < P["BLACK_R"] and t[k] < bt for bp, bt in black): Cm[i, j] = 9.0; continue   # (7)
                    if not S["seeded"] and (t[k] - t[ks[0]]) / 1e9 <= P["SEED_TIMEOUT_S"]:
                        Cm[i, j] = np.linalg.norm(c - np.array(SEEDS[a])) / P["SEED_R"]; continue     # (1) seed lock: start marker only
                    if lost[a]:
                        cost += P["SIDE_W"] * float(np.sign(c[1]) != side[a] and abs(c[1]) > 0.05)
                        o = "right" if a == "left" else "left"
                        if not lost[o] and np.linalg.norm(c - st[o]["c"]) < P["MIN_SEP"]: cost = 9.0
                    elif md == 2 and np.linalg.norm(c - pred[a]) > P["JUMP"]: cost = 9.0          # tracked arms do not jump to a global mode
                    Cm[i, j] = cost
            ri, ci = linear_sum_assignment(Cm)
            for i, j in zip(ri, ci):
                if Cm[i, j] > 1.0: continue
                a = arms[i]; c, npt, ext, md = cands[j]; S = st[a]; dt = (t[k] - S["tl"]) / 1e9
                if not lost[a] and np.linalg.norm(c - pred[a]) > P["JUMP"] + P["V_GATE"] * max(dt - 0.067, 0): continue
                S["v"] = (0.5 * S["v"] + 0.5 * (c - S["c"]) / max(dt, P["VEL_DT_MIN"])) if (not lost[a] and 0 < dt <= P["VEL_DT_MAX"]) else np.zeros(3)   # (4)
                res[f"dpred_{a}"][k] = np.linalg.norm(c - pred[a]); res[f"cost_{a}"][k] = Cm[i, j]
                S["c"], S["tl"], S["seeded"] = c, t[k], True
                hist[a] = [(tt_, cc_) for tt_, cc_ in hist[a] if t[k] - tt_ <= P["STATIC_WIN_S"] * 1e9] + [(t[k], c)]
                if (t[k] - hist[a][0][0]) >= 0.8 * P["STATIC_WIN_S"] * 1e9:
                    H_ = np.array([cc_ for _, cc_ in hist[a]]); ti_, wv_ = omega[a]; mw = (ti_ >= hist[a][0][0]) & (ti_ <= t[k])
                    if np.linalg.norm(H_ - H_.mean(0), axis=1).max() < P["STATIC_D"] and mw.any() and np.median(wv_[mw]) > P["STATIC_W_DPS"]:
                        S["nstat"] = S.get("nstat", 0) + 1
                        if S["nstat"] >= P["STATIC_N"]:
                            black.append((c.copy(), t[k] + int(P["BLACK_S"] * 1e9))); S["seeded"] = True; S["tl"] = t[k] - int(10e9); hist[a] = []; S["nstat"] = 0
                            res_static[a][k] = True; continue                    # unlocked: this frame's centre is not written
                    else: S["nstat"] = 0
                res[f"c_{a}"][k] = c; res[f"npts_{a}"][k] = npt; res[f"ext_{a}"][k] = ext; res[f"mode_{a}"][k] = md
    for a in arms: res[f"valid_{a}"] = np.isfinite(res[f"c_{a}"][:, 0])
    res.update({f"static_unlock_{a}": res_static[a] for a in ("left", "right")})
    np.savez(OUT / f"{os.environ.get('V31_OUT', 'trackv31')}_{take}.npz", t_ns=t, z_table=z_tab, roi=np.array(roi), params=json.dumps(P), global_cands=np.array(glog).reshape(-1, 6), **res)
    inw = np.zeros(n, bool)
    for w0, w1 in wins: inw |= (t >= w0) & (t <= w1)
    print(take, "static unlocks", {a: int(res_static[a].sum()) for a in arms}, "roi", np.round(roi, 2), {a: round(float(res[f"valid_{a}"][inw].mean()), 3) for a in arms}, "global-mode share", {a: round(float((res[f"mode_{a}"][inw] == 2).mean()), 3) for a in arms})


if __name__ == "__main__":
    main(sys.argv[1])
