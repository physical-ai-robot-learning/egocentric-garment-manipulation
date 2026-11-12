"""11-11 tag-free start seeds (user removed every AprilTag; two black L tapes on the table mark the start spots).
From the rests right before the first demos (units resting at their L; else the first REST_S seconds of the take): table plane from depth (RANSAC, base frame via camcal),
L tapes = dark pixels ON the plane (|h| < ON_H) in compact components, hand units = dark points RAISED above the plane (H_LO..H_HI)
within NEAR_R of an L tape.  Per L: the median unit point -> start_seed_<take>.json {left|right: [x, y, z]} (left = larger base y,
as every earlier seed file), plus a debug image and tagfree_L_<take>.json (L corner pixels / base xyz, L long-leg yaw).
    python tagfree_seed.py <take> [--rest-s 4]"""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT)]
import camcal  # noqa
TAG = Path.home() / "ego_collector/datasets/tag_rig"
DARK, UNIT_DARK, ON_H, H_LO, H_HI, NEAR_R, L_AREA = 70, 45, 0.02, 0.025, 0.20, 0.22, (40, 3000)
START_X, START_Y = (0.10, 0.50), 0.45
REST_BEFORE_S, N_REST_DEMOS = 1.2, 3                                    # base frame [m]
PREV = {"left": np.array([0.289, 0.118]), "right": np.array([0.300, -0.222])}   # 11-11 L tapes: median of the clean detections
PREV_Z = {"left": 0.015, "right": 0.021}                                         # tape height [m], same detections
L_MAX_M = 0.04                                                                  # (121020, 124743, 133633-141226; 133202 right excluded)
REST_XYZ = {"left": np.array([0.220, 0.180, 0.090]), "right": np.array([0.243, -0.278, 0.095])}   # unit body at rest on its L (median of the
SEED_WARN_M = 0.05                                                                             # clean 11-11 takes 121020/124743/134741/135212)


def lift(depth, I, T_B_C):
    v, u = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]; z = depth.astype(float) / 1000
    P = np.stack([(u - I["cx"]) * z / I["fx"], (v - I["cy"]) * z / I["fy"], z], -1)
    return P @ T_B_C[:3, :3].T + T_B_C[:3, 3], z > 0.2


def plane(P, ok, it=300, th=0.006):
    rng = np.random.default_rng(0); Q = P[ok][::7]; best = (0, None)
    for _ in range(it):
        a, b, c = Q[rng.choice(len(Q), 3, replace=False)]; n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9: continue
        n /= np.linalg.norm(n); n = n if n[2] > 0 else -n; k = (np.abs((Q - a) @ n) < th).sum()
        if k > best[0]: best = (k, (n, a))
    n, a = best[1]; inl = np.abs((Q - a) @ n) < th; c0 = Q[inl].mean(0); _, _, vt = np.linalg.svd(Q[inl] - c0); n = vt[2] if vt[2][2] > 0 else -vt[2]
    return n, c0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("take"); ap.add_argument("--rest-s", type=float, default=4.0); a = ap.parse_args()
    tk = TAG / a.take; m = json.loads((tk / "meta.json").read_text()); I = m["intrinsics"]; t = np.load(tk / "t_ns.npy")
    # 11-11 take 124743: at the take start the units were still in the hands -> use the rests at the L right before the first demos
    # (the cue "멈추세요", REST_BEFORE_S before each demo start a0); fall back to the take start
    t0 = int(m["t_start_monotonic_ns"]); A0 = [t0 + int(p["t0"] * 1e9) for p in m["protocol"] if p["text"].startswith("자유:")][:N_REST_DEMOS]
    ks = np.concatenate([np.flatnonzero((t >= x - REST_BEFORE_S * 1e9) & (t <= x)) for x in A0] or [np.zeros(0, int)])[::2]
    if len(ks) < 5: ks = np.flatnonzero((t - t[0]) / 1e9 <= a.rest_s)[::3]
    T_B_C = camcal.T_B_C(a.take)
    G = np.median([cv2.imread(str(tk / f"color_{k:05d}.jpg"), 0) for k in ks], 0).astype(np.uint8)
    Ds = np.stack([cv2.imread(str(tk / f"depth_{k:05d}.png"), -1).astype(float) for k in ks]); Ds[Ds == 0] = np.nan; Dm = np.nan_to_num(np.nanmedian(Ds, 0)).astype(np.uint16)
    P, ok = lift(Dm, I, T_B_C); n, c0 = plane(P, ok); h = (P - c0) @ n; h[~ok] = np.nan
    dark = G < DARK
    on = (dark & (np.abs(np.nan_to_num(h, nan=9)) < ON_H)).astype(np.uint8)
    nc, lab, st, cen = cv2.connectedComponentsWithStats(on); Ls = []
    for i in range(1, nc):
        x, y, w, hh, area = st[i]
        if not (L_AREA[0] <= area <= L_AREA[1]) or max(w, hh) > 80: continue
        pix = np.argwhere(lab == i); Q = P[pix[:, 0], pix[:, 1]]; Q = Q[np.isfinite(Q[:, 0])]
        if len(Q) < 10: continue
        ext = np.ptp(Q[:, :2], 0).max()
        if not 0.025 <= ext <= 0.08: continue                # an L tape leg is a few cm; drop specks, cloth edges and long edges
        _, _, vt = np.linalg.svd(Q[:, :2] - Q[:, :2].mean(0)); yaw = float(np.degrees(np.arctan2(vt[0][1], vt[0][0])))
        Ls.append(dict(px=[float(cen[i][0]), float(cen[i][1])], xyz=Q.mean(0).round(4).tolist(), area=int(area), ext_m=round(float(ext), 3), main_axis_yaw_deg=round(yaw, 1)))
    raised = (G < UNIT_DARK) & (np.nan_to_num(h, nan=-1) > H_LO) & (np.nan_to_num(h, nan=9) < H_HI)
    R_ = P[raised]; seeds = {}
    # start area = where the start spots have always been (base x START_X, left y > 0 > right y); one L per side = the candidate
    # closest to that side's previous start spot
    Ls = [L for L in Ls if START_X[0] <= L["xyz"][0] <= START_X[1] and abs(L["xyz"][1]) <= START_Y]
    pick = {}
    for s, sg in (("left", 1), ("right", -1)):
        c = [L for L in Ls if sg * L["xyz"][1] > 0.03]
        if c: pick[s] = min(c, key=lambda L: np.linalg.norm(np.array(L["xyz"][:2]) - PREV[s]))
        # 11-11 take 133202: a black-shirt fold next to the right L passed the tape test 16 cm away -> the L-geometry start seed
        # (v32_common.tf_rest_centre) put the right body on the shirt for the whole take.  The tapes do not move: > L_MAX_M -> stored.
        if s in pick and np.linalg.norm(np.array(pick[s]["xyz"][:2]) - PREV[s]) > L_MAX_M:
            print(f"[seed] WARNING {s}: L candidate {np.round(np.array(pick[s]['xyz'][:2]) * 1000).tolist()} mm is "
                  f"{np.linalg.norm(np.array(pick[s]['xyz'][:2]) - PREV[s]) * 100:.0f} cm from the stored L -- stored L used"); del pick[s]
    for s_ in ("left", "right"):          # a unit resting ON its L hides the tape -> the stored (fixed) L position
        if s_ not in pick: pick[s_] = dict(px=[np.nan, np.nan], xyz=[float(PREV[s_][0]), float(PREV[s_][1]), float(PREV_Z[s_])], area=0, ext_m=np.nan, main_axis_yaw_deg=np.nan, stored=True)
    for s, L in pick.items():
        d = np.linalg.norm(R_[:, :2] - np.array(L["xyz"][:2]), axis=1); near = R_[d < NEAR_R]
        L["n_unit_pts"] = int(len(near))
        if len(near) < 30: raise SystemExit(f"[seed] no hand unit within {NEAR_R} m of the {s} L tape -- units must rest at the L at the take start")
        L["unit_xyz"] = np.median(near, 0).round(4).tolist()
    Ls = [pick["left"], pick["right"]]; seeds = {"left": Ls[0]["unit_xyz"], "right": Ls[1]["unit_xyz"]}
    # 11-11 take 133202 (black shirt): the crumpled dark shirt next to the right L joined the "unit" points -> right seed 25 cm off ->
    # the workspace ROI grew onto the desk equipment and the right track locked there.  The units rest at the SAME spot every take
    # (that is the protocol), so the seed is the fixed REST_XYZ; the detection is only a check (warning when > SEED_WARN_M off).
    for s_ in ("left", "right"):
        dd = float(np.linalg.norm(np.array(seeds[s_][:2]) - REST_XYZ[s_][:2])); Ls[0 if s_ == "left" else 1]["detected_minus_fixed_m"] = round(dd, 3)
        if dd > SEED_WARN_M: print(f"[seed] WARNING {s_}: detected rest {np.round(seeds[s_], 3).tolist()} is {dd * 100:.0f} cm from the fixed spot -- fixed spot used")
    seeds = {s_: REST_XYZ[s_].round(4).tolist() for s_ in ("left", "right")}
    (HERE / f"start_seed_{a.take}.json").write_text(json.dumps(seeds, indent=1))
    (HERE / f"tagfree_L_{a.take}.json").write_text(json.dumps(dict(take=a.take, rest_s=a.rest_s, plane_n=n.round(4).tolist(), L={s: L for s, L in zip(("left", "right"), Ls)}), indent=1))
    vis = cv2.cvtColor(G, cv2.COLOR_GRAY2BGR); vis[on > 0] = (0, 0, 255); vis[raised] = (0, 200, 0)
    for s, L in zip(("left", "right"), Ls):
        if np.isfinite(L["px"][0]): cv2.putText(vis, f"L {s}", tuple(int(v) for v in L["px"]), 0, 0.6, (255, 0, 0), 2)
    cv2.imwrite(str(HERE / f"tagfree_seed_{a.take}.jpg"), vis)
    print("[seed]", json.dumps(seeds), "| L", [(s, "stored" if L.get("stored") else L["px"], L["n_unit_pts"]) for s, L in zip(("left", "right"), Ls)])


if __name__ == "__main__":
    main()
