"""10-25 CoTracker v2 validity gates (solve2 output -> solve3_<arm>[_h<off>].npz, same keys; conf set to 0 where a gate fires).
Targets the three right-arm failure modes of the 10-24 held-out / new-20 review (thresholds fixed a priori, checked on dev 155004 / 123747 only):
  G1 imu   IMU-motion consistency: track acceleration (imu_motion.track_acc) vs IMU linear acceleration in the base frame (AHRS rotation,
           per-demo IMU<->Orbbec lag from the track, gravity = median), centred-window NCC (WIN_S, energy floor E_MIN).  A window with
           NCC < NCC_BAD is inconsistent (the IMU moves and the track does not = hand left the view / stale points; or the track moves
           differently = hair, cloth).  Bad runs >= MIN_BAD_S, dilated by half a window -> invalid, UNLESS >= TAG_N visible tag frames in
           the run agree with the track (median < TAG_BAD): 10-25 audit, tag-checked G1 cuts were either 325-600 mm wrong (true failures)
           or 40-70 mm = as good as the kept frames (123747: IMU<->track lag wanders -240..-40 ms, NCC 0.55-0.78 at best lag).
  G2 lr    L/R collision: both arms valid and |c_L - c_R| < LR_MIN -> per collision run (dilated LR_PAD) the arm whose own visible tags
           disagree (> TAG_BAD) or, without tag evidence, the lower mean G1 NCC loses (invalid); no evidence either way -> both invalid.
  G3 tag   visible (non-hidden) >= 1-tag body points vs the track: > TAG_BAD on >= TAG_N frames inside a TAG_WIN window -> track invalid there.
In hidden runs only non-hidden tags are used (yaw anchor too), as everywhere else.
    python gate.py <pack> [hide_off]"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
HERE = Path(__file__).resolve().parent; sys.path[:0] = [str(HERE.parent), str(HERE.parents[2]), str(HERE.parents[2] / "stitch")]
import core as C  # noqa
import imu_motion as IM  # noqa
import build_candidate_v0 as V0  # noqa
from wdo import hidden_mask  # noqa
FD = C.FD
G = dict(WIN_S=1.0, E_MIN=0.5, NCC_BAD=0.2, MIN_BAD_S=0.4, LR_MIN=0.08, LR_PAD_S=0.3, TAG_BAD=0.10, TAG_N=3, TAG_WIN_S=1.0)
OFF = json.loads((HERE.parent / "body_offset_by_rig_v1.json").read_text())["offsets_by_rig"]


def runs(b):
    x = np.r_[0, b.astype(int), 0]; return list(zip(np.flatnonzero(np.diff(x) == 1), np.flatnonzero(np.diff(x) == -1)))


def imu_ncc(D, t, c, W, hid):
    """per Orbbec-pack frame: G1 window NCC (NaN = undecided) + the AHRS rotation on those frames"""
    tt = D["t"]; tg2 = (D["ntags"] >= 2) & (D["n_tagrot"] >= 2) & np.isfinite(D["R_tag"][:, 0, 0]) & ~hid; win = (tt >= W["a0"]) & (tt <= W["w1"] + 2e9)
    anc = np.flatnonzero(win & tg2)
    if not len(anc): j_ = np.flatnonzero(tg2); anc = j_[[np.argmin(np.abs(tt[j_] - W["a0"]))]]
    ta, Ra = tt[anc[0]], R.from_matrix(D["R_tag"][anc[0]]); v = np.isfinite(c[:, 0])
    lag, _ = V0.sync_lag(D, t, c, np.arange(len(t)), ta, Ra) if v.sum() > 30 else (0, 0.0)
    g = IM.grid(t[0], t[-1]); Rg = IM.ahrs_base(D, g + lag, ta + lag, Ra)
    f = np.stack([np.interp(g + lag, D["imu_t"], D["acc"][:, k]) for k in range(3)], 1); ai = Rg.apply(f); ai -= np.median(ai, 0)
    ai = gaussian_filter1d(ai, IM.SIG_S * IM.HZ, axis=0, mode="nearest"); ao = IM.track_acc(t, c, g)
    ncc = IM.window_ncc(ao, ai, win_s=G["WIN_S"], e_min=G["E_MIN"])
    Rk = IM.ahrs_base(D, t + lag, ta + lag, Ra).as_matrix()
    return np.interp(t, g, np.nan_to_num(ncc, nan=9.0)), np.interp(t, g, np.isfinite(ncc).astype(float)) > 0.5, Rk


def main(pack, hoff):
    for d in sorted(Path(pack).iterdir()):
        fz = {a: d / (f"solve2_{a}.npz" if hoff is None else f"solve2_{a}_h{hoff}.npz") for a in ("left", "right")}
        if not all(f.exists() for f in fz.values()): continue
        meta = json.loads((d / "meta.json").read_text()); take, demo = meta["take"], int(meta["demo"]); W = {w["demo"]: w for w in FD.trims()[take]}[demo]
        asg = json.loads((Path.home() / "laundry_dit_training/data/tag_rig_derived" / take / "assignment.json").read_text())
        Z = {a: dict(np.load(fz[a])) for a in fz}; t = Z["left"]["t"].astype(np.int64); n = len(t); dt = np.median(np.diff(t)) / 1e9
        bad = {}; ncc = {}; tagd = {}
        for a in ("left", "right"):
            D = FD.load_arm(take, a, with_slam=False); C.FC.prep(D); D["b_rest"] = IM.rest_bias(D, take, FD)
            hid = hidden_mask(D["t"], hoff) if hoff is not None else np.zeros(len(D["t"]), bool)
            c = np.where((Z[a]["conf"] > 0)[:, None], Z[a]["c"], np.nan); nc, dec, Rk = imu_ncc(D, t, c, W, hid); ncc[a] = np.where(dec, nc, np.nan)
            k = np.clip(np.searchsorted(D["t"], t), 0, len(D["t"]) - 1); vis = (D["ntags"][k] >= 1) & ~hid[k] & (np.abs(D["t"][k] - t) < 3e7)
            body = D["p_tag"][k] + np.einsum("nij,j->ni", Rk, np.array(OFF[asg["rig_of_arm"][a]]) / 1000)
            dd = np.linalg.norm(c - body, axis=1); dis = vis & np.isfinite(dd) & (dd > G["TAG_BAD"]); tagd[a] = np.where(vis & np.isfinite(dd), dd, np.nan)
            b1 = np.zeros(n, bool); h = int(G["WIN_S"] / 2 / dt)                       # G1 (vetoed where visible tags agree with the track)
            for s, e in runs(dec & (nc < G["NCC_BAD"])):
                if (e - s) * dt < G["MIN_BAD_S"]: continue
                s2, e2 = max(s - h, 0), e + h; td = tagd[a][s2:e2]; td = td[np.isfinite(td)]
                if len(td) >= G["TAG_N"] and np.median(td) < G["TAG_BAD"]: continue
                b1[s2:e2] = True
            b3 = np.zeros(n, bool); h3 = int(G["TAG_WIN_S"] / 2 / dt)                  # G3
            for j in np.flatnonzero(dis):
                if dis[max(j - h3, 0):j + h3 + 1].sum() >= G["TAG_N"]: b3[max(j - h3, 0):j + h3 + 1] = True
            bad[a] = dict(imu=b1, tag=b3, lr=np.zeros(n, bool))
        cl, cr = [np.where((Z[a]["conf"] > 0)[:, None], Z[a]["c"], np.nan) for a in ("left", "right")]
        coll = np.linalg.norm(cl - cr, axis=1) < G["LR_MIN"]; hp = int(G["LR_PAD_S"] / dt)       # G2
        for s, e in runs(coll):
            s, e = max(s - hp, 0), min(e + hp, n)
            ev = {a: (np.nanmedian(tagd[a][s:e]) if np.isfinite(tagd[a][s:e]).any() else np.nan, np.nanmean(ncc[a][s:e]) if np.isfinite(ncc[a][s:e]).any() else np.nan) for a in ("left", "right")}
            tl, tr_ = ev["left"][0], ev["right"][0]
            if np.isfinite(tl) or np.isfinite(tr_):
                lose = [a for a in ("left", "right") if np.isfinite(ev[a][0]) and ev[a][0] > G["TAG_BAD"]] or ([] if (np.isfinite(tl) and np.isfinite(tr_)) else
                        [a for a in ("left", "right") if not np.isfinite(ev[a][0])])
            elif np.isfinite(ev["left"][1]) and np.isfinite(ev["right"][1]) and abs(ev["left"][1] - ev["right"][1]) > 0.15:
                lose = ["left" if ev["left"][1] < ev["right"][1] else "right"]
            else: lose = ["left", "right"]
            for a in lose: bad[a]["lr"][s:e] = True
        for a in ("left", "right"):
            b = bad[a]; anyb = b["imu"] | b["tag"] | b["lr"]; z = Z[a]; z["conf"] = np.where(anyb, 0, z["conf"]); z["gate_imu"] = b["imu"]; z["gate_tag"] = b["tag"]; z["gate_lr"] = b["lr"]
            z["ncc"] = ncc[a]; z["gate_params"] = json.dumps(G)
            np.savez(d / (f"solve3_{a}.npz" if hoff is None else f"solve3_{a}_h{hoff}.npz"), **z)
            v0 = (Z[a]["c"][:, 0] == Z[a]["c"][:, 0]).mean()
            print(f"{d.name} {a[0]} h{hoff if hoff is not None else -1}: cut imu {b['imu'].mean() * 100:4.1f}% tag {b['tag'].mean() * 100:4.1f}% lr {b['lr'].mean() * 100:4.1f}% "
                  f"| valid {np.mean(Z[a]['conf'] > 0) * 100:4.1f}%", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
