"""11-11 A0 = marker-free label test on the S03G flatten episodes (user: "한번 해보").
What changes when the hand tags are removed, everything else = production Cg (fus_Cg_g h-1):
  yaw   AHRS (Mahony tilt, rest gyro bias) with an ARBITRARY yaw, then ONE yaw theta per demo x arm from motion
        (IMU horizontal linear acc rotated onto the Orbbec track horizontal acc, closed-form 2-D Procrustes = yaw_from_motion v2)
  lag   IMU<->Orbbec clock offset from |horizontal acc| NCC (yaw-invariant), not from the tag-anchored sync
  bias  none (a constant per-demo xyz offset cancels in REL32 actions and in the task-anchor state)
  side  which track is the left arm = the one with the larger mean base-y over the demo (no tag rig identity)
Truth = Cg (AHRS + tag yaw anchor).  Label impact = REL32 translation difference, R_t^T (p(t+k) - p(t)), A0 vs Cg, for the
S03 chunk span (32 rows x 0.05 s x 0.3 = 0.48 s real) and 1.6 s, plus the task-anchor state translation (vs the demo start).
Caveat: the Orbbec track used for the yaw fit is the Cg position (tag-fused); CoTracker-only solve3 outputs no longer exist on disk.
    python a0_tagfree_test.py <out_dir>"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
ROOT = Path.home() / "ego_collector/datasets/m3_runs/_imu_consistency"; V2B = ROOT / "orbbec/v2b"
sys.path[:0] = [str(V2B), str(ROOT), str(ROOT / "stitch")]
import core as C  # noqa
import imu_motion as IM  # noqa
FD = C.FD; QA = ROOT / "stitch/qa/fus_Cg_g"; LR = Path.home() / "laundry_dit_training/data/lerobot"


def theta_hat(ai, ao):
    m = np.isfinite(ao[:, 0]) & np.isfinite(ai[:, 0]) & (np.linalg.norm(ao[:, :2], axis=1) > IM.E_MIN) & (np.linalg.norm(ai[:, :2], axis=1) > IM.E_MIN)
    if m.sum() < 15: return np.nan, int(m.sum())
    x, y = ai[m, :2], ao[m, :2]
    return float(np.arctan2(np.sum(x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]), np.sum(x[:, 0] * y[:, 0] + x[:, 1] * y[:, 1]))), int(m.sum())


def imu_acc(D, g, ta):
    Rf = IM.ahrs_base(D, g, ta, R.identity())                                   # arbitrary yaw (0 at ta)
    f = np.stack([np.interp(g, D["imu_t"], D["acc"][:, k]) for k in range(3)], 1); fB = Rf.apply(f)
    return gaussian_filter1d(fB - np.median(fB, 0), IM.SIG_S * IM.HZ, axis=0, mode="nearest")


def free_lag(D, g, ao, ta):
    """yaw-invariant sync: |horizontal acc| IMU vs track, NCC over -500..300 ms (same search range as build_candidate_v0.sync_lag)"""
    m = np.isfinite(ao[:, 0]); B = np.linalg.norm(ao[m, :2], axis=1); B = B - B.mean(); best = (0, -9.0)
    for L in range(-500, 301, 20):
        a = np.linalg.norm(imu_acc(D, g + int(L * 1e6), ta + int(L * 1e6))[m, :2], axis=1); a = a - a.mean()
        cc = float((a * B).sum() / np.sqrt((a * a).sum() * (B * B).sum() + 1e-12))
        if cc > best[1]: best = (L, cc)
    return int(best[0] * 1e6), best[1]


def rel_err(t, P, Ra, Rb, span_s):
    k = int(round(span_s / 0.05)); n = len(t) - k
    if n <= 0: return np.array([])
    d = P[k:] - P[:n]; ea = np.einsum("nji,nj->ni", Ra[:n], d); eb = np.einsum("nji,nj->ni", Rb[:n], d)
    ok = np.isfinite(ea[:, 0]) & np.isfinite(eb[:, 0]); return np.linalg.norm(ea - eb, axis=1)[ok] * 1000


def main(out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True); eps = []
    for sp in ("train", "val", "test"):
        e = json.loads((LR / f"flat1103_1104_1105_S03G_xvla_{sp}/meta/split.json").read_text())["episodes"]
        eps += [(sp, x) for x in e.get(sp, sum(e.values(), []))]
    TR = FD.trims(); rows = []; cache = {}
    for sp, x in eps:
        take, dm = x.split("/"); demo = int(dm[4:]); W = {w["demo"]: w for w in TR[take]}.get(demo)
        Z = {arm: QA / f"{take}_{arm}_d{demo:02d}_h-1.npz" for arm in ("left", "right")}
        if W is None or not all(f.exists() for f in Z.values()): print("skip", x, flush=True); continue
        ymean = {}
        for arm in ("left", "right"):
            if (take, arm) not in cache:
                if len(cache) >= 2 and all(k[0] != take for k in cache): cache.clear()
                D = FD.load_arm(take, arm, with_slam=False); C.FC.prep(D); D["b_rest"] = IM.rest_bias(D, take, FD); cache[(take, arm)] = D
            D = cache[(take, arm)]; z = np.load(Z[arm]); t, P, Rc = z["t_ns"].astype(np.int64), z["p_B"], z["R_B_tcp"]; lag_c = int(z["lag_ns"])
            ymean[arm] = float(np.nanmean(P[:, 1])); g = IM.grid(t[0], t[-1]); ao = IM.track_acc(t, P, g); ta = t[0]
            lag, ncc = free_lag(D, g, ao, ta)
            th, n = theta_hat(imu_acc(D, g + lag, ta + lag), ao)
            r = dict(split=sp, take=take, demo=demo, arm=arm, lag_cg_ms=lag_c / 1e6, lag_free_ms=lag / 1e6, ncc=ncc, n_yaw=n)
            if np.isfinite(th):
                Ra = (R.from_euler("z", th) * IM.ahrs_base(D, t + lag, ta + lag, R.identity())).as_matrix()
                dR = R.from_matrix(Ra) * R.from_matrix(Rc).inv(); eul = dR.as_euler("zyx", degrees=True)
                r.update(yaw_err_deg=float(np.median(eul[:, 0])), tilt_diff_deg=float(np.median(np.hypot(eul[:, 1], eul[:, 2]))),
                         rot_diff_p90_deg=float(np.percentile(dR.magnitude(), 90) * 180 / np.pi))
                for nm, s in (("rel048", 0.48), ("rel160", 1.6)):
                    e = rel_err(t, P, Ra, Rc, s); r[f"{nm}_p50"], r[f"{nm}_p90"] = (np.median(e), np.percentile(e, 90)) if len(e) else (np.nan, np.nan)
                d0 = P - P[0]; e = np.linalg.norm(np.einsum("ji,nj->ni", Ra[0], d0) - np.einsum("ji,nj->ni", Rc[0], d0), axis=1) * 1000
                r["state_p50"], r["state_p90"] = np.nanmedian(e), np.nanpercentile(e, 90)
            rows.append(r)
        for r in rows[-2:]: r["side_ok"] = ymean["left"] > ymean["right"]
        a = rows[-2:]; print(x, " | ".join(f"{q['arm'][0]} yaw {q.get('yaw_err_deg', np.nan):6.1f} lag {q['lag_free_ms']:5.0f}/{q['lag_cg_ms']:5.0f} rel048 {q.get('rel048_p90', np.nan):5.1f}" for q in a), "side", a[0]["side_ok"], flush=True)
    df = pd.DataFrame(rows); df.to_csv(out / "a0_tagfree.csv", index=False, float_format="%.3f")
    ay = df["yaw_err_deg"].abs()
    print(f"\ndemo-arms {len(df)}, no yaw estimate {ay.isna().sum()}")
    print(f"|yaw err| deg: p50 {ay.median():.1f} p90 {ay.quantile(.9):.1f} max {ay.max():.1f} | <=2 {(ay <= 2).mean():.2f} <=5 {(ay <= 5).mean():.2f} >10 {(ay > 10).mean():.2f}")
    print(f"tilt diff deg p50 {df['tilt_diff_deg'].median():.2f} | lag diff ms p50 {(df['lag_free_ms'] - df['lag_cg_ms']).abs().median():.0f} p90 {(df['lag_free_ms'] - df['lag_cg_ms']).abs().quantile(.9):.0f}")
    for c in ("rel048", "rel160", "state"):
        print(f"{c:7s} label diff mm: median-of-p50 {df[c + '_p50'].median():.1f}  median-of-p90 {df[c + '_p90'].median():.1f}  worst p90 {df[c + '_p90'].max():.1f}")
    print(f"side rule correct {df.groupby(['take', 'demo'])['side_ok'].first().mean():.3f}")
    print(df.groupby("arm")["yaw_err_deg"].apply(lambda s: s.abs().quantile([.5, .9]).round(1).tolist()).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
