"""11-11 tag-free yaw anchors: per demo x arm, the world yaw of the Mahony gravity frame from motion (no marker).
The IMU horizontal linear acceleration (Mahony frame, rest gyro bias) is rotated about z onto the Orbbec track horizontal acceleration
(closed-form 2-D Procrustes over samples where both move, = fusion/yaw_from_motion.py v2); the IMU<->Orbbec lag is found first on
|horizontal acc| (yaw-invariant NCC, same search range as build_candidate_v0.sync_lag).  Track = --track (trackv31speed: the v31
Orbbec blob tracker, needs no rotation) or --pack <ctpack dir> (solve3 CoTracker centres, for a second pass).
Writes TAG/<take>/tagfree.json anchors[arm][demo] = yaw [deg] (+ yaw_info with n, residual, half-split yaws, lag, ncc).
11-11 L mode (user: "IMU + L start pose"): the IMU has no absolute yaw (no magnetometer; gravity is yaw-blind) and the rest-bias gyro
yaw drifts ~0.6-4 deg/min, nearly linearly (5 tagged takes) -> every demo starts with the unit resting at its L tape (cue "멈추세요",
REST_S before a0), whose world heading H_L[arm] is a constant (tagfree_L_heading.json).  Per demo psi_k = H_L - azimuth of the Mahony
x axis at that rest; a robust line psi(t) over the take's demos (drift) evaluated at each demo's middle = the anchor (falls back to
psi_k with < 3 inlier demos).  The motion yaw stays as a check (|L - motion| > CHECK_DEG -> flagged in yaw_info).
--calibrate-L: H_L from this take's motion yaws (robust circular mean over demos), shifted by TAG_OFFSET so the headings follow the
tag convention of all earlier (tag-labelled) training data: A0 test on the S03G demos, motion yaw = tag yaw + 5.7 (left) / +4.5 (right).
    python tagfree_yaw.py <take> [--track trackv31speed | --pack DIR] [--calibrate-L] [--mode L|motion]"""
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "stitch")]
import core as C  # noqa
import imu_motion as IM  # noqa
FD = C.FD
LH = HERE / "tagfree_L_heading.json"; REST_S, OUT_DEG, FIT_OUT_DEG, CHECK_DEG, REST_DPS = 1.5, 15.0, 8.0, 5.0, 8.0   # REST_DPS: unit not resting at the L -> no L anchor for that demo
TAG_OFFSET = {"left": 5.57, "right": 4.56}   # motion yaw - tag yaw [deg], A0 test (all 392 S03G demos, 11-11 final)


def wrap(a): return (np.asarray(a) + 180) % 360 - 180


def rest_azimuth(D, ts):
    """azimuth [deg] of the unit x axis in the Mahony gravity frame (yaw 0) at ts, and the median |gyro| [deg/s] over the REST_S before ts"""
    x = IM.ahrs_base(D, np.array([ts]), ts, R.identity())[0].as_matrix()[:, 0]
    m = (D["imu_t"] >= ts - int(REST_S * 1e9)) & (D["imu_t"] <= ts)
    return float(np.degrees(np.arctan2(x[1], x[0]))), float(np.degrees(np.median(np.linalg.norm(D["gyro"][m] - D["b_rest"], axis=1)))) if m.any() else np.nan


def theta_hat(ai, ao):
    m = np.isfinite(ao[:, 0]) & np.isfinite(ai[:, 0]) & (np.linalg.norm(ao[:, :2], axis=1) > IM.E_MIN) & (np.linalg.norm(ai[:, :2], axis=1) > IM.E_MIN)
    if m.sum() < 15: return np.nan, int(m.sum()), np.nan
    x, y = ai[m, :2], ao[m, :2]; th = np.arctan2(np.sum(x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]), np.sum(x[:, 0] * y[:, 0] + x[:, 1] * y[:, 1]))
    c, s = np.cos(th), np.sin(th); xr = np.c_[c * x[:, 0] - s * x[:, 1], s * x[:, 0] + c * x[:, 1]]
    return float(np.degrees(th)), int(m.sum()), float(np.mean(np.linalg.norm(y - xr, axis=1)) / max(np.mean(np.linalg.norm(y, axis=1)), 1e-9))


def imu_acc(D, g):
    Rf = IM.ahrs_base(D, g, g[0], R.identity())             # tag-free D: anchor = yaw 0 -> the Mahony gravity frame itself
    f = np.stack([np.interp(g, D["imu_t"], D["acc"][:, k]) for k in range(3)], 1); fB = Rf.apply(f)
    return gaussian_filter1d(fB - np.median(fB, 0), IM.SIG_S * IM.HZ, axis=0, mode="nearest")


def free_lag(D, g, ao):
    m = np.isfinite(ao[:, 0]); B = np.linalg.norm(ao[m, :2], axis=1); B = B - B.mean(); best = (0, -9.0)
    for L in range(-500, 301, 20):
        a = np.linalg.norm(imu_acc(D, g + int(L * 1e6))[m, :2], axis=1); a = a - a.mean()
        cc = float((a * B).sum() / np.sqrt((a * a).sum() * (B * B).sum() + 1e-12))
        if cc > best[1]: best = (L, cc)
    return int(best[0] * 1e6), best[1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("take"); ap.add_argument("--track", default="trackv31speed"); ap.add_argument("--pack", default="")
    ap.add_argument("--calibrate-L", action="store_true"); ap.add_argument("--mode", default="L", choices=["L", "motion"])
    a = ap.parse_args(); take = a.take; f = FD.TAG / take / "tagfree.json"; tf = json.loads(f.read_text())
    tf["anchors"] = {}; info = {}; motion = {}; rest = {}
    for arm in ("left", "right"):
        D = FD.load_arm(take, arm, with_slam=False); C.FC.prep(D); D["b_rest"] = IM.rest_bias(D, take, FD)
        for W in FD.trims()[take]:
            dm = W["demo"]; ts = W["a0"] - int(0.05e9); rest.setdefault(arm, {})[dm] = (ts, W["a0"] + (W["w1"] - W["a0"]) // 2) + rest_azimuth(D, ts)
            if a.pack:
                d = Path(a.pack) / f"{take}_d{dm:02d}"
                if not (d / f"solve3_{arm}.npz").exists(): continue
                z = np.load(d / f"solve3_{arm}.npz"); tc = z["t"].astype(np.int64); cc = np.where((z["conf"] > 0)[:, None], z["c"], np.nan)
            else:
                z = np.load(HERE / "out" / f"{a.track}_{take}.npz"); tc = z["t_ns"].astype(np.int64); cc = np.where(z[f"valid_{arm}"].astype(bool)[:, None], z[f"c_{arm}"], np.nan)
            k = (tc >= W["w0"]) & (tc <= W["w1"]); tc, cc = tc[k], cc[k]
            if np.isfinite(cc[:, 0]).sum() < 30: continue
            g = IM.grid(tc[0], tc[-1]); ao = IM.track_acc(tc, cc, g); lag, ncc = free_lag(D, g, ao)
            ai = imu_acc(D, g + lag); th, n, res = theta_hat(ai, ao); h = g <= g[0] + (g[-1] - g[0]) // 2
            t1, _, _ = theta_hat(ai[h], ao[h]); t2, _, _ = theta_hat(ai[~h], ao[~h])
            if np.isfinite(th): motion.setdefault(arm, {})[dm] = th
            info.setdefault(arm, {})[str(dm)] = dict(n=n, resid=round(res, 3), half1=round(t1, 1), half2=round(t2, 1), lag_ms=lag / 1e6, ncc=round(ncc, 3))
            print(f"{take} d{dm} {arm[0]}: yaw {th:7.1f} deg  halves {t1:6.1f} / {t2:6.1f}  n {n} resid {res:.2f} lag {lag / 1e6:.0f} ms ncc {ncc:.2f}", flush=True)
    if a.calibrate_L:
        H = {}
        for arm in ("left", "right"):
            hk = {dm: motion[arm][dm] + rest[arm][dm][2] for dm in motion.get(arm, {})}; v = np.array(list(hk.values()))
            c = np.degrees(np.angle(np.median(np.exp(1j * np.radians(v))))) if len(v) else np.nan
            keep = [dm for dm in hk if abs(wrap(hk[dm] - c)) <= OUT_DEG]; c = np.degrees(np.angle(np.mean(np.exp(1j * np.radians([hk[d] for d in keep])))))
            H[arm] = dict(heading_deg=round(float(wrap(c - TAG_OFFSET[arm])), 2), motion_heading_deg=round(float(c), 2), tag_offset_deg=TAG_OFFSET[arm],
                          demos=keep, dropped={str(d): round(float(wrap(hk[d] - c)), 1) for d in hk if d not in keep}, spread_deg={str(d): round(float(wrap(hk[d] - c)), 1) for d in keep})
            print(f"[L] {arm}: H_L {H[arm]['heading_deg']} deg (motion {c:.1f}, tag offset {TAG_OFFSET[arm]}) from demos {keep}, spread {H[arm]['spread_deg']}, dropped {H[arm]['dropped']}")
        LH.write_text(json.dumps(dict(take=take, note="world heading of the unit x axis resting at its L tape (tag convention)", **H), indent=1))
    use_L = a.mode == "L" and LH.exists(); HL = json.loads(LH.read_text()) if use_L else None
    for arm in ("left", "right"):
        if not use_L:
            tf["anchors"][arm] = {str(dm): round(float(th), 2) for dm, th in motion.get(arm, {}).items()}; continue
        dms = sorted(rest[arm]); ts = np.array([rest[arm][d][0] for d in dms]); tm = np.array([rest[arm][d][1] for d in dms])
        psi = np.array([HL[arm]["heading_deg"] - rest[arm][d][2] for d in dms]); psi = np.degrees(np.unwrap(np.radians(psi)))
        still = np.array([np.isfinite(rest[arm][d][3]) and rest[arm][d][3] <= REST_DPS for d in dms]); ok = still.copy(); fit = None; x = (ts - ts[0]) / 60e9
        while ok.sum() >= 3:                       # drift line over the resting demos; drop the single worst point while it is > FIT_OUT_DEG off
            fit = np.polyfit(x[ok], psi[ok], 1); r = np.abs(psi - np.polyval(fit, x)); r[~ok] = -1; j = int(np.argmax(r))
            if r[j] <= FIT_OUT_DEG: break
            ok[j] = False; fit = None
        for i, d in enumerate(dms):
            mo = motion.get(arm, {}).get(d)
            if fit is not None: v = float(np.polyval(fit, (tm[i] - ts[0]) / 60e9)); src = "L-drift-line"
            elif still[i]: v = float(psi[i]); src = "L-own-rest"
            elif mo is not None: v = mo - TAG_OFFSET[arm]; src = "motion (not resting at the L)"
            else: continue
            tf["anchors"].setdefault(arm, {})[str(d)] = round(float(wrap(v)), 2)
            chk = float(wrap(v - (mo - TAG_OFFSET[arm]))) if mo is not None else np.nan
            info.setdefault(arm, {}).setdefault(str(d), {}).update(src=src, L_psi=round(float(wrap(psi[i])), 2), L_inlier=bool(ok[i]), rest_gyro_dps=round(rest[arm][d][3], 1),
                                                                    L_minus_motion_deg=round(chk, 1), flag=bool(np.isfinite(chk) and abs(chk) > CHECK_DEG))
            print(f"[L] {take} d{d} {arm[0]}: anchor {tf['anchors'][arm][str(d)]:7.1f}  ({src}) own-rest {wrap(psi[i]):7.1f} line-inlier {ok[i]} rest gyro {rest[arm][d][3]:.1f} dps  L-motion {chk:+.1f}")
        if fit is not None: info[arm]["drift_deg_per_min"] = round(float(fit[0]), 2)
    tf["yaw_info"] = info; tf["yaw_src"] = ("L+" if use_L else "") + (a.pack or a.track); f.write_text(json.dumps(tf, indent=1))


if __name__ == "__main__":
    main()
