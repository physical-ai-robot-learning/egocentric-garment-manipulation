"""10-23 orbbec_candidate_v0: continuous Orbbec + AHRS TCP trajectory per take/arm -> a SEPARATE derived root that the unchanged
production builder (laundry_dit_training/datasets/tag_rig_unsplit.py -> robotize_ik.py) turns into a candidate LeRobot dataset.

Per Orbbec frame (the umi_traj grid), per arm:
  position  ORBBEC  v2b track (out/trackv2b_<take>.npz: v1 dark detector, tag-free start-marker seed) -> TCP = c - R o
                    (o leave-one-demo-out, tags = calibration only), kept EXACTLY (no scale / world refit, no smoothing)
            gaps    shape = the production tag/IMU trajectory p_B (umi_traj, tag-anchored where tags are seen), shifted by a delta that
                    blends linearly from (orbbec - p_B) at the gap start to the gap end -> continuous at both ends (smooth rejoin, no snap)
                    every fill frame is INTERP (tag_in_gap flags gaps that contain >= 2-tag frames); TAG = tag measurement used directly
                    as the position (none in v0).  SLAM fill: not in v0.
            ends    before the first / after the last Orbbec sample of a demo: p_B + constant delta of the nearest sample (INTERP, unresolved)
  rotation  AHRS (rest-period gyro bias + accel tilt), one yaw anchor per demo window at its first >= 2-tag RAW PnP rotation (R_tag,
            calibration; the stored R_fused carries the old fgo/tag_rig_fuse gyro bias and is not used), IMU sampled at t + lag where lag =
            per-demo Orbbec<->IMU clock offset (xcorr of body vs IMU linear acceleration, -500..+300 ms; ncc < LAG_MIN_NCC -> 0 + flag)
  offset o  body centre -> TCP in the TCP frame, leave-one-demo-out on >= 2-tag frames with R_tag; per demo n / cond / norm / deviation
            from the pooled median are stored in cand_info
  anchor    if a demo has no >= 2-tag rotation, the nearest one of another demo is used (AHRS across the rest; flagged in cand_info)
  demos     no anchor at all or no Orbbec sample -> demo_valid False, conf LOW everywhere, production pose NOT used (pose = NaN-free
            copy kept only so the file loads; the demo must be excluded from the candidate build -- see candidate_demos.json)
  flags     src 0 ORBBEC | 1 TAG | 2 INTERP;  conf 2 HIGH (ORBBEC, npts >= HIGH_PTS) | 1 MID (other ORBBEC, INTERP <= 0.5 s) | 0 LOW
            fused_valid = ORBBEC only (the builder's 'measured');  gap_len_s = length of the gap containing the frame (0 on ORBBEC)
  conf file <conf_dir>/<take>/<arm>.npz {t_ns, conf} with HIGH 1.0 / MID 0.5 / LOW 0.0 -> builder --pos-conf-dir (variant imuconf)
Writes ONLY under --derived-root / --conf-dir.  Production derived dirs are read (assignment.json, segments.json, umi_traj) and copied.
    python build_candidate_v0.py --takes 20261023_123747 20261022_143329 20261022_160308 \
        --derived-root ~/laundry_dit_training/data/tag_rig_derived_orbbec_v0 --conf-dir ~/laundry_dit_training/data/orbbec_conf_v0"""
import argparse, json, shutil, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(1, str(HERE.parents[1]))
import core as C  # noqa
import imu_motion as IM  # noqa
FD = C.FD
PROD = Path.home() / "laundry_dit_training/data/tag_rig_derived"
HIGH_PTS, SHORT_S, UNRES_S, LAG_MIN_NCC = 120, 0.5, 1.0, 0.2
from scipy.ndimage import gaussian_filter1d
SRC = {"ORBBEC": 0, "TAG": 1, "INTERP": 2}


def runs(b):
    x = np.r_[0, b.astype(int), 0]; return list(zip(np.flatnonzero(np.diff(x) == 1), np.flatnonzero(np.diff(x) == -1)))


def build_arm(take, a, z, prod_traj):
    D = FD.load_arm(take, a, with_slam=False); C.FC.prep(D); D["b_rest"] = IM.rest_bias(D, take, FD)
    t = z["t_ns"]; n = len(t); assert np.array_equal(t, D["t"]) and np.array_equal(t, prod_traj["t_ns"])
    c = z["c_" + a]; npts = z["npts_" + a]; v = np.isfinite(c[:, 0]); Rb = D["R_tag"]; tg2 = (D["ntags"] >= 2) & (D["n_tagrot"] >= 2) & np.isfinite(Rb[:, 0, 0])
    trims = FD.trims()[take]; dem = np.full(n, -1)
    for W in trims: dem[(t >= W["a0"]) & (t <= W["w1"] + 2e9)] = W["demo"]
    p = prod_traj["p_B"].copy(); Rt = R.from_matrix(prod_traj["R_B_tcp"]); src = np.full(n, SRC["INTERP"]); conf = np.zeros(n, int)
    gap_len = np.zeros(n); unres = np.zeros(n, bool); tag_in_gap = np.zeros(n, bool); demo_valid = np.zeros(n, bool); info = []
    allo = []
    for W in trims:
        i = W["demo"]; k = np.flatnonzero(dem == i)
        tr = tg2 & v & (dem >= 0) & (dem != i); A = Rb[tr].reshape(-1, 3)
        o = np.linalg.lstsq(A, (c[tr] - D["p_tag"][tr]).reshape(-1), rcond=None)[0]; allo.append(o)
        oinfo = dict(n=int(tr.sum()), cond=round(float(np.linalg.cond(A)), 2), norm_mm=round(float(np.linalg.norm(o)) * 1000, 1))
        anc = np.flatnonzero((dem == i) & tg2); anc_other = False
        if not len(anc) and tg2.any():                                  # no anchor in this demo: nearest >= 2-tag rotation of another demo,
            j_ = np.flatnonzero(tg2); anc = j_[[np.argmin(np.abs(t[j_] - t[k[0]]))]]; anc_other = True   # AHRS propagates across the rest
        if not len(k) or not len(anc) or not v[k].any():
            conf[k] = 0; info.append(dict(demo=i, demo_valid=False, why="no >=2-tag anchor" if not len(anc) else "no orbbec sample", **oinfo)); continue
        lag, ncc = sync_lag(D, t, c, k, t[anc[0]], R.from_matrix(Rb[anc[0]]))
        Rk = IM.ahrs_base(D, t[k] + lag, t[anc[0]] + lag, R.from_matrix(Rb[anc[0]])); Rt_m = Rt.as_matrix(); Rt_m[k] = Rk.as_matrix(); Rt = R.from_matrix(Rt_m)
        vk = v[k]; po = np.full((len(k), 3), np.nan); po[vk] = c[k][vk] - Rk[vk].apply(o)
        pk = p[k].copy(); sk = np.full(len(k), SRC["INTERP"]); ck = np.zeros(len(k), int); gk = np.zeros(len(k)); uk = np.zeros(len(k), bool)
        pk[vk] = po[vk]; sk[vk] = SRC["ORBBEC"]; ck[vk] = np.where(npts[k][vk] >= HIGH_PTS, 2, 1)
        iv = np.flatnonzero(vk)
        if len(iv) == 0: info.append(dict(demo=i, skipped="no orbbec")); continue
        for s_, e_ in runs(~vk):
            L = (t[k][min(e_, len(k) - 1)] - t[k][max(s_ - 1, 0)]) / 1e9; gk[s_:e_] = L
            base = p[k][s_:e_]
            if s_ > 0 and e_ < len(k):                                   # interior gap: linear delta blend, continuous at both ends
                d0 = po[s_ - 1] - p[k][s_ - 1]; d1 = po[e_] - p[k][e_]; w = ((t[k][s_:e_] - t[k][s_ - 1]) / max(t[k][e_] - t[k][s_ - 1], 1))[:, None]
                pk[s_:e_] = base + (1 - w) * d0 + w * d1
                sk[s_:e_] = SRC["INTERP"]; tag_in_gap[k[s_:e_]] = tg2[k][s_:e_].any()
                ck[s_:e_] = 1 if L <= SHORT_S else 0; uk[s_:e_] = L > UNRES_S
            else:                                                       # before the first / after the last Orbbec sample
                j = iv[0] if s_ == 0 else iv[-1]; pk[s_:e_] = base + (po[j] - p[k][j]); ck[s_:e_] = 0; uk[s_:e_] = True; gk[s_:e_] = 9.9
        p[k] = pk; src[k] = sk; conf[k] = ck; gap_len[k] = gk; unres[k] = uk; demo_valid[k] = True
        info.append(dict(demo=i, demo_valid=True, offset_mm=(o * 1000).round(1).tolist(), lag_ms=round(lag / 1e6), lag_ncc=round(ncc, 2), lag_flag=ncc < LAG_MIN_NCC, anchor_from_other_demo=anc_other,
                         anchor_dt_s=round(float(t[k[0]] - t[anc[0]]) / 1e9, 1), **oinfo))
    med = np.median(np.array(allo), 0)
    for r_, o_ in zip(info, allo): r_["offset_dev_from_pooled_mm"] = round(float(np.linalg.norm(o_ - med)) * 1000, 1)
    fused_valid = (src == SRC["ORBBEC"]) & demo_valid
    out = dict(prod_traj); out.update(p_B=p, R_B_tcp=Rt.as_matrix(), fused_valid=fused_valid, gap_len_s=gap_len, src=src, conf=conf, unresolved=unres,
                                     tag_in_gap=tag_in_gap, demo_valid=demo_valid,
                                     p_orbbec_body=c, npts=npts, p_B_prod=prod_traj["p_B"], R_B_tcp_prod=prod_traj["R_B_tcp"],
                                     pose_src=np.array("orbbec_candidate_v0"), cand_info=np.array(json.dumps(info)))
    return out, conf, info


def sync_lag(D, t, c, k, anc_t, anc_R):
    """per-demo Orbbec<->IMU clock offset (ns): IMU sampled at t + lag best matches the Orbbec body acceleration (low-passed ncc)"""
    g = IM.grid(t[k][0], t[k][-1]); ao = IM.track_acc(t[k], c[k], g); m = np.isfinite(ao[:, 0])
    if m.sum() < 30: return 0, 0.0
    best = (0, -9.0)
    for L in range(-500, 301, 20):
        gg = g + int(L * 1e6); Rt = IM.ahrs_base(D, gg, anc_t + int(L * 1e6), anc_R)
        f = np.stack([np.interp(gg, D["imu_t"], D["acc"][:, j]) for j in range(3)], 1); ai = Rt.apply(f); ai[:, 2] -= np.median(np.linalg.norm(f, axis=1))
        ai = gaussian_filter1d(ai, IM.SIG_S * IM.HZ, axis=0, mode="nearest"); A, B = ao[m], ai[m]
        cc = float((A * B).sum() / np.sqrt((A * A).sum() * (B * B).sum() + 1e-12))
        if cc > best[1]: best = (L, cc)
    return (int(best[0] * 1e6), best[1]) if best[1] >= LAG_MIN_NCC else (0, best[1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--takes", nargs="+", required=True); ap.add_argument("--derived-root", required=True); ap.add_argument("--conf-dir", required=True)
    ap.add_argument("--track", default="trackv2b"); a = ap.parse_args(); infos = {}
    for take in a.takes:
        src_dir = PROD / f"{take}_fused" if (PROD / f"{take}_fused").exists() else PROD / take
        dst = Path(a.derived_root).expanduser() / f"{take}_fused"; dst.mkdir(parents=True, exist_ok=True)
        asg = json.loads((src_dir / "assignment.json").read_text()); asg["pose_src"] = "orbbec_candidate_v0"; asg["candidate_from"] = str(src_dir)
        (dst / "assignment.json").write_text(json.dumps(asg, indent=1, ensure_ascii=False)); shutil.copy(src_dir / "segments.json", dst / "segments.json")
        z = np.load(HERE / f"out/{a.track}_{take}.npz")
        for arm in ("left", "right"):
            prod = dict(np.load(src_dir / f"umi_traj_{arm}.npz", allow_pickle=True))
            assert str(prod["source_rig_label"]) == asg["rig_of_arm"][arm]
            out, conf, info = build_arm(take, arm, z, prod); np.savez(dst / f"umi_traj_{arm}.npz", **out); infos.setdefault(take, {})[arm] = info
            cd = Path(a.conf_dir).expanduser() / take; cd.mkdir(parents=True, exist_ok=True)
            np.savez(cd / f"{arm}.npz", t_ns=out["t_ns"], conf=np.array([0.0, 0.5, 1.0])[conf])
            print(take, arm, "src ORBBEC/TAG/INTERP %", [round(float((out["src"] == s).mean()) * 100, 1) for s in (0, 1, 2)],
                  "invalid demos", [r["demo"] for r in info if not r["demo_valid"]], "lags", [r.get("lag_ms") for r in info], flush=True)
    root = Path(a.derived_root).expanduser(); f = root / "candidate_info.json"; old = json.loads(f.read_text()) if f.exists() else {}
    old.update(infos); f.write_text(json.dumps(old, indent=1))


if __name__ == "__main__":
    main()
