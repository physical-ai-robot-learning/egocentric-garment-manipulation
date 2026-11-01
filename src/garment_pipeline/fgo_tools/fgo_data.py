"""10-23 FGO data access: per take / physical arm -> orbbec-grid tag + fused data, raw tag rotation, IMU (TCP frame, orbbec clock),
raw MASt3R segments (camera poses on the wrist clock), trim windows."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

from egogarment import paths as egopaths  # noqa: E402
EGO = egopaths.DATA_ROOT; TAG = egopaths.TAG_RIG; DER = egopaths.DERIVED
M3 = EGO / "datasets/m3_runs"; FR = EGO / "datasets/m3_frames"; FGO = M3 / "_fgo_1023"
TAKES = ["20261022_121938", "20261022_123614", "20261022_124754", "20261022_130007", "20261022_134315", "20261022_143329",
         "20261022_144321", "20261022_160308", "20261022_161756", "20261022_170651", "20261023_113550", "20261023_114817", "20261023_123747"]
RIGNAME = {"right": "UMI 오른손", "left": "UMI 왼손"}
SLAM_DT = 20e6     # chain_v1 convention: SLAM pose at orbbec time t is the wrist pose at t + 20 ms


ARCHIVE = egopaths.CKPT_ROOT / "tag_rig_raw"


def trims():
    out = {}
    for r in csv.DictReader(open(egopaths.IDLE_TRIM_CSV)):
        out.setdefault(r["take"], []).append(r)
    res = {}
    for take, rows in out.items():
        mf = TAG / take / "meta.json"
        if not mf.exists(): mf = ARCHIVE / take / "meta.json"   # 10-26: raw takes 10-16..10-23 moved to the PortableSSD
        if not mf.exists(): continue
        m = json.loads(mf.read_text()); t0 = int(m["t_start_monotonic_ns"])
        P = [p for p in m["protocol"] if p["text"].startswith("자유:")]
        res[take] = [dict(demo=int(r["demo"]), a0=t0 + int(P[int(r["demo"])]["t0"] * 1e9), w0=t0 + int((P[int(r["demo"])]["t0"] + float(r["keep_start_s"])) * 1e9),
                          w1=t0 + int((P[int(r["demo"])]["t0"] + float(r["keep_end_s"])) * 1e9), kept_s=float(r["kept_s"])) for r in rows]
    return res


def assignment(take):
    aj = DER / f"{take}_fused/assignment.json"
    if aj.exists():
        a = json.loads(aj.read_text()); return a["rig_of_arm"], a["cam_of_arm"], True
    ym = {s: float(np.median((z := np.load(TAG / take / f"umi_traj_{s}.npz"))["p_B"][z["ntags"] > 0, 1])) for s in ("left", "right")}
    rig_of = {("left" if ym[s] > 0 else "right"): s for s in ym}
    assert set(rig_of) == {"left", "right"}, ym
    return rig_of, {"left": "wrist_left", "right": "wrist"}, False


def load_segments(manifest, frames_json):
    fr = {f["i"]: f["t_ns"] for f in json.loads(Path(frames_json).read_text())["frames"]}; segs = []
    for m in json.loads(Path(manifest).read_text())["segments"]:
        p = Path(manifest).parent / Path(m["csv"]).name
        if not p.exists() or m["cut"] < 10: continue
        T, P, Q, S = [], [], [], []
        for r in csv.DictReader(open(p)):
            k = int(r["frame_idx"])
            if k >= m["cut"] or r["state"] not in ("INIT", "TRACKING"): continue
            q = [float(r[c]) for c in ("q_x", "q_y", "q_z", "q_w")]; xyz = [float(r["x"]), float(r["y"]), float(r["z"])]
            if not np.all(np.isfinite(q + xyz)) or np.linalg.norm(q) < 1e-3 or (m["start"] + k) not in fr: continue
            T.append(fr[m["start"] + k]); P.append(xyz); Q.append(q)
        if len(T) >= 10:
            t = np.array(T, np.int64); o = np.argsort(t, kind="stable"); t, keep = np.unique(t[o], return_index=True); o = o[keep]
            segs.append(dict(t=t - int(SLAM_DT), p=np.array(P)[o], R=R.from_quat(np.array(Q)[o]), start=m["start"]))
    return segs


def load_arm(take, arm, with_slam=True):
    rig_of, cam_of, fused = assignment(take); rig = rig_of[arm]; cam = cam_of[arm]
    u = dict(np.load(DER / f"{take}_fused/umi_traj_{arm}.npz", allow_pickle=True)) if fused else dict(np.load(TAG / take / f"umi_traj_{rig}.npz", allow_pickle=True))
    t = u["t_ns"].astype(np.int64); nt = u["ntags"]
    D = dict(take=take, arm=arm, rig=rig, cam=cam, raw=u, t=t, ntags=nt, p_tag=u.get("p_B_tag", u["p_B"]), p_fused=u["p_B"], R_fused=u["R_B_tcp"],
             slam_ok=u.get("slam_ok", np.zeros(len(t), bool)).astype(bool), fused_valid=u.get("fused_valid", nt > 0).astype(bool))
    # raw tag rotation (PnP only) on the same orbbec frames
    tr = np.load(FGO / "tagrot" / f"{take}.npz"); assert np.array_equal(tr["t_ns"].astype(np.int64), t) or len(tr["t_ns"]) == len(t)
    D["R_tag"] = tr[f"R_{rig}"]; D["n_tagrot"] = tr[f"n_{rig}"]
    # IMU on the orbbec clock (same mapping as tag_rig_fuse), expressed in the TCP (= rig) frame
    from handumi_collector.pose.timing import fit_device_to_host
    cfg = json.loads((EGO / "configs/calibration/umi_tag_rig_v001.json").read_text())[RIGNAME[rig]]
    z = np.load(TAG / take / f"imu_{rig}.npz"); RI = np.array(cfg["imu"]["R_I_rig"])
    ti = fit_device_to_host(z["device_us"], z["host_ns"]).to_host_ns(z["device_us"]).astype(np.int64) + int(cfg["imu"]["offset_ms"] * 1e6)
    o = np.argsort(ti, kind="stable"); ti = ti[o]
    D["imu_t"] = ti; D["gyro"] = z["gyro"][o].astype(float) @ RI; D["acc"] = z["acc"][o].astype(float) @ RI      # v_rig = RI^T v_I
    D["segs"] = []; D["no_slam"] = True
    tag = f"umi_{take}" + ("" if cam == "wrist" else "_left")
    # hand-eye always set (10-23 fix): own chain_v1 fit if fused, else the 10-22 mean for that camera (needed by any visual front-end, set_visual)
    if fused: D["RX"] = R.from_matrix(u["handeye_R"]); D["tX"] = np.array(u["handeye_t"]); D["handeye_src"] = "own_fused"
    else: D["RX"], D["tX"] = handeye_1022(cam); D["handeye_src"] = "mean_1022_" + cam
    if with_slam and slam_ready(take, cam):
        D["segs"] = load_segments(M3 / tag / "segs/manifest.json", FR / tag / "frames.json"); D["no_slam"] = len(D["segs"]) == 0
    return D


def slam_ready(take, cam):
    """merged MASt3R manifest present and (for service-processed takes) all windows collected"""
    tag = f"umi_{take}" + ("" if cam == "wrist" else "_left"); mf = M3 / tag / "segs/manifest.json"
    if not mf.exists() or not (FR / tag / "frames.json").exists(): return False
    m = json.loads(mf.read_text()); return not m.get("missing_windows")


def handeye_1022(cam):
    """10-23 takes have no chain_v1 fusion: camera->TCP hand-eye = rotation mean / translation median over the 10-22 fused arms with that camera"""
    Rs, ts = [], []
    for take in TAKES:
        aj = DER / f"{take}_fused/assignment.json"
        if not aj.exists(): continue
        a = json.loads(aj.read_text())
        for arm, c in a["cam_of_arm"].items():
            if c == cam:
                u = np.load(DER / f"{take}_fused/umi_traj_{arm}.npz"); Rs.append(u["handeye_R"]); ts.append(u["handeye_t"])
    return R.from_matrix(np.array(Rs)).mean(), np.median(np.array(ts), 0)


def set_visual(D, npz, time_offset_ns=0, min_conf=None):
    """replace D["segs"] by a generic visual front-end: npz with t_ns (n,) on the WRIST-camera clock (as m3_frames/frames.json; the -20 ms
    SLAM_DT is applied here like for MASt3R), p (n,3) camera centre in the segment world, q_xyzw (n,4) camera->world rotation, segment_id (n,),
    optional conf (n,) (frames with conf < min_conf dropped). Scale/world of every segment are free (estimated by the graph)."""
    z = np.load(npz); t = z["t_ns"].astype(np.int64) - int(SLAM_DT) + int(time_offset_ns); sid = z["segment_id"]; keep = np.ones(len(t), bool)
    if min_conf is not None and "conf" in z.files: keep &= z["conf"] >= min_conf
    segs = []
    for k in np.unique(sid):
        m = keep & (sid == k)
        if m.sum() < 10: continue
        o = np.argsort(t[m]); tt, u = np.unique(t[m][o], return_index=True); o = o[u]
        segs.append(dict(t=tt, p=z["p"][m][o], R=R.from_quat(z["q_xyzw"][m][o]), start=int(k)))
    if "RX" not in D: raise RuntimeError("set_visual: D has no hand-eye RX/tX (load with fgo_data.load_arm)")
    D["segs"] = segs; D["no_slam"] = len(segs) == 0; return D


def export_visual(D, npz):
    """write D["segs"] (e.g. MASt3R) in the generic format above (wrist clock)"""
    t = np.concatenate([s["t"] + int(SLAM_DT) for s in D["segs"]]); p = np.concatenate([s["p"] for s in D["segs"]])
    q = np.concatenate([s["R"].as_quat() for s in D["segs"]]); sid = np.concatenate([np.full(len(s["t"]), i) for i, s in enumerate(D["segs"])])
    np.savez(npz, t_ns=t, p=p, q_xyzw=q, segment_id=sid)
