#!/usr/bin/env python3
"""[2026-10-23] Tag-rig laundry dataset v2: ONE demo = ONE unsplit LeRobot episode over the idle-trimmed window, with PER-ARM, PER-STEP loss
masks instead of cutting (user requirement 10-23; v1 kept only frames where BOTH arms were valid and split at every gap -> 2-6 s pieces).

Window   reports/idle_trim.csv (canonical, coordinator 10-23): demo start (`자유:` cue) + [keep_start_s, keep_end_s]
Pose     robotized ik_v1 (datasets/robotize_ik.py) over the whole window: fused / tag position where measured, the tracker's interpolated
         position + IMU-filled rotation elsewhere (umi_traj --imu-fill), then IK.  Every frame stays in sequence.
Per frame, per arm (stored):
  measured[2]   the pose sample is measured (fused_valid, or ntags > 0 for tag-only) -- else filled
  fill_long[2]  inside a filled run longer than FILL_MASK_S (0.5 s)  -> never a training target (IMU shows real motion hidden there)
  anchor_ok[4]  [pos_ok_L, rot_ok_L, pos_ok_R, rot_ok_R]: pos_ok = not in a fill > 0.5 s AND IK <= 10 mm AND grasp gate ok;
                rot_ok = IK <= 10 mm AND grasp gate ok (IMU-filled rotation stays valid in pose gaps -- 10-23 mask spec)
  action_mask[20]  validity of the stored absolute target (= frame i+1): xyz dims = pos_ok, rot6d dims = rot_ok, gripper = jaw label valid
  The training glue (training/tagrig_common.py) builds the chunk mask from frames t..t+31 (ACT) / t-1..t+31 (DiT) and the anchor's pose_ok.
  The state stays filled (no mask).
  NOTE (11-01 audit): the IK terms above apply only to --label ikfk.  For --label rob / fused (labels A / F) the IK error and
  the grasp displacement are set to 0 (see `if LABEL in ("rob", "fused")` below), so --ik-fail-mm has no effect: pos_ok /
  rot_ok are all 1 in stitched mode and come from fill length only in gated mode.  A demo without an ik_v1 file is not built.
Episode drop only if: measured < 50 % for BOTH arms after the trim, or a filled run > 5 s overlaps a grasp window (+-0.5 s of a close).
Split: frozen by demo -- demos of the v1 split keep their split; new eligible demos get a stratified, seeded 85/8/7 assignment.
usage: tag_rig_unsplit.py --out data/lerobot/tagrig_laundry_v2u --takes TAKE... [--old-split .../meta/split.json] [--render 4] [--cam-shift-ms 0]"""
import argparse, csv, json, pathlib, shutil, sys, time

import cv2, numpy as np

HERE = pathlib.Path(__file__).resolve().parent
from garment_pipeline.labels_dataset import tag_rig_to_lerobot as T2L  # noqa: E402
from xvla_core import abs_action_rows, nearest_frame, resample_pose  # noqa: E402
from umi_to_lerobot import ABS_NAMES, proprio_rows_lr  # noqa: E402
from contracts.action_contract import ACTION_DT_S, T_to_pose9, make_T  # noqa: E402
from contracts.observation_contract import PROPRIO_DIM, PROPRIO_NAMES  # noqa: E402

EXCLUDE_TAKES = {"20261023_113219", "20261023_115848", "20261023_112832", "20261023_113038"}   # user 10-23: never use (aborted / empty)
FILL_MASK_S, IK_FAIL_MM, GRASP_OK_MM, GRASP_WARN_MM, GRASP_W = 0.5, 10.0, 20.0, 50.0, 0.5
DROP_MEASURED, DROP_FILL_IN_GRASP_S = 0.5, 5.0
VARIANTS = ["strict", "relaxed_2s", "weighted"]   # tag-gap reference baselines; "imuconf" is appended when --pos-conf-dir is given (MAIN track)
MASK_MODE = "gated"   # "stitched" (10-23 user GO): position weight 1.0 everywhere on the continuous stitched trajectory; hard masks only
CONF = dict(dir=None, high=None, mid=None, w_mid=0.3)
LABEL = "ikfk"   # 10-26 contract: "rob" = action_cartesian_robotized (robotize_ik <arm>_p/R_rob, before IK; IK residual is QA only, never a mask)
                 #                 "ikfk" = old label FK(q) of the offline IK path (<arm>_p/R), kept for the A-vs-B comparison   # 10-23 user rule: position weight from the marker-free SLAM<->IMU consistency score


# 10-31 jaw relabel: per-arm jaw label file (default umi_jaw.npz). JAW_FILE_LEFT=umi_jaw_motorlike.npz -> the crop->servo relabel for the
# left hand only (user review: left OK, right wrong on black cloth); its `valid` (0 = UNKNOWN) masks the BCE jaw loss like any jaw gap.
JAW_FILE = {"left": __import__("os").environ.get("JAW_FILE_LEFT", "umi_jaw.npz"), "right": __import__("os").environ.get("JAW_FILE_RIGHT", "umi_jaw.npz")}

def load_pos_conf(take, rig, arm):
    """per-frame 0..1 SLAM<->IMU consistency for one UMI -> (t_ns, conf). Accepts <dir>/<take>/<rig|arm>.npz, <dir>/<take>_<rig|arm>.npz or
    <dir>/<take>/umi_conf_<rig|arm>.npz with keys t_ns + one of conf / score / confidence (format confirmed when the files land)"""
    d = pathlib.Path(CONF["dir"])
    for name in (rig, arm, f"umi_conf_{rig}", f"umi_conf_{arm}"):
        for f in (d / take / f"{name}.npz", d / f"{take}_{name}.npz"):
            if f.exists():
                z = np.load(f); k = next(k for k in ("conf", "score", "confidence") if k in z.files)
                return z["t_ns"].astype(np.int64), z[k].astype(float)
    raise FileNotFoundError(f"no SLAM<->IMU confidence for {take} {rig}/{arm} under {d}")


def conf_weight(c):
    """High (>= high) 1.0 | Mid (>= mid) w_mid | Low masked; thresholds from the calibration (--conf-high / --conf-mid)"""
    return np.where(c >= CONF["high"], 1.0, np.where(c >= CONF["mid"], CONF["w_mid"], 0.0))   # 10-23: position-weight variants A/B/C (same frames, same split; selected at train time)
SPEED = 1.0          # 10-29 (user: demos too fast for the reBot): playback factor. Output rows stay 20 Hz (ACTION_DT_S) but advance DT_SRC =
                     # ACTION_DT_S * SPEED of SOURCE time, so pose, action, jaw edges and EVERY camera frame (nearest source frame at the warped time)
                     # share one stretched timeline; 0.3 = 3.33x slower. Source-time windows (fill, grasp, jaw ramp) keep their real-time meaning.
JAW_WIN_S = 0.5                                     # grasp window = +-0.5 s around jaw TRANSITIONS (open and close edges)


def pos_weights(gl, fgo_valid, jw):
    """per-frame position loss weight per variant from the fgo gap length (s; 0 on tag frames) -> dict variant -> (n,)"""
    strict = fgo_valid.astype(float)                                       # tag | gap <= 1.0 s with SLAM | <= 0.5 s without (fgo valid_rule)
    relaxed = (gl <= 2.0).astype(float)
    w = np.where(gl <= 0.5, 1.0, np.where(gl <= 1.0, 0.7, np.where(gl <= 2.0, 0.25, 0.0)))
    w = np.where(jw & (gl > 1.0), 0.0, w)                                  # contact rule: no long-gap positions around jaw events
    return dict(strict=strict, relaxed_2s=relaxed, weighted=w)
CAMS = T2L.CAMS; IMG_WH = T2L.IMG_WH
TRIM_CSV = HERE.parent / "reports/idle_trim.csv"


def runs(b):
    x = np.r_[0, b.astype(int), 0]; return list(zip(np.flatnonzero(np.diff(x) == 1), np.flatnonzero(np.diff(x) == -1)))


def measured_flag(z):
    if "fused_valid" in z.files: return z["fused_valid"].astype(bool)
    if "slam_ok" in z.files: return z["slam_ok"].astype(bool) | (z["ntags"] > 0)
    return z["ntags"] > 0


def build(take, dtake, asg, demo, win, cam_shift_ns=0):
    IK = np.load(dtake / "ik_v1" / f"demo{demo['idx']:02d}.npz")
    tc = np.load(take / "t_ns.npy").astype(np.int64) + cam_shift_ns; ti = IK["t_ns"].astype(np.int64) + cam_shift_ns
    lo = max(win[0], int(ti[0]), int(tc[0])); hi = min(win[1], int(ti[-1]), int(tc[-1]))
    if hi - lo < 5e9: raise ValueError(f"trimmed window inside the recording only {(hi - lo) / 1e9:.1f} s")
    DT_SRC = ACTION_DT_S * SPEED
    g_ns = lo + np.round(np.arange(int((hi - lo) / 1e9 / DT_SRC) + 1) * DT_SRC * 1e9).astype(np.int64); n = len(g_ns); tg = (g_ns - lo) / 1e9
    tis = (ti - lo) / 1e9; ki, _ = nearest_frame(tis, tg); out = dict(t_ns=g_ns, n=n, gap_m=IK["gap"][ki])
    for a, s in (("left", "L"), ("right", "R")):
        z = np.load(dtake / f"umi_traj_{a}.npz", allow_pickle=True); tz = (z["t_ns"].astype(np.int64) + cam_shift_ns - lo) / 1e9
        sfx = {"rob": "_rob", "fused": "_fused"}.get(LABEL, "")   # 10-29 fused = the human TCP, NO table-clearance lift (rob lifted 44 % of frames by ~27 mm)
        T, _, _, _ = resample_pose(tis, make_T(IK[f"{a}_R{sfx}"], IK[f"{a}_p{sfx}"]), np.zeros(len(tis)), tg, max_gap=1.0)
        Ph = np.stack([np.interp(tg, tis, IK[f"{a}_p_h"][:, j]) for j in range(3)], 1)
        kz, rz = nearest_frame(tz, tg); meas = measured_flag(z)[kz] & (rz <= 0.07)
        gl = (z["gap_len_s"][kz] if "gap_len_s" in z.files else np.where(meas, 0.0, 9.9)).astype(float); gl = np.where(rz <= 0.07, gl, 9.9)
        fl = np.zeros(n, bool)
        for x, y in runs(~meas):
            if (y - x) * DT_SRC > FILL_MASK_S: fl[x:y] = True
        ikmm = IK[f"{a}_ik_mm"][ki]; disp = np.linalg.norm(T[:, :3, 3] - Ph, axis=1) * 1000
        if LABEL in ("rob", "fused"): ikmm = np.zeros(n); disp = np.zeros(n)   # A: no IK mask / grasp-displacement gate (IK stats live in ik_v1 + ik_qa_stats.py)
        J0 = np.load(take / asg["cam_of_arm"][a] / JAW_FILE[a]); sb = J0["state"][nearest_frame((J0["t_ns"].astype(np.int64) - lo) / 1e9, tg)[0]].astype(int)
        grasp = np.zeros(n, bool); hw = int(JAW_WIN_S / DT_SRC)
        for c in np.flatnonzero(np.diff(sb) != 0) + 1: grasp[max(c - hw, 0): c + hw + 1] = True       # both edges (open + close)
        gfail = np.zeros(n, bool); gw = np.ones(n, np.float32); info = []
        for x, y in runs(grasp):
            mx = float(disp[x:y].max()); c = (x + y) // 2                       # the jaw-close instant (window = +-0.5 s around it)
            longest = max([(e - b) * DT_SRC for b, e in runs(~meas) if b <= c < e] or [0.0])   # filled run CONTAINING the close
            info.append(dict(max_mm=round(mx), fill_overlap_s=round(longest, 2)))
            if mx > GRASP_WARN_MM: gfail[x:y] = True
            elif mx >= GRASP_OK_MM: gw[x:y] = GRASP_W
        J = np.load(take / asg["cam_of_arm"][a] / JAW_FILE[a]); tj = (J["t_ns"].astype(np.int64) - lo) / 1e9; kj, rj = nearest_frame(tj, tg)
        fine = tg[:, None] + np.linspace(-T2L.GRIP_RAMP_S / 2, T2L.GRIP_RAMP_S / 2, 21)[None]
        g = J["state"][nearest_frame(tj, fine.ravel())[0]].reshape(fine.shape).astype(float).mean(1)
        jv = (J["valid"][kj] > 0) & (rj <= T2L.MAX_JAW_GAP_S)
        ok_ik = (ikmm <= IK_FAIL_MM) & (~gfail)
        pw_all = pos_weights(gl, meas, grasp)
        if CONF["dir"]:                                                   # MAIN track: marker-free consistency confidence -> position weight
            tcf, cf = load_pos_conf(take.name, str(z["source_rig_label"]), a); kc, rc = nearest_frame((tcf + cam_shift_ns - lo) / 1e9, tg)
            pw_all["imuconf"] = np.where(rc <= 0.07, conf_weight(cf[kc]), 0.0)
        pwv = {v: w * ok_ik for v, w in pw_all.items()}
        out.update({f"T_{a}": T, f"Ph_{a}": Ph, f"meas_{s}": meas, f"fill_{s}": fl, f"ikmm_{s}": ikmm, f"grasp_{s}": grasp, f"gfail_{s}": gfail, f"gw_{s}": gw,
                    f"ikwarn_{s}": (ikmm > IK_FAIL_MM) & (ikmm <= 50.0), f"iksev_{s}": ikmm > 50.0,
                    f"g_{a}": g, f"gc_{a}": np.clip(np.interp(tg, tj, J["g_cont"].astype(float)), 0, 1), f"jv_{s}": jv, f"ginfo_{s}": info, f"disp_{s}": disp,
                    f"rot_ok_{s}": (ikmm <= IK_FAIL_MM) & ((~gfail) if MASK_MODE != "stitched" else True),   # IMU-filled rotation stays a target
                    f"pos_ok_{s}": ((~fl) & (ikmm <= IK_FAIL_MM) & (~gfail)) if MASK_MODE != "stitched" else (ikmm <= IK_FAIL_MM),   # stitched: IK only
                    f"gl_{s}": gl, f"conf_{s}": (gl <= 0.5) & (rz <= 0.07), **{f"pw_{v}_{s}": pwv[v] for v in VARIANTS}})
    for c, tt in (("global", tc), ("left_wrist", np.load(take / asg["cam_of_arm"]["left"] / "t_ns.npy").astype(np.int64)),
                  ("right_wrist", np.load(take / asg["cam_of_arm"]["right"] / "t_ns.npy").astype(np.int64))):
        idx, res = nearest_frame(tt.astype(np.float64), g_ns.astype(np.float64)); out[f"img_{c}"] = idx; out[f"res_{c}"] = res / 1e6
    out["cam_dir"] = {"global": take, "left_wrist": take / asg["cam_of_arm"]["left"], "right_wrist": take / asg["cam_of_arm"]["right"]}
    nx = np.minimum(np.arange(n) + 1, n - 1)
    # 10-23 spec: gaps mask only POSITION (xyz); rotation (IMU) and gripper (wrist-camera jaw) stay supervised
    out["action_mask"] = np.concatenate([np.repeat(out["pos_ok_L"][nx, None], 3, 1), np.repeat(out["rot_ok_L"][nx, None], 6, 1),
                                         np.repeat(out["pos_ok_R"][nx, None], 3, 1), np.repeat(out["rot_ok_R"][nx, None], 6, 1),
                                         out["jv_L"][nx, None], out["jv_R"][nx, None]], 1).astype(np.float32)
    out["anchor_ok"] = np.stack([out["pos_ok_L"], out["rot_ok_L"], out["pos_ok_R"], out["rot_ok_R"]], 1).astype(np.float32)
    for v in VARIANTS:
        out[f"aw_{v}"] = np.concatenate([np.repeat(out[f"pw_{v}_L"][nx, None], 3, 1), np.repeat(out["rot_ok_L"][nx, None], 6, 1),
                                         np.repeat(out[f"pw_{v}_R"][nx, None], 3, 1), np.repeat(out["rot_ok_R"][nx, None], 6, 1),
                                         out["jv_L"][nx, None], out["jv_R"][nx, None]], 1).astype(np.float32)
        out[f"anw_{v}"] = np.stack([out[f"pw_{v}_L"], out["rot_ok_L"], out[f"pw_{v}_R"], out["rot_ok_R"]], 1).astype(np.float32)
    out["weight"] = np.minimum(out["gw_L"], out["gw_R"]) if MASK_MODE != "stitched" else np.ones(n, np.float32)
    # 10-30: EP_WEIGHT env = per-sample loss weight of every frame of this build (new motor-jaw flattening mixed into GOLD-281 at 2x)
    out["weight"] = (out["weight"] * float(__import__("os").environ.get("EP_WEIGHT", "1"))).astype(np.float32)
    return out


def render(b, path, title):
    n = b["n"]; vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (960, 330))
    bars = {s: np.stack([b[f"pos_ok_{s}"], b[f"rot_ok_{s}"], b[f"jv_{s}"]], 0) for s in "LR"}
    for i in range(n):
        ims = [cv2.resize(cv2.imread(str(b["cam_dir"][c] / (f"color_{int(b[f'img_{c}'][i]):05d}.jpg" if c == "global" else f"wrist_{int(b[f'img_{c}'][i]):05d}.jpg"))), (320, 180))
               for c in ("global", "left_wrist", "right_wrist")]
        panel = np.full((150, 960, 3), 25, np.uint8); y0 = 30
        for s, lab in (("L", "LEFT"), ("R", "RIGHT")):
            for r, nm in ((0, "pos"), (1, "rot"), (2, "grip")):
                ok = bars[s][r]; xs = (np.arange(n) / max(n - 1, 1) * 900 + 50).astype(int)
                for j in range(n - 1):
                    col = (60, 200, 60) if ok[j] else ((130, 130, 130) if b[f"meas_{s}"][j] or nm != "pos" else (90, 90, 90))
                    cv2.line(panel, (xs[j], y0), (xs[j + 1], y0), col, 10)
                cv2.putText(panel, f"{lab} {nm}", (2, y0 + 4), 0, 0.35, (230, 230, 230), 1); y0 += 17
        cx = int(i / max(n - 1, 1) * 900 + 50); cv2.line(panel, (cx, 15), (cx, 135), (255, 255, 255), 1)
        cv2.putText(panel, f"{title}  t={i * ACTION_DT_S:5.1f}s  green = in loss, gray = masked (dark = filled pose)", (50, 145), 0, 0.4, (230, 230, 230), 1)
        vw.write(np.vstack([np.hstack(ims), panel]))
    vw.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--takes", nargs="+", required=True); ap.add_argument("--derived", default=str(HERE.parent / "data/tag_rig_derived"))
    ap.add_argument("--old-split", default=str(HERE.parent / "data/lerobot/tagrig_laundry_v1_fused_train/meta/split.json"))
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--render", type=int, default=4); ap.add_argument("--render-dir", default=str(HERE.parent / "reports/tagrig_qa/unsplit_videos"))
    ap.add_argument("--cam-shift-ms", type=float, default=0.0, help="shift Orbbec (pose + colour) time; coordinator 10-23: ~-40 ms vs IMU, OFF until confirmed")
    ap.add_argument("--pose-src", default="auto", choices=["auto", "_fused", "_fgo", ""], help="auto = _fused when present else tag-only; _fgo = fgo_v0 (10-23)")
    ap.add_argument("--pos-conf-dir", default="", help="per-frame SLAM<->IMU consistency files (~/ego_collector/datasets/m3_runs/_imu_consistency/) -> variant imuconf")
    ap.add_argument("--conf-high", type=float, default=None); ap.add_argument("--conf-mid", type=float, default=None); ap.add_argument("--w-mid", type=float, default=0.3)
    ap.add_argument("--demos", nargs="*", default=[], help="subset of demo keys take/demoNN (QA passes on a few demos)")
    ap.add_argument("--ik-fail-mm", type=float, default=30.0, help="10-23 user: hard IK cutoff for the pos/rot masks (was 10 mm)")
    ap.add_argument("--mask-mode", default="gated", choices=["gated", "stitched"], help="stitched: pos weight 1.0 on the continuous stitched trajectory, masks = IK > 10 mm (pos+rot) + invalid jaw (grip); no drops")
    ap.add_argument("--grasp-fill-drop", default="on", choices=["on", "off"], help="off = v2u_all (10-23 decision): keep demos whose grasp is inside a > 5 s fill (positions stay masked)")
    ap.add_argument("--label", default="ikfk", choices=["ikfk", "rob", "fused"], help="10-26: rob = action_cartesian_robotized (A, policy target), ikfk = old IK->FK (B); 10-29: fused = human TCP without the table lift (F)")
    ap.add_argument("--speed", type=float, default=1.0, help="10-29: playback factor, 0.3 = demos 3.33x slower (pose, jaw and video on one timeline)"); ap.add_argument("--qa-only", default=""); ap.add_argument("--overwrite", action="store_true"); a = ap.parse_args()
    T2L.check_jaw_calib(); T2L.ROBOTIZE = "ik_v1"
    global MASK_MODE, IK_FAIL_MM, LABEL, SPEED; MASK_MODE = a.mask_mode; IK_FAIL_MM = a.ik_fail_mm; LABEL = a.label; SPEED = a.speed
    if a.pos_conf_dir:
        if a.conf_high is None or a.conf_mid is None: sys.exit("--pos-conf-dir needs --conf-high and --conf-mid (from the calibration)")
        CONF.update(dir=a.pos_conf_dir, high=a.conf_high, mid=a.conf_mid, w_mid=a.w_mid); VARIANTS.append("imuconf")
    trim = {(r["take"], int(r["demo"])): r for r in csv.DictReader(open(TRIM_CSV))}
    takes = [pathlib.Path(t).expanduser().resolve() for t in a.takes]
    bad = [t.name for t in takes if t.name in EXCLUDE_TAKES]
    if bad: sys.exit(f"excluded takes passed: {bad}")
    derived = pathlib.Path(a.derived); rows, built, asgs, src = [], {}, {}, {}
    for t in takes:
        fused = all((t / f"umi_traj_{s}_fused.npz").exists() and "fused_valid" in np.load(t / f"umi_traj_{s}_fused.npz").files for s in ("left", "right"))
        if a.pose_src == "auto": T2L.POSE_SRC = "_fused" if fused else ""
        else:
            T2L.POSE_SRC = a.pose_src
            if a.pose_src and not all((t / f"umi_traj_{s}{a.pose_src}.npz").exists() for s in ("left", "right")): print(f"[prep] {t.name}: no {a.pose_src} files"); src[t.name] = None; continue
        src[t.name] = T2L.POSE_SRC
        try:
            asgs[t.name] = T2L.prep_take(t, derived); print(f"[prep] {t.name} pose_src={'fused' if fused else 'tag_only'} rig_of_arm={asgs[t.name]['rig_of_arm']}", flush=True)
        except BaseException as ex:
            print(f"[prep] {t.name} FAILED {type(ex).__name__}: {ex}", flush=True)
        for d in T2L.demos_of(t):
            key = f"{t.name}/demo{d['idx']:02d}"; tr = trim.get((t.name, d["idx"]))
            if a.demos and key not in a.demos: continue
            row = dict(episode_key=key, take=t.name, demo=d["idx"], pose_src=(src.get(t.name) or "tag_only").lstrip("_") if src.get(t.name) is not None else "missing", status="", reason="")
            try:
                if t.name not in asgs: raise ValueError("take prep failed")
                if tr is None: raise ValueError("no idle-trim row (demo outside the recording)")
                win = (d["t0_ns"] + int(float(tr["keep_start_s"]) * 1e9), d["t0_ns"] + int(float(tr["keep_end_s"]) * 1e9))
                b = build(t, derived / (t.name + src[t.name]), asgs[t.name], d, win, int(a.cam_shift_ms * 1e6))
                n = b["n"]; row.update(trim_start_s=tr["keep_start_s"], trim_end_s=tr["keep_end_s"], cut_head_s=tr["cut_head_s"], cut_tail_s=tr["cut_tail_s"], frames=n, length_s=round(n * ACTION_DT_S, 1))
                for s in "LR":
                    gi = b[f"ginfo_{s}"]
                    row.update({f"measured_{s}_pct": round(float(b[f"meas_{s}"].mean()) * 100, 1), f"fill_long_{s}_pct": round(float(b[f"fill_{s}"].mean()) * 100, 1),
                                f"ik_fail_{s}_pct": round(float((b[f"ikmm_{s}"] > IK_FAIL_MM).mean()) * 100, 1), f"pos_masked_{s}_pct": round(float(1 - b[f"pos_ok_{s}"].mean()) * 100, 1), f"rot_masked_{s}_pct": round(float(1 - b[f"rot_ok_{s}"].mean()) * 100, 1),
                                f"no_pose_{s}_pct": round(float(1 - b[f"meas_{s}"].mean()) * 100, 1),
                                f"grip_masked_{s}_pct": round(float(1 - b[f"jv_{s}"].mean()) * 100, 1), f"grasps_{s}": len(gi),
                                f"grasp_warn_{s}": sum(GRASP_OK_MM <= x["max_mm"] <= GRASP_WARN_MM for x in gi), f"grasp_fail_{s}": sum(x["max_mm"] > GRASP_WARN_MM for x in gi),
                                f"grasp_max_mm_{s}": " ".join(str(x["max_mm"]) for x in gi)})
                for v in VARIANTS:
                    row.update({f"pos_eff_{v}_{s}_frames": round(float(b[f"pw_{v}_{s}"].sum()), 1) for s in "LR"})
                for s in "LR":
                    e = b[f"ikmm_{s}"]
                    for lab, lo_, hi_ in (("le10", -1, 10), ("10_20", 10, 20), ("20_30", 20, 30), ("30_50", 30, 50), ("gt50", 50, 1e9)):
                        row[f"ik_{lab}_{s}_pct"] = round(float(((e > lo_) & (e <= hi_)).mean()) * 100, 1)
                    rs = [(y - x) * ACTION_DT_S for x, y in runs(e > 50.0)]
                    long_ = [r for r in rs if r >= 0.5]
                    row.update({f"severe_runs_{s}": len(long_), f"severe_runs_s_{s}": round(sum(long_), 1), f"severe_flag_{s}": bool(any(r >= 1.0 for r in rs)), f"severe_runs_1s_{s}": int(sum(r >= 1.0 for r in rs)),
                                f"severe_longest_s_{s}": round(max(rs or [0.0]), 1)})
                row.update(conf_L_pct=round(float(b["conf_L"].mean()) * 100, 1), conf_R_pct=round(float(b["conf_R"].mean()) * 100, 1))
                row.update(min_link_gap_mm=round(float(b["gap_m"].min()) * 1000, 1), pos_supervised_frames=int(np.round(b["action_mask"][:, [0, 9]].mean(1).sum())),
                           rot_supervised_frames=int(np.round(b["action_mask"][:, [3, 12]].mean(1).sum())),
                           img_res_max_ms=round(float(max(b[f"res_{c}"].max() for c in CAMS)), 1))
                if MASK_MODE != "stitched" and row["measured_L_pct"] < DROP_MEASURED * 100 and row["measured_R_pct"] < DROP_MEASURED * 100:
                    raise ValueError(f"measured < 50 % for both arms ({row['measured_L_pct']} / {row['measured_R_pct']} %)")
                long_in_grasp = max([x["fill_overlap_s"] for s in "LR" for x in b[f"ginfo_{s}"]] or [0.0])
                row["fill_in_grasp_s"] = round(long_in_grasp, 1)
                if MASK_MODE != "stitched" and a.grasp_fill_drop == "on" and long_in_grasp > DROP_FILL_IN_GRASP_S: raise ValueError(f"filled run of {long_in_grasp:.1f} s inside a grasp window")
                row["status"] = "eligible"; built[key] = (t, b)
            except Exception as ex:
                row.update(status="dropped", reason=f"{type(ex).__name__}: {ex}")
            rows.append(row); print(f"{key} {row['status']} {row['reason'][:80]} len {row.get('length_s')} s meas L/R {row.get('measured_L_pct')}/{row.get('measured_R_pct')} pos-masked L/R {row.get('pos_masked_L_pct')}/{row.get('pos_masked_R_pct')}", flush=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    if a.qa_only:
        with open(a.qa_only, "w", newline="") as fh: w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
        print("qa-only:", sum(r["status"] == "eligible" for r in rows), "eligible of", len(rows)); return
    # ---- frozen split: old keys keep their split, new keys stratified
    old = json.loads(pathlib.Path(a.old_split).read_text())["episodes"] if pathlib.Path(a.old_split).exists() else {}
    oldmap = {k: s for s, ks in old.items() for k in ks}; elig = [r["episode_key"] for r in rows if r["status"] == "eligible"]
    split = {k: oldmap[k] for k in elig if k in oldmap}; split.update(T2L.split_assign([k for k in elig if k not in oldmap], a.seed))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    base = pathlib.Path(a.out).expanduser(); roots = {s: base.parent / f"{base.name}_{s}" for s in ("train", "val", "test")}
    for r in roots.values():
        if r.exists():
            if not a.overwrite: sys.exit(f"{r} exists (pass --overwrite)")
            shutil.rmtree(r)
    feats = {f"observation.images.{c}": dict(dtype="video", shape=(IMG_WH[1], IMG_WH[0], 3), names=["height", "width", "channels"]) for c in CAMS}
    feats.update({"observation.state": dict(dtype="float32", shape=(PROPRIO_DIM,), names=list(PROPRIO_NAMES)),
                  "action": dict(dtype="float32", shape=(20,), names=ABS_NAMES), "pose_abs": dict(dtype="float32", shape=(20,), names=ABS_NAMES),
                  "action_mask": dict(dtype="float32", shape=(20,), names=[f"mask_{x}" for x in ABS_NAMES]),
                  "anchor_ok": dict(dtype="float32", shape=(4,), names=["pos_ok_L", "rot_ok_L", "pos_ok_R", "rot_ok_R"]), "measured": dict(dtype="float32", shape=(2,), names=["measured_L", "measured_R"]),
                  "fill_long": dict(dtype="float32", shape=(2,), names=["fill_long_L", "fill_long_R"]),
                  "gap_len_s": dict(dtype="float32", shape=(2,), names=["gap_len_L", "gap_len_R"]), "pos_conf": dict(dtype="float32", shape=(2,), names=["conf_L", "conf_R"]),
                  "ik_warn": dict(dtype="float32", shape=(2,), names=["ik_warn_L", "ik_warn_R"]), "ik_severe": dict(dtype="float32", shape=(2,), names=["ik_severe_L", "ik_severe_R"]),
                  "ik_err_mm": dict(dtype="float32", shape=(2,), names=["ik_err_L", "ik_err_R"]),
                  "grasp_win": dict(dtype="float32", shape=(2,), names=["grasp_win_L", "grasp_win_R"]),
                  **{f"action_weight_{v}": dict(dtype="float32", shape=(20,), names=[f"w_{x}" for x in ABS_NAMES]) for v in VARIANTS},
                  **{f"anchor_w_{v}": dict(dtype="float32", shape=(4,), names=["pos_L", "rot_L", "pos_R", "rot_R"]) for v in VARIANTS},
                  **{f"pos_w_{v}": dict(dtype="float32", shape=(2,), names=["pos_w_L", "pos_w_R"]) for v in VARIANTS}, "weight": dict(dtype="float32", shape=(1,), names=["weight"]),
                  "gripper_cont": dict(dtype="float32", shape=(2,), names=["g_cont_left", "g_cont_right"]),
                  "src_index": dict(dtype="int64", shape=(3,), names=list(CAMS)), "img_residual_ms": dict(dtype="float32", shape=(3,), names=list(CAMS))})
    dss = {}; rd = pathlib.Path(a.render_dir); rd.mkdir(parents=True, exist_ok=True); nrend = 0
    for row in rows:
        key = row["episode_key"]
        if row["status"] != "eligible": continue
        sp = split[key]; row["split"] = sp; t, b = built[key]
        if sp not in dss:
            dss[sp] = LeRobotDataset.create(f"local/{base.name}_{sp}", fps=20, features=feats, root=roots[sp], robot_type="handumi_tagrig_x2",
                                            use_videos=True, image_writer_threads=0, streaming_encoding=True, encoder_threads=2)
        ds = dss[sp]; TL, TR, gL, gR = b["T_left"], b["T_right"], b["g_left"], b["g_right"]
        st = proprio_rows_lr(TL, TR, gL, gR); act = abs_action_rows(TL, TR, gL, gR)
        pa = np.concatenate([T_to_pose9(TL), T_to_pose9(TR), gL[:, None], gR[:, None]], -1).astype(np.float32)
        for i in range(b["n"]):
            fr = {f"observation.images.{c}": T2L.load_img(b, c, int(b[f"img_{c}"][i])) for c in CAMS}
            fr.update({"observation.state": st[i], "action": act[i], "pose_abs": pa[i], "action_mask": b["action_mask"][i],
                       "anchor_ok": b["anchor_ok"][i], "measured": np.array([b["meas_L"][i], b["meas_R"][i]], np.float32),
                       "fill_long": np.array([b["fill_L"][i], b["fill_R"][i]], np.float32),
                       "gap_len_s": np.array([b["gl_L"][i], b["gl_R"][i]], np.float32), "pos_conf": np.array([b["conf_L"][i], b["conf_R"][i]], np.float32),
                       "ik_warn": np.array([b["ikwarn_L"][i], b["ikwarn_R"][i]], np.float32), "ik_severe": np.array([b["iksev_L"][i], b["iksev_R"][i]], np.float32),
                       "ik_err_mm": np.array([b["ikmm_L"][i], b["ikmm_R"][i]], np.float32),
                       "grasp_win": np.array([b["grasp_L"][i], b["grasp_R"][i]], np.float32),
                       **{f"action_weight_{v}": b[f"aw_{v}"][i] for v in VARIANTS}, **{f"anchor_w_{v}": b[f"anw_{v}"][i] for v in VARIANTS},
                       **{f"pos_w_{v}": np.array([b[f"pw_{v}_L"][i], b[f"pw_{v}_R"][i]], np.float32) for v in VARIANTS}, "weight": np.array([b["weight"][i]], np.float32),
                       "gripper_cont": np.array([b["gc_left"][i], b["gc_right"][i]], np.float32), "src_index": np.array([b[f"img_{c}"][i] for c in CAMS], np.int64),
                       "img_residual_ms": np.array([b[f"res_{c}"][i] for c in CAMS], np.float32), "task": T2L.TASK})
            ds.add_frame(fr)
        row["lerobot_episode"] = ds.meta.total_episodes; ds.save_episode(); row["status"] = "converted"
        if nrend < a.render and row["take"] in {"20261022_124754", "20261022_134315", "20261022_161756", "20261023_113550"} and row["take"] not in {r_.get("rendered") for r_ in rows}:
            render(b, rd / f"{key.replace('/', '_')}.mp4", key); row["rendered"] = row["take"]; nrend += 1
        print(f"wrote {key} [{sp}] {b['n']} frames", flush=True)
    for d in dss.values(): d.finalize()
    # 10-23 hard rule: NO split episodes. #episodes per split == #converted demos in it, frame_index continuous 0..n-1, length == built length
    for sp_, r_ in roots.items():
        if sp_ not in dss: continue
        chk = LeRobotDataset(f"local/{base.name}_{sp_}", root=r_); hf_ = chk.hf_dataset.with_format("numpy"); ep_ = np.asarray(hf_["episode_index"]); fi_ = np.asarray(hf_["frame_index"])
        conv = [r for r in rows if r["status"] == "converted" and r.get("split") == sp_]
        assert chk.meta.total_episodes == len(conv), f"{sp_}: {chk.meta.total_episodes} episodes != {len(conv)} demos"
        for r in conv:
            fi = fi_[ep_ == r["lerobot_episode"]]
            assert len(fi) == r["frames"] and np.array_equal(fi, np.arange(len(fi))), f"{r['episode_key']}: episode split / frame_index not continuous"
        print(f"[assert] {sp_}: {len(conv)} demos == {chk.meta.total_episodes} episodes, every frame_index continuous", flush=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    split_doc = dict(method="FROZEN by demo: v1 split kept for its demos; new eligible demos stratified (seed %d, 85/8/7)" % a.seed, seed=a.seed,
                     episodes={s: sorted(k for k, v in split.items() if v == s) for s in ("train", "val", "test")}, old_split=a.old_split)
    meta = dict(T2L.META, dataset="tagrig_laundry_v2u (unsplit + per-arm masks)", converter="tag_rig_unsplit.py", speed=SPEED, label=LABEL, pose_src=src, robotize="ik_v1", trim_csv=str(TRIM_CSV),
                fill_mask_s=FILL_MASK_S, ik_fail_mm=IK_FAIL_MM, ik_qa_flags="ik_warn: cutoff < err <= 50 mm, ik_severe: err > 50 mm (QA only, not weights); severe_flag = a > 50 mm run >= 1 s", grasp_gate=dict(ok_mm=GRASP_OK_MM, warn_mm=GRASP_WARN_MM, warn_weight=GRASP_W),
                drop_rule=f"measured < {DROP_MEASURED * 100:.0f} % both arms" + (f", or a filled run > {DROP_FILL_IN_GRASP_S} s containing a jaw close" if a.grasp_fill_drop == "on" else " (grasp-fill drop OFF: v2u_all)"), cam_shift_ms=a.cam_shift_ms,
                exclude_takes=sorted(EXCLUDE_TAKES), contract_version=__import__("contracts").CONTRACT_VERSION)
    for s, r in roots.items():
        r.mkdir(parents=True, exist_ok=True); (r / "meta").mkdir(exist_ok=True)
        with open(r / "episode_table.csv", "w", newline="") as fh: w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
        (r / "meta" / "split.json").write_text(json.dumps(dict(split_doc, this_split=s), indent=1)); (r / "source_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
        shutil.copy(r / "episode_table.csv", r / "conversion_manifest.csv")
    print("done", sum(r["status"] == "converted" for r in rows), "converted,", sum(r["status"] == "dropped" for r in rows), "dropped;",
          {s: sum(1 for r in rows if r.get("split") == s and r["status"] == "converted") for s in roots})


if __name__ == "__main__":
    main()
