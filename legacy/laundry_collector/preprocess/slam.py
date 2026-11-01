"""MASt3R-SLAM (RGB only, offline) for one wrist video -> camera 6DoF (spec §13).

Backend = the per-side runner already used for HRA_A100 on the 4090 (sm_89 build, /mnt/shared/johann/c8/robotlike/
robotlike_resident_4090.sh): input dir per side = raw_video.mp4 (960x540) + orbslam_setting.yaml (KannalaBrandt8), output
<out>/<tag>/ss1.csv (frame_idx,timestamp,state,is_lost,is_keyframe,x,y,z,q_x,q_y,q_z,q_w,map_updated,kf_ref,sim3_scale,match_frac),
pose = T_slam_camera (camera -> SLAM world), NON-METRIC (world_align.py applies a scale).

  stage(...)    episode + side -> <stage>/<tag>/{raw_video.mp4, orbslam_setting.yaml, frame_map.csv}   (Mac, no GPU)
  submit(...)   rsync the staged dirs to the 4090 and `ray job submit` the runner (one side per process, retry x3)
  collect(...)  ss1.csv -> <episode>/processed/<side>_camera_pose.npy + <side>_tracking_quality.csv
"""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SLAM_W, SLAM_H = 960, 540
REMOTE = dict(host="bh-aiteam@100.64.0.2", jump="head-lp", root="/home/bh-aiteam/laundry_slam",
              runner="/mnt/shared/johann/c8/robotlike/robotlike_resident_4090.sh", ray="http://100.64.0.1:8265", node="node:100.64.0.2")
POSE_COLS = ["frame_idx", "host_timestamp_ns", "valid", "x", "y", "z", "qx", "qy", "qz", "qw", "sim3_scale"]


def tag_of(ep: Path, side: str) -> str:
    return f"{ep.parent.name}__{ep.name}__{side}"


def orbslam_setting(intr: dict, out: Path, src_w: int, src_h: int) -> None:
    if intr.get("model") != "kannala_brandt":
        raise ValueError(f"wrist intrinsics must be a calibrated kannala_brandt fisheye, got {intr.get('model')} -- calibrate first")
    sx, sy = SLAM_W / src_w, SLAM_H / src_h
    K, D = np.asarray(intr["K"], float), [float(x) for x in intr["D"]]
    out.write_text("%YAML:1.0\n\n# laundry_collector stage: fisheye intrinsics " + str(intr.get("calibration_file")) + f", scaled {src_w}x{src_h} -> {SLAM_W}x{SLAM_H}\n"
                   'File.version: "1.0"\nCamera.type: "KannalaBrandt8"\n\n'
                   f"Camera1.fx: {K[0, 0] * sx:.6f}\nCamera1.fy: {K[1, 1] * sy:.6f}\nCamera1.cx: {K[0, 2] * sx:.6f}\nCamera1.cy: {K[1, 2] * sy:.6f}\n\n"
                   + "".join(f"Camera1.k{i + 1}: {d:.9f}\n" for i, d in enumerate(D[:4]))
                   + f"\nCamera.width: {SLAM_W}\nCamera.height: {SLAM_H}\nCamera.fps: 30\nCamera.RGB: 1\n")


def stage(ep: Path, side: str, stage_root: Path, *, phase: str | None = None) -> Path:
    """Transcode <side>_wrist/rgb/rgb.mp4 to 960x540 + write the SLAM setting. phase=None keeps the whole recording (the
    initialization hold is needed for the world alignment), else only that phase."""
    import av, cv2
    ep = Path(ep); s = f"{side}_wrist"
    intr = json.loads((ep / "calibration" / f"{side}_intrinsics.json").read_text())
    ts = pd.read_csv(ep / s / "timestamps.csv")
    md = json.loads((ep / "metadata.json").read_text()) if (ep / "metadata.json").exists() else {}
    keep = np.ones(len(ts), bool)
    if phase:
        ph = next(p for p in md.get("phases", []) if p["phase"] == phase)
        keep = (ts.host_timestamp_ns >= ph["start_host_ns"]) & (ts.host_timestamp_ns <= (ph["end_host_ns"] or 1 << 62))
    out = Path(stage_root) / tag_of(ep, side); out.mkdir(parents=True, exist_ok=True)
    src = av.open(str(ep / s / "rgb" / "rgb.mp4")); vs = src.streams.video[0]
    w, h = vs.codec_context.width, vs.codec_context.height
    orbslam_setting(intr, out / "orbslam_setting.yaml", w, h)
    dst = av.open(str(out / "raw_video.mp4"), "w"); st = dst.add_stream("libx264", rate=30)
    st.width, st.height, st.pix_fmt = SLAM_W, SLAM_H, "yuv420p"; st.options = {"crf": "16", "preset": "medium"}
    rows = []
    for i, fr in enumerate(src.decode(video=0)):
        if i >= len(keep) or not keep[i]:
            continue
        img = cv2.resize(fr.to_ndarray(format="bgr24"), (SLAM_W, SLAM_H), interpolation=cv2.INTER_AREA)
        for p in st.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
            dst.mux(p)
        rows.append((len(rows), i, int(ts.host_timestamp_ns.iloc[i])))
    for p in st.encode(None):
        dst.mux(p)
    dst.close(); src.close()
    with open(out / "frame_map.csv", "w", newline="") as f:
        csv.writer(f).writerows([("slam_frame", "video_frame", "host_timestamp_ns")] + rows)
    return out


def _ssh(cmd: str) -> str:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-J", REMOTE["jump"], REMOTE["host"], cmd], check=True, capture_output=True, text=True).stdout


def submit(staged: list[Path], job_name: str) -> str:
    """Copy staged sides to the 4090 and submit ONE Ray job that runs them one per process (MEMSTOP=1, as for HRA_A100).
    A side with < 40 TRACKING rows is renamed .failN and retried, max 3 tries (never deleted)."""
    r = REMOTE; rin, rout = f"{r['root']}/in", f"{r['root']}/runs"
    _ssh(f"mkdir -p {rin} {rout} {r['root']}/lists")
    for d in staged:
        subprocess.run(["/opt/homebrew/bin/rsync", "-a", "-e", f"ssh -o BatchMode=yes -J {r['jump']}", f"{d}/", f"{r['host']}:{rin}/{d.name}/"], check=True)
    lst = f"{r['root']}/lists/{job_name}.txt"
    _ssh(f"printf '%s\\n' {' '.join(d.name for d in staged)} > {lst}")
    loop = (f"export MEMSTOP=1; for t in $(cat {lst}); do for k in 1 2 3; do [ -f {rout}/$t/ss1.csv ] && break; "
            f"echo $t > {lst}.one; bash {r['runner']} {lst}.one {rout} {rin} | grep -E '^(OK|FAIL)'; "
            f"v=$(cut -d, -f3 {rout}/$t/ss1.csv 2>/dev/null | grep -c TRACKING); [ $v -ge 40 ] && break; "
            f"[ -d {rout}/$t ] && mv {rout}/$t {rout}/$t.fail$k; done; done; echo LOOP_DONE")
    subprocess.run(["ray", "job", "submit", "--address", r["ray"], "--no-wait", "--submission-id", job_name, "--entrypoint-num-gpus", "1",
                    "--entrypoint-resources", json.dumps({r["node"]: 0.001}), "--", "bash", "-c", loop], check=True)
    return job_name


def collect(ep: Path, side: str, stage_root: Path, *, pull: bool = True) -> Path:
    """ss1.csv (+ frame_map.csv) -> processed/<side>_camera_pose.npy (columns POSE_COLS) + <side>_tracking_quality.csv."""
    ep = Path(ep); tag = tag_of(ep, side); d = Path(stage_root) / tag
    if pull:
        r = REMOTE
        subprocess.run(["/opt/homebrew/bin/rsync", "-a", "-e", f"ssh -o BatchMode=yes -J {r['jump']}", f"{r['host']}:{r['root']}/runs/{tag}/ss1.csv", f"{d}/"], check=True)
    ss = pd.read_csv(d / "ss1.csv"); fm = pd.read_csv(d / "frame_map.csv").set_index("slam_frame")
    valid = (ss.state.isin(["INIT", "TRACKING"]) & (ss.is_lost.astype(str).str.lower() == "false")).to_numpy()
    host = fm.loc[ss.frame_idx, "host_timestamp_ns"].to_numpy(np.int64)
    arr = np.column_stack([fm.loc[ss.frame_idx, "video_frame"].to_numpy(), host, valid, ss[["x", "y", "z", "q_x", "q_y", "q_z", "q_w", "sim3_scale"]].to_numpy(float)])
    pd_dir = ep / "processed"; pd_dir.mkdir(exist_ok=True)
    np.save(pd_dir / f"{side}_camera_pose.npy", arr)
    (pd_dir / f"{side}_camera_pose.columns.json").write_text(json.dumps(dict(columns=POSE_COLS, frame="T_slam_camera (non-metric)", frame_idx="wrist video frame")))
    q = ss[["frame_idx", "state", "is_lost", "is_keyframe", "kf_ref", "match_frac"]].copy(); q["valid"] = valid
    q.to_csv(pd_dir / f"{side}_tracking_quality.csv", index=False)
    return pd_dir / f"{side}_camera_pose.npy"
