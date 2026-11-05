"""Offline virtual-gripper pass for one wrist video (renderer spec P0 steps 1-7):

  hand keypoints (wrist camera frame) -> gripper_frame (per frame) -> smoothing (whole sequence, zero-phase)
  -> T_camera_gripper(t), jaw(t)  ==>  BOTH the rendered overlay video AND gripper_pose.{csv,npy} (labels read the same array)

  python -m laundry_collector.preprocess.virtual_gripper --episode <ep> --side right --keypoints <npz> [--debug] [--config gripper.yaml]

keypoints npz (from a HandPoseProvider run): keypoints_3d (N,21,3) metres camera frame, confidence (N,21), video_frame (N,) [optional].
Outputs <ep>/processed/<side>/: wrist_raw.mp4, wrist_virtual_overlay.mp4, [wrist_debug.mp4], gripper_pose.csv, gripper_pose.npy,
T_camera_gripper.npy (N,4,4), gripper_qa.json.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
from pathlib import Path

import av
import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation as R

from .gripper_frame import GripperConfig, gripper_from_keypoints
from .renderer import OpenCVRenderer, draw_debug
from .smoothing import smooth_trajectory

QA_TH = dict(min_valid=0.9, max_pos_jump_mm=30.0, max_rot_jump_deg=20.0)
COLS = ["frame_index", "timestamp_ns", "hand_pose_valid", "rot_valid", "x", "y", "z", "qx", "qy", "qz", "qw",
        "pinch_distance_m", "jaw_width_m", "jaw_normalized", "thumb_confidence", "index_confidence"]


def compute_track(kp: np.ndarray, conf: np.ndarray | None, side: str, cfg: GripperConfig) -> dict:
    n = len(kp); P = np.zeros((n, 3)); Q = np.tile([0, 0, 0, 1.0], (n, 1)); pinch = np.zeros(n)
    pv = np.zeros(n, bool); rv = np.zeros(n, bool); reasons = []
    for i in range(n):
        g = gripper_from_keypoints(kp[i], None if conf is None else conf[i], side, cfg)
        reasons.append(g["reason"])
        if g["valid"]:
            P[i], pinch[i], pv[i] = g["p"], g["pinch_m"], True
        if g["rot_valid"]:
            Q[i], rv[i] = R.from_matrix(g["R"]).as_quat(), True
    Ps, Qs, Js = smooth_trajectory(P, Q, pinch, pv, rv & pv, alpha_position=cfg.alpha_position, alpha_rotation=cfg.alpha_rotation,
                                   alpha_jaw=cfg.alpha_jaw, zero_phase=cfg.zero_phase)
    T = np.tile(np.eye(4), (n, 1, 1)); T[:, :3, :3] = R.from_quat(Qs).as_matrix(); T[:, :3, 3] = Ps
    return dict(T=T, pinch=Js, jaw_m=cfg.jaw_width(Js), jaw_n=cfg.jaw_normalized(Js), pos_valid=pv, rot_valid=rv, reasons=reasons,
                raw_p=P, raw_pinch=pinch)


def qa(track: dict) -> dict:
    T, pv = track["T"], track["pos_valid"]
    dp = np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1) * 1000
    dr = np.degrees(R.from_matrix(np.einsum("nji,njk->nik", T[:-1, :3, :3], T[1:, :3, :3])).magnitude())
    raw = track["raw_p"][pv]; rawj = np.linalg.norm(np.diff(raw, axis=0), axis=1) * 1000 if len(raw) > 1 else np.zeros(1)
    pin = track["raw_pinch"][pv] * 1000 if pv.any() else np.zeros(1)
    flags_pos = np.flatnonzero(dp > QA_TH["max_pos_jump_mm"]).tolist(); flags_rot = np.flatnonzero(dr > QA_TH["max_rot_jump_deg"]).tolist()
    r = dict(frames_total=int(len(T)), hand_pose_valid=round(float(pv.mean()), 4), invalid_pose_frames=int((~pv).sum()),
             rotation_valid=round(float(track["rot_valid"].mean()), 4),
             invalid_reasons=pd.Series([x for x in track["reasons"] if x]).value_counts().to_dict(),
             pinch_distance_mm=dict(p05=round(float(np.percentile(pin, 5)), 1), p50=round(float(np.percentile(pin, 50)), 1), p95=round(float(np.percentile(pin, 95)), 1)),
             position_jump_max_mm_per_frame=round(float(dp.max()) if len(dp) else 0, 2), raw_position_jump_max_mm=round(float(rawj.max()), 2),
             rotation_jump_max_deg_per_frame=round(float(dr.max()) if len(dr) else 0, 2),
             flagged_frames=dict(position=flags_pos[:50], rotation=flags_rot[:50]), thresholds=QA_TH)
    r["verdict"] = "PASS" if (r["hand_pose_valid"] >= QA_TH["min_valid"] and not flags_pos and not flags_rot) else "CHECK"
    return r


def run(ep: Path, side: str, keypoints: Path, cfg: GripperConfig, debug: bool = False) -> dict:
    ep = Path(ep); s = f"{side}_wrist"; out = ep / "processed" / side; out.mkdir(parents=True, exist_ok=True)
    intr = json.loads((ep / "calibration" / f"{side}_intrinsics.json").read_text())
    ts = pd.read_csv(ep / s / "timestamps.csv")
    z = np.load(keypoints)
    kp, conf = z["keypoints_3d"].astype(float), (z["confidence"].astype(float) if "confidence" in z else None)
    vf = z["video_frame"].astype(int) if "video_frame" in z else np.arange(len(kp))
    if len(vf) != len(ts) or not (vf == np.arange(len(ts))).all():
        # keypoints for a subset of frames: expand to every video frame, missing -> NaN (invalid -> filled by the smoother)
        full = np.full((len(ts), 21, 3), np.nan); full[vf] = kp; kp = full
        if conf is not None:
            c = np.zeros((len(ts), 21)); c[vf] = conf; conf = c
    tr = compute_track(kp, conf, side, cfg)
    hand_T = tr["T"].copy()
    robot_view = cfg.pose_mode == "camera_fixed" and cfg.view == "robot_c922"
    if robot_view:
        from .robot_view import T_camera_tcp
        tr["T"] = np.broadcast_to(T_camera_tcp(), hand_T.shape).copy()      # robot TCP in the robot camera = the robot data convention
    elif cfg.pose_mode == "camera_fixed":
        tr["T"] = np.broadcast_to(cfg.mount_T(), hand_T.shape).copy()
    T = tr["T"]
    # one array -> labels
    q = R.from_matrix(T[:, :3, :3]).as_quat()
    df = pd.DataFrame(dict(frame_index=np.arange(len(T)), timestamp_ns=ts.host_timestamp_ns.to_numpy(), hand_pose_valid=tr["pos_valid"],
                           rot_valid=tr["rot_valid"], x=T[:, 0, 3], y=T[:, 1, 3], z=T[:, 2, 3], qx=q[:, 0], qy=q[:, 1], qz=q[:, 2], qw=q[:, 3],
                           pinch_distance_m=tr["pinch"], jaw_width_m=tr["jaw_m"], jaw_normalized=tr["jaw_n"],
                           thumb_confidence=conf[:, 4] if conf is not None else np.nan, index_confidence=conf[:, 8] if conf is not None else np.nan))[COLS]
    df["hand_tcp_x"], df["hand_tcp_y"], df["hand_tcp_z"] = hand_T[:, 0, 3], hand_T[:, 1, 3], hand_T[:, 2, 3]   # diagnostic: tracked pinch point
    df.to_csv(out / "gripper_pose.csv", index=False)
    np.save(out / "gripper_pose.npy", df.to_numpy(np.float64)); (out / "gripper_pose.columns.json").write_text(json.dumps(list(df.columns)))
    np.save(out / "T_camera_gripper.npy", T)
    (out / "gripper_frame.json").write_text(json.dumps(dict(
        convention="origin = TCP between the finger ends; +y approach; +x jaw axis; +z = x x y",
        pose_mode=cfg.pose_mode, mount_T_camera_gripper=cfg.mount_T().tolist(),
        frame="wrist camera (OpenCV: x right, y down, z forward)", config=dataclasses.asdict(cfg), keypoints=str(keypoints)), indent=1))
    # one array -> pixels
    shutil.copyfile(ep / s / "rgb" / "rgb.mp4", out / "wrist_raw.mp4")
    from .renderer import PiperTipsMesh, PrimitiveGripperMesh
    rend = OpenCVRenderer(PiperTipsMesh() if cfg.pose_mode == "camera_fixed" else PrimitiveGripperMesh())   # HandUMI's mounted tips
    src = av.open(str(ep / s / "rgb" / "rgb.mp4")); vs = src.streams.video[0]
    W, H = vs.codec_context.width, vs.codec_context.height
    def writer(name):
        c = av.open(str(out / name), "w"); st = c.add_stream("libx264", rate=30); st.width, st.height, st.pix_fmt = W, H, "yuv420p"; st.options = {"crf": "18"}
        return c, st
    if robot_view:
        from .robot_view import RobotC922View, RobotTipsSprite
        V, S = RobotC922View(intr, (W, H)), RobotTipsSprite()
        W, H = V.size
        wr = writer("wrist_robotview_raw.mp4"); wg = writer("wrist_robotview_gripper.mp4")
        for i, fr in enumerate(src.decode(video=0)):
            if i >= len(T):
                break
            v = V.render(fr.to_ndarray(format="bgr24")); g, _ = S.composite(v, float(tr["jaw_m"][i]))
            for (c, st), im in ((wr, v), (wg, g)):
                for pk in st.encode(av.VideoFrame.from_ndarray(im, format="bgr24")):
                    c.mux(pk)
        for c, st in (wr, wg):
            for pk in st.encode(None):
                c.mux(pk)
            c.close()
        src.close()
        rep = qa(tr); rep.update(episode=ep.name, side=side, view="robot_c922 (measured reBot right-wrist C922, real tips sprite)")
        (out / "gripper_qa.json").write_text(json.dumps(rep, indent=1))
        return rep
    wo = writer("wrist_virtual_overlay.mp4"); wd = writer("wrist_debug.mp4") if debug else None
    for i, fr in enumerate(src.decode(video=0)):
        if i >= len(T):
            break
        img = fr.to_ndarray(format="bgr24")
        ov, _, _ = rend.render(img, T[i], tr["jaw_m"][i], intr)
        outs = [(wo, ov)]
        if wd:
            outs.append((wd, draw_debug(ov.copy(), intr, kp[i] if np.isfinite(kp[i]).all() else None, T[i])))
        for (c, st), im in outs:
            for p in st.encode(av.VideoFrame.from_ndarray(im, format="bgr24")):
                c.mux(p)
    for c, st in (wo, wd) if wd else (wo,):
        for p in st.encode(None):
            c.mux(p)
        c.close()
    src.close()
    rep = qa(tr); rep.update(episode=ep.name, side=side)
    (out / "gripper_qa.json").write_text(json.dumps(rep, indent=1))
    return rep


def load_cfg(path: Path | None) -> GripperConfig:
    cfg = GripperConfig()
    if path:
        for k, v in (yaml.safe_load(Path(path).read_text()) or {}).items():
            if not hasattr(cfg, k):
                raise ValueError(f"unknown gripper config key {k}")
            setattr(cfg, k, v)
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=Path, required=True); ap.add_argument("--side", choices=["left", "right"], required=True)
    ap.add_argument("--keypoints", type=Path, required=True); ap.add_argument("--config", type=Path); ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    print(json.dumps(run(a.episode, a.side, a.keypoints, load_cfg(a.config), a.debug), indent=1))
