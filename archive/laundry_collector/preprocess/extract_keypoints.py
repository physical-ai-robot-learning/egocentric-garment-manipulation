"""Wrist video -> keypoints npz for virtual_gripper.py (MediaPipePnPProvider).
  python -m laundry_collector.preprocess.extract_keypoints --episode <ep> --side left [--out kp.npz]"""
import argparse, json
from pathlib import Path

import av
import numpy as np
import pandas as pd

from .hand_pose import MediaPipePnPProvider, WilorProvider

ap = argparse.ArgumentParser(); ap.add_argument("--episode", type=Path, required=True); ap.add_argument("--side", required=True); ap.add_argument("--out", type=Path); ap.add_argument("--provider", default="wilor", choices=["wilor", "mediapipe_pnp"])
a = ap.parse_args()
ep, s = a.episode, f"{a.side}_wrist"
intr = json.loads((ep / "calibration" / f"{a.side}_intrinsics.json").read_text())
ts = pd.read_csv(ep / s / "timestamps.csv")
c = av.open(str(ep / s / "rgb" / "rgb.mp4")); vs = c.streams.video[0]
prov = (WilorProvider if a.provider == "wilor" else MediaPipePnPProvider)(intr, (vs.codec_context.width, vs.codec_context.height))
n = len(ts); kp = np.full((n, 21, 3), np.nan); conf = np.zeros((n, 21)); reproj = np.full(n, np.nan)
t0 = int(ts.host_timestamp_ns.iloc[0])
for i, fr in enumerate(c.decode(video=0)):
    if i >= n:
        break
    hp = prov.process(fr.to_ndarray(format="bgr24"), side=a.side, timestamp_ms=(int(ts.host_timestamp_ns.iloc[i]) - t0) // 1_000_000)
    if hp is not None:
        kp[i], conf[i], reproj[i] = hp.keypoints_3d, hp.confidence, hp.extra.get("reproj_px", np.nan)
out = a.out or ep / "processed" / a.side / f"keypoints_{a.provider}.npz"; out.parent.mkdir(parents=True, exist_ok=True)
np.savez(out, keypoints_3d=kp, confidence=conf, video_frame=np.arange(n), reproj_px=reproj)
det = np.isfinite(kp[:, 0, 0])
print(f"{out}: detected {det.mean():.1%} of {n} frames, reproj median {np.nanmedian(reproj):.1f} px, "
      f"pinch median {np.nanmedian(np.linalg.norm(kp[det, 4] - kp[det, 8], axis=1)) * 1000:.0f} mm, wrist depth median {np.nanmedian(kp[det, 0, 2]):.2f} m")
