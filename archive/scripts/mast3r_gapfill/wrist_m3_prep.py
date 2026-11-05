"""Wrist fisheye video of a tag_rig_record --wrist take -> MASt3R-SLAM input (2026-10-20, GPT plan: SLAM relative trajectory + IMU +
tag anchors). Same transform as ~/umi_bridge/trackA_mast3r_pose_v1/prep_undistort.py (fisheye_to_pinhole_v1): KannalaBrandt K,D ->
cv2.fisheye.initUndistortRectifyMap(R = I, P = estimateNewCameraMatrix(balance 0)) at 960x540, so a MASt3R pose is a pose of the SAME
camera optical frame. Intrinsics: --calib (default fisheye_right_v003, 1920x1080) scaled to the recorded size (1280x720).
  !! fisheye_right_v003 was calibrated for the earlier right-wrist Arducam; whether this UMI carries that same camera/lens is UNVERIFIED.
Writes <out>/<tag>/rgb/%06d.png (every --every-th wrist frame), calib.yaml (width/height/[fx,fy,cx,cy]), frames.json (index -> t_ns).
    .venv/bin/python scripts/wrist_m3_prep.py datasets/tag_rig/<take> [--every 1] -> datasets/m3_frames/<tag>/"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take"); ap.add_argument("--calib", default=str(ROOT / "configs/calibration/fisheye_right_v003.yaml"))
    ap.add_argument("--every", type=int, default=1); ap.add_argument("--hand", default="right", choices=["right", "left"]); ap.add_argument("--wrist-dir", help="손목 영상이 따로 녹화된 경우 (umi_jaw record 폴더)"); ap.add_argument("--out", default=str(ROOT / "datasets/m3_frames")); a = ap.parse_args()
    E = Path(a.take); W = Path(a.wrist_dir) if a.wrist_dir else E / ("wrist" if a.hand == "right" else "wrist_left"); t = np.load(W / "t_ns.npy"); meta = json.loads((W / "meta.json").read_text())
    w0, h0 = meta["size"] if "size" in meta else cv2.imread(str(W / "wrist_00000.jpg")).shape[1::-1]      # older 15 Hz records: 960x540, no size
    c = yaml.safe_load(Path(a.calib).read_text()); K = np.array(c["K"], float); D = np.array(c["D"], float).reshape(4, 1)
    sx, sy = w0 / c["image_size"][0], h0 / c["image_size"][1]; K = K.copy(); K[0] *= sx; K[1] *= sy
    P = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w0, h0), np.eye(3), balance=0.0)
    P[0] *= 960 / w0; P[1] *= 540 / h0
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), P, (960, 540), cv2.CV_16SC2)
    tag = f"umi_{E.name}" + ("" if a.hand == "right" else "_left"); o = Path(a.out) / tag; (o / "rgb").mkdir(parents=True, exist_ok=True); idx = []
    for n, k in enumerate(range(0, len(t), a.every)):
        f = cv2.imread(str(W / f"wrist_{k:05d}.jpg"))
        if f is None: continue
        cv2.imwrite(str(o / "rgb" / f"{n:06d}.png"), cv2.remap(f, m1, m2, cv2.INTER_LINEAR)); idx.append(dict(i=n, wrist_k=k, t_ns=int(t[k])))
    (o / "calib.yaml").write_text(f"width: 960\nheight: 540\ncalibration: [{P[0, 0]:.6f}, {P[1, 1]:.6f}, {P[0, 2]:.6f}, {P[1, 2]:.6f}]\n")
    (o / "frames.json").write_text(json.dumps(dict(take=str(E), calib=a.calib, every=a.every, P=P.tolist(), frames=idx)))
    print(f"{o}: {len(idx)} 프레임, P fx {P[0, 0]:.1f} cx {P[0, 2]:.1f} cy {P[1, 2]:.1f}")


if __name__ == "__main__":
    main()
