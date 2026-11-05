#!/usr/bin/env python3
"""[2026-09-23] trackA_mast3r_pose_v1, step 1 (CPU, node, holobrain env): fisheye video -> pinhole PNG frames.

MASt3R-SLAM's calibrated mode takes pinhole + OpenCV radtan only; the HandUMI wrist camera is KannalaBrandt8
(fisheye_{side}_v002.yaml, copied into each export's orbslam_setting.yaml). So the frames are undistorted here, ONCE,
as a named transform:

    fisheye_to_pinhole_v1:  cv2.fisheye.initUndistortRectifyMap(K, D, R = I, P, (960, 540))
                            P = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (960, 540), I, balance=0.0)

R = I keeps the camera OPTICAL FRAME unchanged, so a MASt3R pose is a pose of the same camera frame that
camera_tcp_v2 and the ArUco reference use. MASt3R-SLAM is then given P with zero distortion.
Writes <out>/<tag>/rgb/%06d.png (every source frame, index = source frame index), calib.yaml, prep.json.
"""
import hashlib, json, os, pathlib, re, sys
import numpy as np, cv2

BASE = pathlib.Path(os.environ.get("M3_BASE", pathlib.Path.home() / "trackA_mast3r_pose_v1"))
SRC = [BASE / "val/valB", BASE / "val/valC"]
OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "frames"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def setting(p):
    t = p.read_text(); g = lambda k: float(re.search(rf"^{k}:\s*([-\d.eE]+)", t, re.M).group(1))
    K = np.array([[g("Camera1.fx"), 0, g("Camera1.cx")], [0, g("Camera1.fy"), g("Camera1.cy")], [0, 0, 1.0]])
    D = np.array([g(f"Camera1.k{i}") for i in (1, 2, 3, 4)]).reshape(4, 1)
    return K, D, int(g("Camera.width")), int(g("Camera.height"))


for root in SRC:
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "raw_video.mp4").is_file()):
        tag = f"{root.name}_{d.name}"; o = OUT / tag; (o / "rgb").mkdir(parents=True, exist_ok=True)
        K, D, W, H = setting(d / "orbslam_setting.yaml")
        P = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (W, H), np.eye(3), balance=0.0)
        mx, my = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), P, (W, H), cv2.CV_32FC1)
        cap = cv2.VideoCapture(str(d / "raw_video.mp4")); n = 0
        while True:
            ok, f = cap.read()
            if not ok: break
            cv2.imwrite(str(o / "rgb" / f"{n:06d}.png"), cv2.remap(f, mx, my, cv2.INTER_LINEAR), [cv2.IMWRITE_PNG_COMPRESSION, 1]); n += 1
        cap.release()
        calib = dict(width=W, height=H, calibration=[float(P[0, 0]), float(P[1, 1]), float(P[0, 2]), float(P[1, 2])])
        (o / "calib.yaml").write_text("width: %d\nheight: %d\ncalibration: [%.6f, %.6f, %.6f, %.6f]\n" % (W, H, *calib["calibration"]))
        hfov = float(np.degrees(2 * np.arctan(W / 2 / P[0, 0])))
        json.dump(dict(tag=tag, source_dir=str(d), source_video_md5=md5(d / "raw_video.mp4"),
                       source_setting_md5=md5(d / "orbslam_setting.yaml"), n_frames=n, transform="fisheye_to_pinhole_v1",
                       K_fisheye=K.tolist(), D_fisheye=D.ravel().tolist(), P_pinhole=P.tolist(), R="identity", balance=0.0,
                       pinhole_hfov_deg=hfov, intrinsic_hash=hashlib.md5(np.round(P, 6).tobytes()).hexdigest()),
                  open(o / "prep.json", "w"), indent=1)
        print(f"{tag}: {n} frames  pinhole f {P[0,0]:.1f}  hfov {hfov:.0f} deg", flush=True)
