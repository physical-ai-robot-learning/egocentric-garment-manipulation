"""11-11 tag-free QA video: fused (variant Bg, L yaw) TCP + 5 cm x/z axes projected on the Orbbec frames, wrist cams beside."""
import json, subprocess, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path.home() / "ego_collector/datasets/m3_runs/_imu_consistency"; sys.path[:0] = [str(ROOT), str(ROOT / "stitch")]
import core as C, camcal  # noqa
FD = C.FD; take, var, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]); QA = ROOT / "stitch/qa" / var
ONLY = {int(x) for x in sys.argv[4].split(",")} if len(sys.argv) > 4 else None    # optional demo subset (per-demo clips)
TAGS = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}                       # optional {demo: label} shown in the title
tk = FD.TAG / take; I = json.loads((tk / "meta.json").read_text())["intrinsics"]; T = camcal.T_C_B(take); t = np.load(tk / "t_ns.npy").astype(np.int64)
W = {"left": tk / "wrist_left", "right": tk / "wrist"}; WT = {a: np.load(W[a] / "t_ns.npy").astype(np.int64) for a in W}
COL = {"left": (255, 120, 0), "right": (0, 140, 255)}
def proj(q):
    x = T[:3, :3] @ q + T[:3, 3]; return None if x[2] <= 0.05 or not np.isfinite(x).all() else (int(I["fx"] * x[0] / x[2] + I["cx"]), int(I["fy"] * x[1] / x[2] + I["cy"]))
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1272x480", "-r", "25", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", str(out)], stdin=subprocess.PIPE)
for Wd in FD.trims()[take]:
    dm = Wd["demo"]; Z = {}
    if ONLY is not None and dm not in ONLY: continue
    for a in ("left", "right"):
        f = QA / f"{take}_{a}_d{dm:02d}_h-1.npz"
        if f.exists(): Z[a] = np.load(f)
    if not Z: continue          # 11-12: demos without a fused track (excluded / failed) are left out of the QA video
    ks = np.flatnonzero((t >= Wd["w0"]) & (t <= Wd["w1"]))
    for j, k in enumerate(ks):
        img = cv2.imread(str(tk / f"color_{k:05d}.jpg"))
        for a, z in Z.items():
            tz = z["t_ns"].astype(np.int64); i = int(np.clip(np.searchsorted(tz, t[k]), 0, len(tz) - 1))
            if abs(tz[i] - t[k]) > 1e8: continue
            lo = max(0, i - 30); pts = [proj(q) for q in z["p_B"][lo:i + 1]]
            for u in range(1, len(pts)):
                if pts[u - 1] and pts[u]: cv2.line(img, pts[u - 1], pts[u], COL[a], 2, cv2.LINE_AA)
            p0 = z["p_B"][i]; Rm = z["R_B_tcp"][i]; c0 = proj(p0)
            if c0:
                for ax, cc in ((0, (0, 0, 255)), (2, (0, 200, 0))):
                    c1 = proj(p0 + Rm[:, ax] * 0.06)
                    if c1: cv2.arrowedLine(img, c0, c1, cc, 2, cv2.LINE_AA, tipLength=0.25)
                cv2.circle(img, c0, 5, COL[a], -1)
        cv2.putText(img, f"{take} d{dm} {TAGS.get(str(dm), '')} t={(t[k] - Wd['w0']) / 1e9:4.1f}s  tag-free ({var})  red=x  green=z", (8, 22), 0, 0.55, (255, 255, 255), 2)
        side = []
        for a in ("left", "right"):
            i = int(np.clip(np.searchsorted(WT[a], t[k]), 0, len(WT[a]) - 1)); w = cv2.imread(str(W[a] / f"wrist_{i:05d}.jpg"))
            w = cv2.resize(w, (424, 240)) if w is not None else np.zeros((240, 424, 3), np.uint8); cv2.putText(w, f"{a} wrist", (6, 20), 0, 0.6, COL[a], 2); side.append(w)
        fr = np.hstack([cv2.resize(img, (848, 480)), np.vstack(side)]); ff.stdin.write(fr.tobytes())
ff.stdin.close(); ff.wait(); print("video", out)
