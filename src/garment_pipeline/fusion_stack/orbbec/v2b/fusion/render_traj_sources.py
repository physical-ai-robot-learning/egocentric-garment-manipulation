"""10-27 README GIF: fused jaw-tip paths of one demo coloured by the source of each frame (same layout as the 10-21 SLAM-era GIF).
Left: Orbbec RGB with the path so far projected per arm; right: both arms in the robot base frame (mm).
Sources: green = a gripper marker is visible (markers + CoTracker + depth), blue = CoTracker points only (no marker),
orange = CoTracker gap (IMU acceleration inside the gap).
    python render_traj_sources.py --pack <ctpack> --demo TAKE:demo --out x.gif [--var fus_Cg_g] [--fps 5] [--speed 1.5]"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path
import cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
from garment_pipeline.fusion_stack import core as C  # noqa
from garment_pipeline.fusion_stack import camcal  # noqa
from garment_pipeline.fusion_stack.orbbec.v2b.fusion.render_fusion_qa import interp, load  # noqa
FD = C.FD
ARMS = ("left", "right")
SRC_BGR = {0: (60, 200, 60), 1: (235, 140, 50), 2: (30, 140, 255)}            # marker / CoTracker only / IMU in gap
SRC_RGB = {k: tuple(c / 255 for c in v[::-1]) for k, v in SRC_BGR.items()}
LABELS = ["AprilTag markers + CoTracker + depth", "CoTracker points only (marker hidden)", "IMU inside a CoTracker gap"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pack", required=True); ap.add_argument("--demo", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--var", default="fus_Cg_g"); ap.add_argument("--fps", type=float, default=5); ap.add_argument("--speed", type=float, default=1.5)
    ap.add_argument("--width", type=int, default=1080); a = ap.parse_args()
    take, demo = a.demo.split(":"); demo = int(demo)
    T_C_B = camcal.T_C_B(take); tk = FD.TAG / take; I = json.loads((tk / "meta.json").read_text())["intrinsics"]
    W = {w["demo"]: w for w in FD.trims()[take]}[demo]
    S = {}
    for arm in ARMS:
        D = FD.load_arm(take, arm, with_slam=False); t = D["t"]; k = np.flatnonzero((t >= W["w0"]) & (t <= W["w1"])); tt = t[k]
        prod = load(a.var, take, arm, demo, -1); p = interp(prod["t_ns"], prod["p_B"], tt)
        z = np.load(Path(a.pack) / f"{take}_d{demo:02d}" / f"solve2_{arm}.npz"); tct = z["t"].astype(np.int64); okc = z["conf"] > 0
        gap = ~np.interp(tt, tct, okc.astype(float)).round().astype(bool) | (tt < tct[0]) | (tt > tct[-1])
        src = np.where(gap, 2, np.where(D["ntags"][k] >= 1, 0, 1))
        S[arm] = dict(k=k, tt=tt, p=p, src=src)
    k = S["left"]["k"]; tt = S["left"]["tt"]; ts = (tt - tt[0]) / 1e9
    def proj(q):
        if not np.isfinite(q[0]): return None
        x = T_C_B[:3, :3] @ q + T_C_B[:3, 3]
        return None if x[2] <= 0.05 else (int(I["fx"] * x[0] / x[2] + I["cx"]), int(I["fy"] * x[1] / x[2] + I["cy"]))
    P = {arm: [proj(q) for q in S[arm]["p"]] for arm in ARMS}
    allp = np.concatenate([S[arm]["p"] for arm in ARMS]) * 1000; lo = np.nanmin(allp, 0); hi = np.nanmax(allp, 0)
    frac = {arm: {s: float(np.mean(S[arm]["src"] == s)) for s in (0, 1, 2)} for arm in ARMS}
    print("source fractions", {arm: {LABELS[s].split()[0]: round(v, 3) for s, v in f.items()} for arm, f in frac.items()})
    step = max(1, int(round(len(ts) / (ts[-1] - ts[0]) * a.speed / a.fps))); idx = list(range(0, len(ts), step))
    tmp = Path(tempfile.mkdtemp())
    for n, j in enumerate(idx):
        img = cv2.imread(str(tk / f"color_{k[j]:05d}.jpg"))
        for arm in ARMS:
            s = S[arm]; pts = P[arm]
            for i in range(1, j + 1):
                if pts[i - 1] and pts[i]: cv2.line(img, pts[i - 1], pts[i], SRC_BGR[int(s["src"][i])], 3, cv2.LINE_AA)
            if pts[j]: cv2.circle(img, pts[j], 7, (255, 255, 255), -1); cv2.circle(img, pts[j], 7, SRC_BGR[int(s["src"][j])], 3)
        cv2.putText(img, f"t = {ts[j]:4.1f} s", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        for r, lab in enumerate(LABELS):
            y = 62 + 22 * r; cv2.line(img, (14, y - 5), (40, y - 5), SRC_BGR[r], 4); cv2.putText(img, lab, (48, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        fig = plt.figure(figsize=(5.2, 4.2), dpi=100); ax = fig.add_subplot(projection="3d")
        for arm in ARMS:
            s = S[arm]; q = s["p"][: j + 1] * 1000
            sr = s["src"][: j + 1]; cut = np.flatnonzero(np.diff(sr)) + 1
            for a0, a1 in zip(np.r_[0, cut], np.r_[cut, len(sr)]):                 # one line per same-source run (joined to the next point)
                seg = q[a0:min(a1 + 1, len(q))]; ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=SRC_RGB[int(sr[a0])], lw=1.6)
            if np.isfinite(q[-1, 0]): ax.scatter(*q[-1], s=40, color="k"); ax.text(*q[-1], f" {arm} hand", fontsize=8, weight="bold")
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(min(lo[2], 0), hi[2])
        ax.set_xlabel("x [mm]", fontsize=7); ax.set_ylabel("y [mm]", fontsize=7); ax.set_zlabel("z [mm]", fontsize=7); ax.tick_params(labelsize=6)
        ax.set_title("jaw-tip paths in the robot base frame", fontsize=9); ax.view_init(24, -135 + 30 * n / max(1, len(idx) - 1))
        fig.tight_layout(); fig.canvas.draw(); pl = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[..., :3], cv2.COLOR_RGB2BGR); plt.close(fig)
        h = 420; img = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h)); pl = cv2.resize(pl, (int(pl.shape[1] * h / pl.shape[0]), h))
        fr = np.hstack([img, np.full((h, 8, 3), 255, np.uint8), pl]); fr = cv2.resize(fr, (a.width, int(fr.shape[0] * a.width / fr.shape[1])))
        cv2.imwrite(str(tmp / f"f_{n:04d}.png"), fr)
    pal = tmp / "pal.png"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(tmp / "f_%04d.png"), "-vf", "palettegen=max_colors=200", str(pal)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", str(tmp / "f_%04d.png"), "-i", str(pal), "-lavfi", "paletteuse=dither=bayer:bayer_scale=4", "-loop", "0", a.out], check=True)
    print("wrote", a.out, len(idx), "frames")


if __name__ == "__main__":
    main()
