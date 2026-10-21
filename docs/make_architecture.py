"""Draw docs/architecture.png (data flow). Usage: python docs/make_architecture.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

C = {  # fill, edge
    "human": ("#fdf2e9", "#ca6f1e"),
    "sensor": ("#eef1f4", "#566573"),
    "geom": ("#e8f1fb", "#2e6db4"),
    "traj": ("#e9f6ec", "#2e8b57"),
    "robot": ("#fdecea", "#c0392b"),
    "calib": ("#fffbe6", "#b7950b"),
    "future": ("#ffffff", "#7d3c98"),
}

fig, ax = plt.subplots(figsize=(13, 8.3))
ax.set_xlim(0, 13)
ax.set_ylim(-0.9, 7.4)
ax.axis("off")


def box(x, y, w, h, title, body, kind, dashed=False):
    f, e = C[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=f, ec=e,
                                lw=1.6, ls="--" if dashed else "-"))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=10.5, weight="bold", color=e)
    ax.text(x + w / 2, y + h - 0.52, body, ha="center", va="top", fontsize=8.4, color="#222", linespacing=1.3)
    return dict(l=(x, y + h / 2), r=(x + w, y + h / 2), t=(x + w / 2, y + h), b=(x + w / 2, y),
                x=x, y=y, w=w, h=h)


def arrow(p, q, kind="geom", dashed=False, label=None, rad=0.0, lpos=0.5, loff=(0, 0.12)):
    e = C[kind][1]
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, lw=1.5, color=e,
                                 ls="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=2, shrinkB=2))
    if label:
        mx, my = p[0] + (q[0] - p[0]) * lpos + loff[0], p[1] + (q[1] - p[1]) * lpos + loff[1]
        ax.text(mx, my, label, ha="center", va="bottom", fontsize=7.8, color=e, style="italic",
                bbox=dict(fc="white", ec="none", pad=0.6))


# Row 1: human and sensors
hum = box(0.2, 5.55, 2.5, 1.55, "Human demonstrator", "two hand-worn grippers\n(HandUMI, third-party design)\nfolds a garment", "human")
fis = box(3.3, 5.55, 2.85, 1.55, "Wrist fisheye ×2", "jaws, contact,\nlocal cloth view", "sensor")
imu = box(6.5, 5.55, 2.6, 1.55, "Wrist IMU ×2", "gyro + accelerometer", "sensor")
rgbd = box(9.45, 5.55, 3.35, 1.55, "Fixed RGB-D camera", "fiducial markers on the grippers\n+ depth; global scene context", "sensor")
ax.text(6.5, 7.27, "Recording: all streams stamped by one host clock", ha="center", fontsize=9, color="#566573")
arrow(hum["r"], fis["l"], "human", label="wears", loff=(0, 0.08))

# Row 2: geometry
jaw = box(0.2, 3.15, 2.9, 1.6, "Jaw state", "image classifier on the\nwrist video → open / closed", "geom")
gap = box(3.55, 3.15, 2.95, 1.6, "Gap filling", "IMU rotation + interpolation;\noptional SLAM filler, accepted\nonly where it agrees", "geom")
pose = box(6.95, 3.15, 3.0, 1.6, "Pose reconstruction", "marker PnP + depth → gripper\nSE(3) → jaw tip, directly in\nthe robot-base world frame", "geom")
cal = box(10.35, 3.15, 2.45, 1.6, "One-time calibration", "camera ↔ robot base,\nstart markers,\nmarker rig → jaw tip", "calib", dashed=True)

arrow(fis["b"], jaw["t"], "sensor", rad=0.15)
arrow(fis["b"], gap["t"], "sensor", dashed=True, label="optional SLAM", lpos=0.55, loff=(0.25, 0.05))
arrow(imu["b"], gap["t"], "sensor", label="rotation in gaps", lpos=0.45, loff=(0.1, 0.05))
arrow(rgbd["b"], pose["t"], "sensor", label="markers + depth", lpos=0.5, loff=(0.25, 0.05))
arrow(cal["l"], pose["r"], "calib", dashed=True)
arrow(pose["l"], gap["r"], "geom", label="gaps", loff=(0, 0.1))

# Row 3: trajectory and robot
traj = box(0.2, 0.55, 2.75, 1.75, "Robot-frame trajectory", "absolute jaw-tip SE(3)\n+ jaw state per frame\n(measured / filled flags)", "traj")
rob = box(3.35, 0.55, 2.85, 1.75, "Robotization", "smooth warps (side, table)\n+ per-frame displacement\n& quality label", "traj")
ik = box(6.6, 0.55, 2.6, 1.75, "IK + safety checks", "joint-limit-aware IK,\nspeed, table, arm-arm\ndistance; plan offline", "robot")
robot = box(9.6, 0.55, 3.2, 1.75, "Bimanual robot → garment", "open-loop replay; human,\ncommand and robot compared\non one clock + same camera", "robot")
arrow(jaw["b"], traj["t"], "geom")
arrow(gap["b"], traj["t"], "geom", rad=-0.1)
arrow(traj["r"], rob["l"], "traj")
arrow(rob["r"], ik["l"], "traj", label="replay (now)", loff=(0, 0.1))
arrow(ik["r"], robot["l"], "robot")

# Future: policy learning (dashed)
ax.add_patch(FancyBboxPatch((4.2, -0.8), 6.4, 0.6, boxstyle="round,pad=0.02,rounding_size=0.1",
                            fc="white", ec=C["future"][1], lw=1.5, ls="--"))
ax.text(7.4, -0.5, "Future: policy learning on relative SE(3) labels  $T_{rel}(t) = T_0^{-1}\\,T(t)$,\n"
        "weighted by the per-frame quality label", ha="center", va="center", fontsize=8.6,
        color=C["future"][1], style="italic")
arrow((rob["x"] + rob["w"] / 2, rob["y"]), (rob["x"] + rob["w"] / 2, -0.2), "future", dashed=True)
arrow((10.6, -0.5), (robot["x"] + robot["w"] / 2, robot["y"]), "future", dashed=True, rad=0.35)

out = Path(__file__).resolve().parent
fig.savefig(out / "architecture.png", dpi=150, bbox_inches="tight", facecolor="white")
fig.savefig(out / "architecture.pdf", bbox_inches="tight", facecolor="white")
print("wrote", out / "architecture.png")
