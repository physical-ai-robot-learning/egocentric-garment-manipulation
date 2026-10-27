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
hum = box(0.2, 5.55, 2.5, 1.55, "Human demonstrator", "two hand-worn grippers\n(HandUMI, third-party design)\nflattens / folds a garment", "human")
fis = box(3.3, 5.55, 2.85, 1.55, "Wrist fisheye ×2", "jaws, contact,\nlocal cloth view", "sensor")
imu = box(6.5, 5.55, 2.6, 1.55, "Wrist IMU ×2", "gyro + accelerometer", "sensor")
rgbd = box(9.45, 5.55, 3.35, 1.55, "Fixed RGB-D camera", "4 fiducial markers per gripper\n+ depth; global scene context", "sensor")
ax.text(6.5, 7.27, "Recording: all streams stamped by one host clock", ha="center", fontsize=9, color="#566573")
arrow(hum["r"], fis["l"], "human", label="wears", loff=(0, 0.08))

# Row 2: geometry
jaw = box(0.2, 3.15, 2.9, 1.6, "Jaw state", "classifier + jaw-tip keypoints\non the wrist video\n→ continuous aperture", "geom")
pt = box(3.55, 3.15, 2.95, 1.6, "Point tracking", "CoTracker3 on the fixed camera,\nlifted by depth, rigid fit\nto the gripper; gated", "geom")
pose = box(6.95, 3.15, 3.0, 1.6, "Pose fusion", "markers anchor the point tracks;\nIMU acceleration only in gaps;\njaw tip in the robot-base frame", "geom")
cal = box(10.35, 3.15, 2.45, 1.6, "Calibration", "camera ↔ robot base\n(per camera move),\nmarker rig → jaw tip", "calib", dashed=True)

arrow(fis["b"], jaw["t"], "sensor", rad=0.15)
arrow((9.75, 5.55), (5.6, 4.75), "sensor", label="RGB + depth", lpos=0.8, loff=(0.0, 0.08))
arrow((7.6, 5.55), (7.6, 4.75), "sensor", label="rotation;\naccel. in gaps", lpos=0.5, loff=(0.72, -0.25))
arrow((11.1, 5.55), (9.4, 4.75), "sensor", label="markers + depth", lpos=0.5, loff=(0.55, 0.0))
arrow(cal["l"], pose["r"], "calib", dashed=True)
arrow(pt["r"], pose["l"], "geom", label="tracks", loff=(0, 0.1))

# Row 3: data and policy
traj = box(0.2, 0.55, 2.75, 1.75, "Graded episodes", "jaw-tip SE(3) + aperture;\nstill start / end trimmed;\nGOLD / SILVER / REJECT", "traj")
lab = box(3.35, 0.55, 2.85, 1.75, "Training labels", "Cartesian pose before IK,\nstored absolute; relative\nchunks built in the loader", "traj")
pol = box(6.6, 0.55, 2.6, 1.75, "Policy", "pretrained on public\ncloth data, fine-tuned\non GOLD demos\n(training starts Oct 27)", "future")
robot = box(9.6, 0.55, 3.2, 1.75, "Bimanual robot → garment", "receding horizon: FK → compose\n→ limiter → IK → safety checks\n→ a few steps, then re-plan", "robot")
arrow(jaw["b"], traj["t"], "geom")
arrow(pose["b"], traj["t"], "geom", rad=-0.12)
arrow(traj["r"], lab["l"], "traj")
arrow(lab["r"], pol["l"], "traj", label="train", loff=(0, 0.1))
arrow(pol["r"], robot["l"], "robot", label="relative motion", loff=(0, 0.1))

# Superseded (grey, dashed)
ax.add_patch(FancyBboxPatch((0.2, -0.85), 12.6, 0.62, boxstyle="round,pad=0.02,rounding_size=0.1",
                            fc="#f4f4f4", ec="#999999", lw=1.2, ls="--"))
ax.text(6.5, -0.54, "No longer used:  wrist-camera SLAM gap filler (lost tracking at grasps)  ·  "
        "IK → FK labels  ·  whole-demo open-loop replay (finished, Oct 8–22)", ha="center", va="center",
        fontsize=8.6, color="#777777", style="italic")

out = Path(__file__).resolve().parent
fig.savefig(out / "architecture.png", dpi=150, bbox_inches="tight", facecolor="white")
fig.savefig(out / "architecture.pdf", bbox_inches="tight", facecolor="white")
print("wrote", out / "architecture.png")
