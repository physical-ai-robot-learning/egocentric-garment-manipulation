"""Generate the SYNTHETIC example trajectories in this folder.

Nothing here comes from a real recording. The motion is hand-designed: two
hands start at rest, reach for the two near corners of a (virtual) garment,
close their jaws, lift, carry the corners forward and toward the middle (a
half fold), release and retreat. To exercise the tools, the generator adds

* marker dropouts around every grasp (the hand turns palm-down and hides the
  markers, as happens with real hand-worn grippers), a few random short
  dropouts and one long dropout that should NOT be filled;
* a short excursion of the right hand across the body midline;
* jaw-tip heights slightly below the table plane during grasps;
* ~1.5 mm position noise and ~1 deg orientation noise on measured frames.

It also writes a simulated "robot replay": the noise-free motion is passed
through scripts/robotize.py (default settings), then delayed by 0.25 s,
scaled by 0.97 about its start and given tracking noise, so that the parity
tool has something to compare against.

Usage:  python examples/make_synthetic.py   (deterministic, seed 0)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import trajio  # noqa: E402
from robotize import robotize  # noqa: E402

FPS = 30.0
DURATION = 30.0


def min_jerk(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def keyframe_path(t: np.ndarray, keys: list[tuple[float, np.ndarray]]) -> np.ndarray:
    """Piecewise minimum-jerk interpolation through (time, value) keyframes."""
    out = np.empty((len(t), len(keys[0][1])))
    for i, ti in enumerate(t):
        if ti <= keys[0][0]:
            out[i] = keys[0][1]
            continue
        for (t0, v0), (t1, v1) in zip(keys[:-1], keys[1:]):
            if t0 <= ti <= t1:
                out[i] = v0 + (v1 - v0) * min_jerk(np.array((ti - t0) / (t1 - t0)))
                break
        else:
            out[i] = keys[-1][1]
    return out


def hand_keys(side: float, cross: bool) -> list[tuple[float, np.ndarray]]:
    """Keyframes [x, y, z, roll, pitch, yaw] (m, deg). side=-1 right, +1 left."""
    k = [
        (0.0, [0.30, 0.16 * side, 0.06, 0, 0, 0]),
        (2.0, [0.30, 0.16 * side, 0.06, 0, 0, 0]),
        (5.0, [0.42, 0.21 * side, 0.03, 0, 25, 15 * side]),   # approach corner
        (6.5, [0.43, 0.22 * side, -0.004, 0, 35, 20 * side]),  # down (slightly below table)
        (8.0, [0.43, 0.22 * side, -0.004, 0, 35, 20 * side]),  # grasp
        (10.0, [0.40, 0.18 * side, 0.14, 0, 15, 10 * side]),  # lift
        (14.0, [0.24, 0.06 * side, 0.15, 0, 10, 0]),          # carry forward / inward
        (16.0, [0.20, 0.05 * side, 0.03, 0, 20, 0]),          # lay down
        (17.5, [0.20, 0.05 * side, 0.03, 0, 20, 0]),          # release
        (20.0, [0.24, 0.10 * side, 0.12, 0, 0, 0]),           # retreat
        (26.0, [0.30, 0.16 * side, 0.06, 0, 0, 0]),
        (30.0, [0.30, 0.16 * side, 0.06, 0, 0, 0]),
    ]
    if cross:  # right hand briefly reaches over the midline while smoothing the cloth
        k.insert(10, (22.5, [0.28, 0.015, 0.05, 0, 10, -10]))
        k.insert(11, (23.5, [0.28, 0.010, 0.05, 0, 10, -10]))
    return [(t, np.asarray(v, dtype=float)) for t, v in k]


def jaw_signal(t: np.ndarray) -> np.ndarray:
    jaw = np.ones_like(t)
    jaw[(t >= 7.2) & (t < 17.0)] = 0.0
    return jaw


def main() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(0.0, DURATION, 1.0 / FPS)
    human: dict[str, np.ndarray] = {"t": t}
    truth: dict[str, np.ndarray] = {"t": t}
    for hand, side in (("R", -1.0), ("L", 1.0)):
        kf = keyframe_path(t, hand_keys(side, cross=(hand == "R")))
        p = kf[:, :3]
        # tool pointing down (180 deg about x), then roll/pitch/yaw perturbations
        rot = Rotation.from_euler("x", 180, degrees=True) * Rotation.from_euler(
            "xyz", kf[:, 3:], degrees=True)
        trajio.set_pose(truth, hand, p, rot.as_quat())
        truth[f"{hand}_jaw"] = jaw_signal(t)
        truth[f"{hand}_visible"] = np.ones_like(t)

        # measured version: noise + dropouts
        pn = p + rng.normal(0, 0.0015, p.shape)
        rn = Rotation.from_rotvec(rng.normal(0, np.deg2rad(1.0), (len(t), 3))) * rot
        vis = np.ones(len(t), dtype=bool)
        # markers hidden around the grasp and the release (palm down)
        for a, b in ((6.6, 7.6), (16.2, 16.9)) if hand == "R" else ((6.9, 7.5), (16.4, 16.7)):
            vis[(t >= a) & (t < b)] = False
        # random short dropouts
        for _ in range(6):
            s = rng.uniform(1.0, 28.0)
            vis[(t >= s) & (t < s + rng.uniform(0.07, 0.25))] = False
        # one long dropout that the gap filler must refuse (2.0 s)
        if hand == "L":
            vis[(t >= 11.0) & (t < 13.0)] = False
        pn[~vis] = np.nan
        q = rn.as_quat()
        q[~vis] = np.nan
        trajio.set_pose(human, hand, pn, q)
        human[f"{hand}_jaw"] = jaw_signal(t)
        human[f"{hand}_visible"] = vis.astype(float)

    trajio.save(HERE / "synthetic_bimanual_demo.csv", human)
    np.savez_compressed(HERE / "synthetic_bimanual_demo.npz", **human)

    # simulated robot replay of the robotized noise-free motion: delayed, 3 % smaller amplitude
    # about the start, tracking noise
    truth = robotize(truth)
    truth.pop("_scale")
    lag = 0.25
    robot: dict[str, np.ndarray] = {"t": t}
    for hand in ("R", "L"):
        p = trajio.position(truth, hand)
        q = Rotation.from_quat(trajio.quat(truth, hand))
        idx = np.clip(np.round((t - lag) * FPS).astype(int), 0, len(t) - 1)
        p = p[idx]
        p = p[0] + 0.97 * (p - p[0])
        p += rng.normal(0, 0.002, p.shape)
        r = Rotation.from_rotvec(rng.normal(0, np.deg2rad(0.8), (len(t), 3))) * q[idx]
        trajio.set_pose(robot, hand, p, r.as_quat())
        robot[f"{hand}_jaw"] = truth[f"{hand}_jaw"][idx]
        robot[f"{hand}_visible"] = np.ones_like(t)
    trajio.save(HERE / "synthetic_robot_replay.csv", robot)
    print("wrote", HERE / "synthetic_bimanual_demo.csv", "/ .npz and", HERE / "synthetic_robot_replay.csv")


if __name__ == "__main__":
    main()
