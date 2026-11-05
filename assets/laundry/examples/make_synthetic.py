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
through laundry_reference.robotize (default settings), then delayed by 0.25 s,
scaled by 0.97 about its start and given tracking noise, so that the parity
tool has something to compare against.

Outputs (next to this file): synthetic_bimanual_demo.csv / .npz, synthetic_robot_replay.csv.
Usage:  PYTHONPATH=src python assets/laundry/examples/make_synthetic.py   (deterministic, seed 0; or after `pip install -e .`)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from laundry_reference import trajio
from laundry_reference.robotize import SCALE_KEY, robotize

HERE = Path(__file__).resolve().parent
FPS = 30.0
DURATION = 30.0
SEED = 0
# measurement noise of the human demo
POSITION_NOISE_M = 0.0015
ROTATION_NOISE_DEG = 1.0
# random short dropouts: count, start range [s], length range [s]
SHORT_DROPOUTS = 6
SHORT_DROPOUT_START_S = (1.0, 28.0)
SHORT_DROPOUT_LEN_S = (0.07, 0.25)
# the long left-hand dropout the gap filler must refuse (2.0 s)
LONG_DROPOUT_S = (11.0, 13.0)
# markers hidden around the grasp and the release (palm down), per hand
GRASP_DROPOUTS_S = {"R": ((6.6, 7.6), (16.2, 16.9)), "L": ((6.9, 7.5), (16.4, 16.7))}
# jaw closed in [JAW_CLOSE_S, JAW_OPEN_S)
JAW_CLOSE_S = 7.2
JAW_OPEN_S = 17.0
# simulated replay: delay, amplitude about the start, tracking noise
REPLAY_LAG_S = 0.25
REPLAY_AMPLITUDE = 0.97
REPLAY_POSITION_NOISE_M = 0.002
REPLAY_ROTATION_NOISE_DEG = 0.8
TOOL_DOWN_DEG = 180


def minJerk(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def keyframePath(t: np.ndarray, keys: list[tuple[float, np.ndarray]]) -> np.ndarray:
    """Piecewise minimum-jerk interpolation through (time, value) keyframes."""
    tOut = np.empty((len(t), len(keys[0][1])))
    for tI, tTime in enumerate(t):
        if tTime <= keys[0][0]:
            tOut[tI] = keys[0][1]
            continue
        for (tT0, tV0), (tT1, tV1) in zip(keys[:-1], keys[1:]):
            if tT0 <= tTime <= tT1:
                tOut[tI] = tV0 + (tV1 - tV0) * minJerk(np.array((tTime - tT0) / (tT1 - tT0)))
                break
        else:
            tOut[tI] = keys[-1][1]
    return tOut


def handKeys(side: float, cross: bool) -> list[tuple[float, np.ndarray]]:
    """Keyframes [x, y, z, roll, pitch, yaw] (m, deg). side=-1 right, +1 left."""
    tKeys = [
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
        tKeys.insert(10, (22.5, [0.28, 0.015, 0.05, 0, 10, -10]))
        tKeys.insert(11, (23.5, [0.28, 0.010, 0.05, 0, 10, -10]))
    return [(tTime, np.asarray(tValue, dtype=float)) for tTime, tValue in tKeys]


def jawSignal(t: np.ndarray) -> np.ndarray:
    tJaw = np.ones_like(t)
    tJaw[(t >= JAW_CLOSE_S) & (t < JAW_OPEN_S)] = 0.0
    return tJaw


def main() -> int:
    tRng = np.random.default_rng(SEED)
    tT = np.arange(0.0, DURATION, 1.0 / FPS)
    tHuman: dict[str, np.ndarray] = {"t": tT}
    tTruth: dict[str, np.ndarray] = {"t": tT}
    for tHand, tSide in (("R", -1.0), ("L", 1.0)):
        tKeyframes = keyframePath(tT, handKeys(tSide, cross=(tHand == "R")))
        tP = tKeyframes[:, :3]
        # tool pointing down (180 deg about x), then roll/pitch/yaw perturbations
        tRot = Rotation.from_euler("x", TOOL_DOWN_DEG, degrees=True) * Rotation.from_euler("xyz", tKeyframes[:, 3:], degrees=True)
        trajio.setPose(tTruth, tHand, tP, tRot.as_quat())
        tTruth[f"{tHand}_jaw"] = jawSignal(tT)
        tTruth[f"{tHand}_visible"] = np.ones_like(tT)

        # measured version: noise + dropouts
        tPNoisy = tP + tRng.normal(0, POSITION_NOISE_M, tP.shape)
        tRNoisy = Rotation.from_rotvec(tRng.normal(0, np.deg2rad(ROTATION_NOISE_DEG), (len(tT), 3))) * tRot
        tVisible = np.ones(len(tT), dtype=bool)
        for tA, tB in GRASP_DROPOUTS_S[tHand]:
            tVisible[(tT >= tA) & (tT < tB)] = False
        for _ in range(SHORT_DROPOUTS):
            tStart = tRng.uniform(*SHORT_DROPOUT_START_S)
            tVisible[(tT >= tStart) & (tT < tStart + tRng.uniform(*SHORT_DROPOUT_LEN_S))] = False
        if tHand == "L":
            tVisible[(tT >= LONG_DROPOUT_S[0]) & (tT < LONG_DROPOUT_S[1])] = False
        tPNoisy[~tVisible] = np.nan
        tQ = tRNoisy.as_quat()
        tQ[~tVisible] = np.nan
        trajio.setPose(tHuman, tHand, tPNoisy, tQ)
        tHuman[f"{tHand}_jaw"] = jawSignal(tT)
        tHuman[f"{tHand}_visible"] = tVisible.astype(float)

    trajio.save(HERE / "synthetic_bimanual_demo.csv", tHuman)
    np.savez_compressed(HERE / "synthetic_bimanual_demo.npz", **tHuman)

    # simulated robot replay of the robotized noise-free motion: delayed, 3 % smaller amplitude about the start, tracking noise
    tTruth = robotize(tTruth)
    tTruth.pop(SCALE_KEY)
    tRobot: dict[str, np.ndarray] = {"t": tT}
    for tHand in ("R", "L"):
        tP = trajio.position(tTruth, tHand)
        tQ = Rotation.from_quat(trajio.quat(tTruth, tHand))
        tIdx = np.clip(np.round((tT - REPLAY_LAG_S) * FPS).astype(int), 0, len(tT) - 1)
        tP = tP[tIdx]
        tP = tP[0] + REPLAY_AMPLITUDE * (tP - tP[0])
        tP += tRng.normal(0, REPLAY_POSITION_NOISE_M, tP.shape)
        tR = Rotation.from_rotvec(tRng.normal(0, np.deg2rad(REPLAY_ROTATION_NOISE_DEG), (len(tT), 3))) * tQ[tIdx]
        trajio.setPose(tRobot, tHand, tP, tR.as_quat())
        tRobot[f"{tHand}_jaw"] = tTruth[f"{tHand}_jaw"][tIdx]
        tRobot[f"{tHand}_visible"] = np.ones_like(tT)
    trajio.save(HERE / "synthetic_robot_replay.csv", tRobot)
    print("wrote", HERE / "synthetic_bimanual_demo.csv", "/ .npz and", HERE / "synthetic_robot_replay.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
