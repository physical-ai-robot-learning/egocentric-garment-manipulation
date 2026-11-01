"""Synthetic 21-landmark hands (MediaPipe order) for tests. Hand-local frame of the RIGHT hand: x = radial (thumb side),
y = wrist -> fingers, z = out of the palm. The left hand is the exact mirror image (x -> -x in its own local frame), so the
canonical gripper frame of BOTH hands must come out as the local frame itself (R = I) -> this is the L/R convention test."""
import numpy as np


def hand_local(aperture=0.04, side="right"):
    c = np.array([0.015, 0.11, 0.045])                                    # pinch point in front of the palm
    u = np.array([0.9, 0.0, 0.44]); u /= np.linalg.norm(u)                 # thumb tip on the radial side of the pinch
    k = np.zeros((21, 3))
    k[1:4] = [[0.02, 0.02, 0.01], [0.035, 0.045, 0.02], [0.04, 0.07, 0.03]]; k[4] = c + 0.5 * aperture * u
    k[5:8] = [[0.03, 0.08, 0.0], [0.03, 0.10, 0.02], [0.025, 0.11, 0.035]]; k[8] = c - 0.5 * aperture * u
    for j, x in zip((9, 13, 17), (0.0, -0.018, -0.035)):
        y0 = 0.085 if j == 9 else (0.08 if j == 13 else 0.07)
        k[j:j + 4] = [[x, y0, 0], [x, y0 + 0.03, 0.01], [x, y0 + 0.05, 0.02], [x, y0 + 0.065, 0.03]]
    if side == "left":
        k[:, 0] *= -1
    return k


def place(k_local, T):
    return k_local @ T[:3, :3].T + T[:3, 3]


def pose(rotvec=(0, 0, 0), t=(0, 0.02, 0.12)):
    from scipy.spatial.transform import Rotation as R
    T = np.eye(4); T[:3, :3] = R.from_rotvec(rotvec).as_matrix(); T[:3, 3] = t
    return T
