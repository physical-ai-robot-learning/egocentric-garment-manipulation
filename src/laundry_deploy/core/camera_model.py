"""Orbbec camera model of the deploy display (UI overlay and run video, never control): calibration files and the projection
of base-frame points to Orbbec pixels.

10-29: the Orbbec was raised on 10-26 -> the overlay extrinsic is v005 (fixed tags + table plane, 10-26 12:08; FK TCP lands on
the real jaw tips, checked on the live frame), intrinsics still from v001 (v002+ store no intrinsics).
"""

from __future__ import annotations
import json

import numpy as np

from laundry_common import paths as egopaths

ROBOT_CAM_FILE = egopaths.DATA_CALIB / "robot_cam_v001.json"  # intrinsics (848x480)
ROBOT_CAM_EXT_FILE = egopaths.DATA_CALIB / "robot_cam_v005.json"  # extrinsic T_C_B
MIN_DEPTH_M = 0.05  # points closer than this (or behind the camera) do not project


def project(camFromBase: np.ndarray, intrinsics: dict, pointsBase) -> np.ndarray:
    """Base-frame points (N, 3) -> Orbbec pixels (N, 2) (848x480 intrinsics), points behind the camera -> nan."""
    tP = (camFromBase[:3, :3] @ np.asarray(pointsBase).T).T + camFromBase[:3, 3]
    tZ = tP[:, 2:3]
    tUv = np.c_[
        intrinsics["fx"] * tP[:, 0] / tP[:, 2] + intrinsics["cx"], intrinsics["fy"] * tP[:, 1] / tP[:, 2] + intrinsics["cy"]
    ]
    tUv[(tZ[:, 0] <= MIN_DEPTH_M)] = np.nan
    return tUv


def robotCam():
    """-> (T_C_B (4, 4), intrinsics dict)"""
    tIntr = json.loads(ROBOT_CAM_FILE.read_text())
    return np.asarray(json.loads(ROBOT_CAM_EXT_FILE.read_text())["T_C_B"]), tIntr["intrinsics"]
