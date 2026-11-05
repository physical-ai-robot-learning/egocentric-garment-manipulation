"""Per-take fixed-Orbbec calibration (was camcal.py, 10-26): the robot_cam_vNNN.json that holds T_B_C for a take.

The camera was raised / re-aimed several times on 2026-10-26; every take uses the calibration of the last segment that started at
or before it (take names sort by time):
  takes <  20261026_111328 -> robot_cam_v001.json   (10-20 robot calibration)
  takes >= 20261026_111328 -> robot_cam_v002.json   (raised once: old 6 cm tag 401 + table depth plane; fixed 70 mm tags 400/401
                                                     then re-measured in it -> start_anchors_v003_70mm.json, calib/robot_cam_from_anchors)
  takes >= 20261026_115441 -> robot_cam_v003.json   (raised again; start_anchors_v003_70mm + table plane)
  takes >= 20261026_120052 -> robot_cam_v004.json   (~113 cm above the desk; take moved < frame 330)
  takes >= 20261026_120504 -> robot_cam_v005.json   (113 cm, re-aimed; take moved < frame 105)
Every tool that lifts Orbbec depth into the robot base or projects into the image takes the matrix from here, per take.
Files live in laundry_common.paths.DATA_CALIB (configs/calibration of the recording workspace).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from laundry_common import paths

RAISED = "20261026_111328"           # first take after the camera was raised
CALIB_SEGMENTS = (("00000000_000000", "robot_cam_v001.json"),
                  ("20261026_111328", "robot_cam_v002.json"),
                  ("20261026_115441", "robot_cam_v003.json"),
                  ("20261026_120052", "robot_cam_v004.json"),
                  ("20261026_120504", "robot_cam_v005.json"))


def calibPath(take: str) -> Path:
    """robot_cam_vNNN.json for a take (was camcal.path)"""
    tName = [tFile for tStart, tFile in CALIB_SEGMENTS if str(take) >= tStart][-1]
    return paths.DATA_CALIB / tName


def baseFromCamera(take: str) -> np.ndarray:
    """T_B_C (4x4): camera frame -> robot base (was camcal.T_B_C)"""
    return np.array(json.loads(calibPath(take).read_text())["T_B_C"])


def cameraFromBase(take: str) -> np.ndarray:
    """T_C_B = inv(T_B_C) (was camcal.T_C_B)"""
    return np.linalg.inv(baseFromCamera(take))

