"""Calibration files the dataset stage reads from the recording workspace (dataset stage).

GRIPPER_ALIGN  per rig label: R_U_E, the UMI rig -> reBot gripper EEF axes rotation (R_B_E = R_B_tcp @ R_U_E, as umi_replay_bi)
JAW_CALIB      jaw keypoint calibration v002; its thresholds are pinned (coordinator 10-22) and checked before any conversion
"""
from __future__ import annotations

import json

import numpy as np

from laundry_common import paths as egopaths

GRIPPER_ALIGN = {"right": egopaths.DATA_CALIB / "umi_gripper_align_v001.json",
                 "left": egopaths.DATA_CALIB / "umi_gripper_align_left_v001.json"}
JAW_CALIB = egopaths.DATA_CALIB / "jaw_kpt_calib_v002.json"
JAW_THRESH_EXPECTED = {"left": 191.0, "right": 150.0}     # coordinator 10-22: fixed v002 calibration


class JawCalibrationError(ValueError):
    """The jaw keypoint calibration thresholds differ from the pinned v002 values."""


def rigToEef(rig: str) -> np.ndarray:
    """R_U_E of a rig label (3x3)"""
    return np.array(json.loads(GRIPPER_ALIGN[rig].read_text())["R_U_E"])


def checkJawCalib() -> dict:
    """-> dict(file, threshold_px) of JAW_CALIB; JawCalibrationError when its thresholds are not the pinned ones (was
    check_jaw_calib)"""
    tCalib = json.loads(JAW_CALIB.read_text())
    tGot = {tH: float(tCalib[tH]["threshold_px"]) for tH in JAW_THRESH_EXPECTED}
    if tGot != JAW_THRESH_EXPECTED:
        raise JawCalibrationError(f"jaw calibration {JAW_CALIB.name} thresholds {tGot} != expected {JAW_THRESH_EXPECTED}")
    return dict(file=str(JAW_CALIB), threshold_px=tGot)
