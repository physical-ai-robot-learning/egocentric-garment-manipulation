"""10-26 per-take fixed-Orbbec calibration.  The camera was RAISED on 2026-10-26 ~11:13 (takes from 20261026_111328 on):
  takes <  RAISED  -> configs/calibration/robot_cam_v001.json   (10-20 robot calibration)
  takes >= RAISED  -> configs/calibration/robot_cam_v002.json   (old 6 cm tag 401 + table depth plane; fixed 70 mm tags 400/401 then
                      re-measured in it -> start_anchors_v003_70mm.json; scripts/robot_cam_from_anchors.py)
  takes >= 20261026_115441 -> robot_cam_v003.json   (camera raised again; start_anchors_v003_70mm + table plane)
Every script that lifts Orbbec depth into the robot base or projects into the image must take the matrix from here, per take."""
import json
from pathlib import Path
import numpy as np

from egogarment import paths as egopaths  # noqa: E402
CAL = egopaths.DATA_CALIB
RAISED = "20261026_111328"
SEGMENTS = [("00000000_000000", "robot_cam_v001.json"),     # 10-20 robot calibration
            ("20261026_111328", "robot_cam_v002.json"),     # raised once (6 cm 401 + table plane)
            ("20261026_115441", "robot_cam_v003.json"),     # raised again 10-26 ~11:54 (70 mm 400/401 v003 + table plane)
            ("20261026_120052", "robot_cam_v004.json"),     # raised to ~113 cm above the desk 10-26 ~12:00 (same method; take moved < frame 330)
            ("20261026_120504", "robot_cam_v005.json")]     # 113 cm, re-aimed 10-26 ~12:05 (take moved < frame 105)


def path(take: str) -> Path:
    f = [name for start, name in SEGMENTS if str(take) >= start][-1]; return CAL / f


def T_B_C(take: str) -> np.ndarray:
    return np.array(json.loads(path(take).read_text())["T_B_C"])


def T_C_B(take: str) -> np.ndarray:
    return np.linalg.inv(T_B_C(take))
