"""Raw per-frame AprilTag rig rotation (PnP, no IMU) in the robot base for both rigs of a take (was _fgo_1023/tools/
tag_rot_extract_v002.py, 10-24: copy of tag_rot_extract.py using the v002 rig config, which adds tags 220 / 221 for takes recorded
with them, e.g. 20261023_155004).  Used by fgo.arm_data as the rotation reference / rotation factors.
Input: datasets/tag_rig/<take> (meta.json intrinsics, t_ns.npy, color_NNNNN.jpg), the rig geometry umi_tag_rig_v002.json, the per-take
camera calibration.  Output: m3_runs/_fgo_1023/tagrot/<take>.npz: t_ns, R_<rig> (n,3,3, NaN where not seen), n_<rig> (tags used),
rmse_<rig> (px).
    python -m laundry_tracking.tag_rotations datasets/tag_rig/<take>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from laundry_common import paths
from laundry_shared.rebot_prep.parity.tag_rig_pose import rig_pose

from .camera_calibration import baseFromCamera

# 11-04 layout move: read from <EGO_DATA_ROOT>/configs/laundry/calibration (the recording workspace keeps it in configs/calibration)
RIG_CONFIG = paths.DATA_CALIB / "umi_tag_rig_v002.json"
RIG_NAMES = {"right": "UMI 오른손", "left": "UMI 왼손"}
TAG_SIZE_M = 0.048
MAX_RMSE_PX = 2.0                  # rig poses with a larger reprojection RMSE are dropped


def tagDetector() -> cv2.aruco.ArucoDetector:
    tParams = cv2.aruco.DetectorParameters()
    tParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), tParams)


def extractTagRotations(take: Path) -> dict:
    """per-frame rig rotation in the robot base for both rigs -> the npz content"""
    tMeta = json.loads((take / "meta.json").read_text())
    tIntr = tMeta["intrinsics"]
    tK = np.array([[tIntr["fx"], 0, tIntr["cx"]], [0, tIntr["fy"], tIntr["cy"]], [0, 0, 1.0]])
    tCfg = json.loads(RIG_CONFIG.read_text())
    tTbc = baseFromCamera(take.name)                            # 10-26: per-take camera calibration
    tRigTags = {tSide: {int(tId): np.array(tT) for tId, tT in tCfg[tName]["T_rig_tag"].items()} for tSide, tName in RIG_NAMES.items()}
    tDet = tagDetector()
    tTimes = np.load(take / "t_ns.npy")
    tN = len(tTimes)
    tOut = dict(t_ns=tTimes)
    for tSide in RIG_NAMES:
        tOut[f"R_{tSide}"] = np.full((tN, 3, 3), np.nan)
        tOut[f"n_{tSide}"] = np.zeros(tN, int)
        tOut[f"rmse_{tSide}"] = np.full(tN, np.nan)
    for tFrame in range(tN):
        tImg = cv2.imread(str(take / f"color_{tFrame:05d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if tImg is None:
            continue
        tCorners, tIds, _ = tDet.detectMarkers(tImg)
        for tSide in RIG_NAMES:
            tPose = rig_pose(tCorners, tIds, tRigTags[tSide], tK, TAG_SIZE_M)
            if tPose is None or tPose["rmse_px"] > MAX_RMSE_PX:
                continue
            tOut[f"R_{tSide}"][tFrame] = tTbc[:3, :3] @ tPose["T"][:3, :3]
            tOut[f"n_{tSide}"][tFrame] = len(tPose["tags"])
            tOut[f"rmse_{tSide}"][tFrame] = tPose["rmse_px"]
    return tOut


def main(argv: list[str] | None = None) -> int:
    tAp = argparse.ArgumentParser(description="raw per-frame tag rig rotations of a take -> _fgo_1023/tagrot/<take>.npz")
    tAp.add_argument("take")
    tTake = Path(tAp.parse_args(argv).take)
    tOut = extractTagRotations(tTake)
    np.savez(paths.M3_RUNS / f"_fgo_1023/tagrot/{tTake.name}.npz", **tOut)
    print(tTake.name, len(tOut["t_ns"]), {tSide: int((tOut[f"n_{tSide}"] > 0).sum()) for tSide in RIG_NAMES})
    return 0


if __name__ == "__main__":
    sys.exit(main())
