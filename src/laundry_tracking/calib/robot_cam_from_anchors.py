"""2026-10-26 re-register the fixed Orbbec after it was MOVED, without the robot: T_B_C from the fixed table tags 400 / 401
(start_anchors_v001.json: tag centres + rotations in the robot base, 6 cm, measured 10-21 with robot_cam_v001) plus the table plane
from depth.  Joint least squares over T_C_B (6 DoF):
  - reprojection of the 4 corners of every visible anchor tag in every selected frame (sigma PX_SIG px)
  - table depth points (RANSAC plane inliers of the first selected frame, subsampled): base z = TABLE_Z (sigma Z_SIG m)
Use a STATIC stretch of the take (camera untouched, table empty enough that both tags are seen).  Prints the residuals, the change vs
robot_cam_v001, how high above the table the image top edge reaches at the workspace centre, and writes --out (same keys as v001).
    python -m laundry_tracking.calib.robot_cam_from_anchors datasets/tag_rig/<take> --frames 2400:2828:4 [--out configs/calibration/robot_cam_v002.json]
(was scripts/pose/robot_cam_from_anchors.py of ego_collector; module path laundry_tracking.calib.robot_cam_from_anchors unchanged by the 11-04 refactor)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from laundry_common import paths

PX_SIG = 0.7                       # px, corner reprojection sigma
Z_SIG = 0.002                      # m, table-plane sigma
TABLE_Z = 0.001                    # m, table top in the base frame
DEFAULT_ANCHORS = paths.CONFIGS / "calibration/start_anchors_v001.json"
REFERENCE_CAL = paths.CONFIGS / "calibration/robot_cam_v001.json"
DEPTH_STRIDE = 4
DEPTH_RANGE_M = (0.3, 2.5)
TABLE_NEAR_M = 0.03                # PnP-init points within 3 cm of the table plane seed the RANSAC
RANSAC_ITERS = 300
RANSAC_SEED = 0
RANSAC_INLIER_M = 0.006
MAX_TABLE_POINTS = 3000
HUBER_F_SCALE = 2.0
WORKSPACE_POINTS = ((0.40, 0.0), (0.35, 0.25), (0.35, -0.25))
HEIGHT_SCAN = (0, 1.5, 301)        # m: heights scanned above each workspace point
IMAGE_W, IMAGE_H = 848, 480


def _cameraHeight(rv, tv) -> float:
    """z of the camera centre in the base frame for a PnP solution (rotation vector, translation) of the anchor tags."""
    tRm = R.from_rotvec(np.ravel(rv)).as_matrix()
    return float((-tRm.T @ np.ravel(tv))[2])


def _buildParser() -> argparse.ArgumentParser:
    tAp = argparse.ArgumentParser()
    tAp.add_argument("take")
    tAp.add_argument("--frames", required=True)
    tAp.add_argument("--out", default="")
    tAp.add_argument("--meta", default="", help="meta.json with the intrinsics (a take still recording has none yet)")
    tAp.add_argument("--anchors", default=str(DEFAULT_ANCHORS))
    return tAp


def main(argv: list[str] | None = None) -> int:
    tArgs = _buildParser().parse_args(argv)
    tTake = Path(tArgs.take)
    tIntr = json.loads(Path(tArgs.meta or tTake / "meta.json").read_text())["intrinsics"]
    tK = np.array([[tIntr["fx"], 0, tIntr["cx"]], [0, tIntr["fy"], tIntr["cy"]], [0, 0, 1.0]])
    tAnchors = json.loads(Path(tArgs.anchors).read_text())
    tHalf = tAnchors["tag_size_m"] / 2
    tObj = np.array([[-tHalf, tHalf, 0], [tHalf, tHalf, 0], [tHalf, -tHalf, 0], [-tHalf, -tHalf, 0]], float)
    tTags = {int(tKk): (np.array(tV["R_B"]), np.array(tV["p_B"])) for tKk, tV in tAnchors["tags"].items()}
    tParams = cv2.aruco.DetectorParameters()
    tParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    tDet = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), tParams)
    tS0, tS1, tSt = (int(tX) for tX in tArgs.frames.split(":"))
    tP3, tP2, tSeen = [], [], {tKk: 0 for tKk in tTags}
    for tFrame in range(tS0, tS1, tSt):
        tGray = cv2.imread(str(tTake / f"color_{tFrame:05d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if tGray is None:
            continue
        tCorners, tIds, _ = tDet.detectMarkers(tGray)
        if tIds is None:
            continue
        for tCc, tId in zip(tCorners, tIds.ravel()):
            if int(tId) in tTags:
                tRB, tPB = tTags[int(tId)]
                tP3.append(tObj @ tRB.T + tPB)
                tP2.append(tCc.reshape(4, 2))
                tSeen[int(tId)] += 1
    if not tP3:
        print(f"anchor tags seen {tSeen} -- none", file=sys.stderr)
        return 1
    if min(tSeen.values()) == 0:
        print(f"WARNING only one anchor tag seen {tSeen}: yaw / x-y rest on a single small tag (provisional)")
    tP3, tP2 = np.concatenate(tP3), np.concatenate(tP2).astype(float)
    # 11-03: the anchors are two coplanar tags -> planar PnP ambiguity; the ITERATIVE init flipped the camera below the table
    # (z ~ 0, 133 deg off, 47 px). Take every IPPE (and ITERATIVE) solution and keep the one with the camera highest above the table.
    tCands = []
    for tFlags in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE):
        try:
            _, tRvs, tTvs, _ = cv2.solvePnPGeneric(tP3, tP2, tK, None, flags=tFlags)
            tCands += list(zip(tRvs, tTvs))
        except cv2.error:
            pass
    tRv, tTv = max(tCands, key=lambda tC: _cameraHeight(*tC))
    print(f"PnP init: camera z {_cameraHeight(tRv, tTv):.3f} m (of {len(tCands)} candidates)")
    # table plane points (camera frame) from the first selected depth frame: RANSAC plane, inliers subsampled
    tD = cv2.imread(str(tTake / f"depth_{tS0:05d}.png"), -1).astype(np.float32) / 1000
    tH, tW = tD.shape
    tVv, tUu = np.mgrid[0:tH:DEPTH_STRIDE, 0:tW:DEPTH_STRIDE]
    tZ = tD[::DEPTH_STRIDE, ::DEPTH_STRIDE]
    tX = np.stack([(tUu - tIntr["cx"]) / tIntr["fx"] * tZ, (tVv - tIntr["cy"]) / tIntr["fy"] * tZ, tZ], -1).reshape(-1, 3)
    tX = tX[(tX[:, 2] > DEPTH_RANGE_M[0]) & (tX[:, 2] < DEPTH_RANGE_M[1])]
    tRc = R.from_rotvec(tRv.ravel()).as_matrix()
    tXB = (tX - tTv.ravel()) @ tRc                                     # camera -> base with the PnP init
    tNear = np.abs(tXB[:, 2] - TABLE_Z) < TABLE_NEAR_M
    tRng = np.random.default_rng(RANSAC_SEED)
    tBest = None
    for _ in range(RANSAC_ITERS):
        tS = tX[tNear][tRng.choice(tNear.sum(), 3, replace=False)]
        tN = np.cross(tS[1] - tS[0], tS[2] - tS[0])
        if np.linalg.norm(tN) < 1e-9:
            continue
        tN /= np.linalg.norm(tN)
        tInl = np.abs((tX - tS[0]) @ tN) < RANSAC_INLIER_M
        if tBest is None or tInl.sum() > tBest.sum():
            tBest = tInl
    tTable = tX[tBest]
    tTable = tTable[tRng.choice(len(tTable), min(MAX_TABLE_POINTS, len(tTable)), replace=False)]

    def _residual(x):
        tRm = R.from_rotvec(x[:3]).as_matrix()
        tT = x[3:]
        tPc = tP3 @ tRm.T + tT
        tPr = tPc[:, :2] / tPc[:, 2:3] * [tIntr["fx"], tIntr["fy"]] + [tIntr["cx"], tIntr["cy"]]
        tZb = ((tTable - tT) @ tRm)[:, 2] - TABLE_Z
        return np.r_[((tPr - tP2) / PX_SIG).ravel(), tZb / Z_SIG]
    tX0 = np.r_[tRv.ravel(), tTv.ravel()]
    tSol = least_squares(_residual, tX0, loss="huber", f_scale=HUBER_F_SCALE)
    tRm = R.from_rotvec(tSol.x[:3]).as_matrix()
    tT = tSol.x[3:]
    tTcb = np.eye(4)
    tTcb[:3, :3] = tRm
    tTcb[:3, 3] = tT
    tTbc = np.linalg.inv(tTcb)
    tR = _residual(tSol.x)
    tN2 = len(tP2) * 2
    tRp = np.abs(tR[:tN2]) * PX_SIG
    tRz = np.abs(tR[tN2:]) * Z_SIG * 1000
    tOld = np.array(json.loads(REFERENCE_CAL.read_text())["T_B_C"])
    tDr = np.degrees(R.from_matrix(tOld[:3, :3].T @ tTbc[:3, :3]).magnitude())
    tDt = np.linalg.norm(tOld[:3, 3] - tTbc[:3, 3]) * 1000
    print(f"anchor detections {tSeen}, {len(tP2) // 4} tag views; table points {len(tTable)}")
    print(f"reprojection |px| p50/p95 {np.median(tRp):.2f}/{np.percentile(tRp, 95):.2f}; table plane |z| p50/p95 {np.median(tRz):.1f}/{np.percentile(tRz, 95):.1f} mm")
    print(f"camera in base: {np.round(tTbc[:3, 3], 3)} m (v001 {np.round(tOld[:3, 3], 3)}), moved {tDt:.0f} mm, rotated {tDr:.1f} deg")
    for tXw, tYw in WORKSPACE_POINTS:                                 # how high above the table the image TOP edge reaches over a workspace point
        tHs = np.linspace(*HEIGHT_SCAN)
        tPts = np.stack([np.full_like(tHs, tXw), np.full_like(tHs, tYw), tHs], 1) @ tTcb[:3, :3].T + tTcb[:3, 3]
        tV = tPts[:, 1] / tPts[:, 2] * tIntr["fy"] + tIntr["cy"]
        tU = tPts[:, 0] / tPts[:, 2] * tIntr["fx"] + tIntr["cx"]
        tInside = (tV >= 0) & (tV < IMAGE_H) & (tU >= 0) & (tU < IMAGE_W)
        tTop = tHs[tInside].max() if tInside.any() else float("nan")
        print(f"  above base point ({tXw:.2f}, {tYw:+.2f}): visible up to z = {tTop:.2f} m")
    if tArgs.out:
        Path(tArgs.out).write_text(json.dumps(dict(T_B_C=tTbc.tolist(), T_C_B=tTcb.tolist(),
                                                   method="fixed tags 400/401 (start_anchors_v001) + table depth plane",
                                                   take=str(tTake), frames=tArgs.frames, reproj_px_p50=float(np.median(tRp)),
                                                   plane_mm_p50=float(np.median(tRz))), indent=1))
        print("wrote", tArgs.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
