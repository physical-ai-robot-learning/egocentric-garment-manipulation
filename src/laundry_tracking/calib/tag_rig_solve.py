"""Solve the UMI AprilTag rig from a tag_rig_record take (2026-10-16): per-tag PnP + RGB-D quality, tag-to-tag rigid transforms, and the
still-window agreement that gates the global anchor (user plan: >= 2 tags agree within 2-3 cm / 5-10 deg -> correction allowed).
Per detection: IPPE_SQUARE PnP (tag size = black square, default 0.048 m; Orbbec colour treated as pinhole with the SDK fx fy cx cy),
reprojection error, pixel side, viewing angle (tag normal vs camera ray), depth check = median aligned depth at the 4 corners vs the PnP
corner z (mm), plane residual of the depth points inside the tag.
Rig frame = the first tag of the group (or the most seen tag).  T_rig_tag_i = robust mean over good co-visible frames (chordal rotation
mean, median translation), chained through the best-supported pairs (maximum spanning tree).
    python -m laundry_tracking.calib.tag_rig_solve datasets/tag_rig/<ts> [--tag-size 0.048]  -> <take>/tag_rig.json + printed report
(was scripts/pose/tag_rig_solve.py of ego_collector; module path laundry_tracking.calib.tag_rig_solve unchanged by the 11-04 refactor)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from laundry_shared.rebot_prep.parity.tag_rig_pose import TAG_SIZE_M, corners_obj, size_of

# 10-23: 30 mm camera-back tags joined the UMI groups (220 = unit with 210-212, 221 = unit with 200-202); robot-arm tags were never mounted
GROUPS = {"UMI 오른손": (200, 201, 202, 221), "UMI 왼손": (210, 211, 212, 220), "로봇 오른팔": (222,), "로봇 왼팔": (230, 231, 232)}
GOOD = dict(reproj_px=1.5, side_px=25, view_deg=65)    # a detection used for the rig: reprojection, pixel side, viewing angle
MM_DEPTH_UNITS = 100               # a depth frame whose max exceeds this is in mm -> metres
MIN_DEPTH_M = 0.05
CORNER_WIN_PX = 2                  # 5x5 window around every corner for the depth check
MIN_PLANE_POINTS = 30
MIN_PAIR_FRAMES = 5                # a tag pair needs this many co-visible good frames to join the rig
STILL_CUE = "그대로"               # protocol cue of the still window
STILL_TRIM_S = 1.0


def poseFromPnp(rvec, tvec) -> np.ndarray:
    """4x4 from an OpenCV rvec / tvec (was tag_rig_solve.T_from)"""
    tT = np.eye(4)
    tT[:3, :3] = cv2.Rodrigues(rvec)[0]
    tT[:3, 3] = np.ravel(tvec)
    return tT


def meanPose(poses) -> np.ndarray:
    """chordal rotation mean + median translation (was tag_rig_solve.mean_T)"""
    tQ = R.from_matrix(np.array([tT[:3, :3] for tT in poses])).mean()
    tT = np.eye(4)
    tT[:3, :3] = tQ.as_matrix()
    tT[:3, 3] = np.median([tP[:3, 3] for tP in poses], 0)
    return tT


def poseDifference(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    """(translation m, rotation deg) of inv(A) B (was tag_rig_solve.dT)"""
    tE = np.linalg.inv(A) @ B
    return float(np.linalg.norm(tE[:3, 3])), float(np.degrees(R.from_matrix(tE[:3, :3]).magnitude()))


def _good(d: dict) -> bool:
    return d["reproj"] < GOOD["reproj_px"] and d["side"] > GOOD["side_px"] and d["view"] < GOOD["view_deg"]


def _detections(take: Path, meta: dict, tagSize: float) -> list[dict]:
    """per frame: {tag id: dict(T, quality ...)}"""
    tIntr = meta["intrinsics"]
    tDs = float(meta["depth_scale"] or 1.0)
    tK = np.array([[tIntr["fx"], 0, tIntr["cx"]], [0, tIntr["fy"], tIntr["cy"]], [0, 0, 1.0]])
    tDist = np.zeros(5)
    tDepth = np.load(take / "depth.npz")["depth"]
    tT = np.load(take / "t_ns.npy")
    tParams = cv2.aruco.DetectorParameters()
    tParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    tDet = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), tParams)
    tFrames = []
    for tFrame in range(len(tT)):
        tImg = cv2.imread(str(take / f"color_{tFrame:05d}.jpg"), cv2.IMREAD_GRAYSCALE)
        tDep = tDepth[tFrame].astype(np.float64) * tDs
        if tDep.max() > MM_DEPTH_UNITS:
            tDep = tDep / 1000.0                                          # depth_scale in mm units -> metres
        tCorners, tIds, _ = tDet.detectMarkers(tImg)
        tFr = {}
        for tCc, tId in zip(tCorners if tIds is not None else [], tIds.ravel() if tIds is not None else []):
            tId = int(tId)
            if tId not in meta["tag_ids"]:
                continue
            tObj = corners_obj(size_of(tId, tagSize))
            tPx = tCc.reshape(4, 2).astype(np.float64)
            tOk, tRv, tTv = cv2.solvePnP(tObj, tPx, tK, tDist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not tOk:
                continue
            tTm = poseFromPnp(tRv, tTv)
            tReproj = np.linalg.norm(cv2.projectPoints(tObj, tRv, tTv, tK, tDist)[0].reshape(4, 2) - tPx, axis=1).mean()
            tSide = float(np.mean(np.linalg.norm(tPx - np.roll(tPx, 1, 0), axis=1)))
            tNormal = tTm[:3, 2]
            tRay = tTm[:3, 3] / np.linalg.norm(tTm[:3, 3])
            tView = float(np.degrees(np.arccos(min(1.0, abs(tNormal @ tRay)))))
            tZc = (tTm[:3, :3] @ tObj.T + tTm[:3, 3:4])[2]
            tDz = []
            for (tU, tV), tZp in zip(tPx, tZc):
                tU0, tV0 = int(round(tU)), int(round(tV))
                tWin = tDep[max(0, tV0 - CORNER_WIN_PX):tV0 + CORNER_WIN_PX + 1, max(0, tU0 - CORNER_WIN_PX):tU0 + CORNER_WIN_PX + 1]
                tWin = tWin[tWin > MIN_DEPTH_M]
                if tWin.size:
                    tDz.append((np.median(tWin) - tZp) * 1000)
            tMask = np.zeros(tImg.shape, np.uint8)
            cv2.fillConvexPoly(tMask, tPx.astype(np.int32), 1)
            tVv, tUu = np.nonzero(tMask)
            tZz = tDep[tVv, tUu]
            tGoodZ = tZz > MIN_DEPTH_M
            tPlaneRes = None
            if tGoodZ.sum() > MIN_PLANE_POINTS:
                tX = np.c_[(tUu[tGoodZ] - tK[0, 2]) * tZz[tGoodZ] / tK[0, 0], (tVv[tGoodZ] - tK[1, 2]) * tZz[tGoodZ] / tK[1, 1], tZz[tGoodZ]]
                tCen = tX.mean(0)
                _, _, tVt = np.linalg.svd(tX - tCen, full_matrices=False)
                tPlaneRes = float(np.sqrt(np.mean(((tX - tCen) @ tVt[2]) ** 2)) * 1000)
            tFr[tId] = dict(T=tTm, reproj=float(tReproj), side=tSide, view=tView,
                            depth_dz_mm=float(np.median(tDz)) if tDz else None, n_depth=len(tDz), plane_res_mm=tPlaneRes)
        tFrames.append(tFr)
    return tFrames


def _buildParser() -> argparse.ArgumentParser:
    tAp = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tAp.add_argument("take")
    tAp.add_argument("--tag-size", type=float, default=0.048)
    return tAp


def main(argv: list[str] | None = None) -> int:
    tArgs = _buildParser().parse_args(argv)
    tTake = Path(tArgs.take)
    tMeta = json.loads((tTake / "meta.json").read_text())
    tT = np.load(tTake / "t_ns.npy")
    tTs = (tT - tMeta["t_start_monotonic_ns"]) / 1e9
    F = _detections(tTake, tMeta, tArgs.tag_size)
    tAllQ = [tD for tF in F for tD in tF.values()]
    if tAllQ:
        print(f"[태그 리그] 프레임 {len(F)} | 품질 p50: 재투영 {np.median([tD['reproj'] for tD in tAllQ]):.2f}px, 한 변 {np.median([tD['side'] for tD in tAllQ]):.0f}px, "
              f"보는 각도 {np.median([tD['view'] for tD in tAllQ]):.0f}°, "
              f"depth−PnP z {np.nanmedian([tD['depth_dz_mm'] if tD['depth_dz_mm'] is not None else np.nan for tD in tAllQ]):+.1f} mm, "
              f"평면 잔차 {np.nanmedian([tD['plane_res_mm'] if tD['plane_res_mm'] is not None else np.nan for tD in tAllQ]):.1f} mm")
    tReport = dict(take=tTake.name, tag_size_m=tArgs.tag_size,
                   tag_size_override_m={str(tK): tV for tK, tV in TAG_SIZE_M.items()}, rigs={})
    tStill = next((tP for tP in tMeta["protocol"] if tP["text"].startswith(STILL_CUE)), None)
    tWin = [tK for tK in range(len(F)) if tStill and tStill["t0"] + STILL_TRIM_S <= tTs[tK] < tStill["t0"] + tStill["dur"] - STILL_TRIM_S]
    for tName, tGroup in GROUPS.items():                                  # 10-17: one rig per UMI / robot gripper
        tCount = {tI: sum(tI in tF for tF in F) for tI in tGroup}
        if sum(tCount.values()) == 0:
            continue
        tRef = tGroup[0] if tCount[tGroup[0]] else max(tCount, key=tCount.get)
        tPair = {}
        for tF in F:
            tG = [tI for tI in tF if tI in tGroup and _good(tF[tI])]
            for tI in tG:
                for tJ in tG:
                    if tI != tJ:
                        tPair.setdefault((tI, tJ), []).append(np.linalg.inv(tF[tI]["T"]) @ tF[tJ]["T"])      # T_tagi_tagj
        # 10-17: connect tags through the BEST-supported pairs (maximum spanning tree on co-visible frame counts). The first version
        # took any pair with >= 5 frames in dict order: right UMI tag 201 was attached via 200-201 (9 frames) instead of 202-201 (57)
        # and the pivot frames (mostly 201+202) scattered 15 cm.
        tRigT = {tRef: np.eye(4)}
        while True:
            tCand = [(len(tLp), tI,
                      tJ) for (tI, tJ), tLp in tPair.items() if tI in tRigT and tJ not in tRigT and len(tLp) >= MIN_PAIR_FRAMES]
            if not tCand:
                break
            _, tI, tJ = max(tCand)
            tRigT[tJ] = tRigT[tI] @ meanPose(tPair[(tI, tJ)])
        print(f"[{tName}] 검출 {tCount} | 기준 {tRef} | 함께 본 좋은 쌍 " + str({f'{tI}-{tJ}': len(tLp) for (tI, tJ), tLp in tPair.items() if tI < tJ})
              + f" | 리그에 들어간 태그 {sorted(tRigT)}")
        for tI, tTm in tRigT.items():
            if tI != tRef and tPair.get((tRef, tI)):
                tRes = np.median([poseDifference(tTm, tM) for tM in tPair[(tRef, tI)]], 0)
                print(f"   T_{tRef}_{tI}: 이동 {np.round(tTm[:3, 3] * 100, 1)} cm, 회전 {np.degrees(R.from_matrix(tTm[:3, :3]).magnitude()):.1f}°, "
                      f"쌍 잔차 p50 {tRes[0] * 100:.2f} cm / {tRes[1]:.1f}°")
        tRr = dict(ref=tRef, counts=tCount, T_rig_tag={str(tI): tTm.tolist() for tI, tTm in tRigT.items()})
        tDis, tPoses = [], []
        for tK in tWin:
            tEst = {tI: F[tK][tI]["T"] @ np.linalg.inv(tRigT[tI]) for tI in F[tK] if tI in tRigT and _good(F[tK][tI])}
            tIds = sorted(tEst)
            for tX in range(len(tIds)):
                for tY in range(tX + 1, len(tIds)):
                    tDis.append(poseDifference(tEst[tIds[tX]], tEst[tIds[tY]]))
            if tEst:
                tPoses.append(meanPose(list(tEst.values())))
        if tDis:
            tD = np.array(tDis)
            tRr.update(still_disagree_cm_p50=float(np.median(tD[:, 0]) * 100),
                       still_disagree_cm_p90=float(np.percentile(tD[:, 0], 90) * 100),
                       still_disagree_deg_p50=float(np.median(tD[:, 1])),
                       still_disagree_deg_p90=float(np.percentile(tD[:, 1], 90)))
            print(f"   정지 구간 태그 간 불일치: 위치 p50 {tRr['still_disagree_cm_p50']:.2f} / p90 {tRr['still_disagree_cm_p90']:.2f} cm, "
                  f"회전 p50 {tRr['still_disagree_deg_p50']:.1f} / p90 {tRr['still_disagree_deg_p90']:.1f}°")
        if tPoses:
            tP = np.array([tTm[:3, 3] for tTm in tPoses])
            tRr.update(still_pos_std_mm=float(np.linalg.norm(tP - tP.mean(0), axis=1).std() * 1000), still_frames=len(tPoses))
            print(f"   정지 구간 리그 위치 흔들림(std) {tRr['still_pos_std_mm']:.1f} mm ({len(tPoses)} 프레임, 2개 이상 "
                  f"{sum(1 for tK in tWin if sum(_good(F[tK][tI]) for tI in F[tK] if tI in tGroup) >= 2)}/{len(tWin)})")
        tReport["rigs"][tName] = tRr
    (tTake / "tag_rig.json").write_text(json.dumps(tReport, indent=1, ensure_ascii=False))
    print(f"   저장 {tTake / 'tag_rig.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
