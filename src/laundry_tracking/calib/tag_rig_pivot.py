"""UMI jaw tip (TCP) in the tag-rig frame by PIVOT calibration (2026-10-17, no stereo cams: user lead decision).

Take = tag_rig_record --proto pivot: the jaw tip is held on one desk point while the UMI is tilted in every direction, right UMI then
left UMI.  Rig poses come from the tags (rig = T_rig_tag of a --rig take solved by tag_rig_solve); the tip p_rig and the fixed point
p_C (Orbbec frame) solve   R_i p_rig + t_i = p_C   for all frames i (linear least squares, progressively tightened inlier gate).
Quality: residual RMS / p90 (mm) and the tilt spread (deg; < 20 deg spread = the tip is poorly observable along the axis).
Output: <take>/tag_rig_pivot.json; --write also updates configs/laundry/calibration/umi_tag_rig_v001.json (repository config).
    python -m laundry_tracking.calib.tag_rig_pivot datasets/tag_rig/<pivot take> --rig datasets/tag_rig/<rig take>/tag_rig.json [--write]
(was scripts/pose/tag_rig_pivot.py of ego_collector; module path laundry_tracking.calib.tag_rig_pivot unchanged by the 11-04 refactor)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from laundry_common import paths

PHASES = {"UMI 오른손": "오른손", "UMI 왼손": "왼손"}
PIVOT_CUE = "집게 끝을 떼지"            # protocol cue of the pivot phase (the cue before it names the hand)
CUE_START_TRIM_S = 1.0
CUE_END_TRIM_S = 0.5
MIN_TAG_SIDE_PX = 20
MIN_FRAMES = 20
GATES_M = (0.05, 0.025, 0.015, 0.010)   # 10-17: progressively tighter gate (3-sigma kept all 256 frames, RMS 10 mm)
MIN_INLIERS = 30
SPREAD_PERCENTILE = 90
CONFIG_OUT = paths.CONFIGS / "calibration/umi_tag_rig_v001.json"


def _buildParser() -> argparse.ArgumentParser:
    tAp = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tAp.add_argument("take")
    tAp.add_argument("--rig", required=True)
    tAp.add_argument("--tag-size", type=float, default=0.048)
    tAp.add_argument("--write", action="store_true")
    return tAp


def main(argv: list[str] | None = None) -> int:
    import cv2
    tArgs = _buildParser().parse_args(argv)
    tTake = Path(tArgs.take)
    tMeta = json.loads((tTake / "meta.json").read_text())
    tRigs = json.loads(Path(tArgs.rig).read_text())["rigs"]
    tIntr = tMeta["intrinsics"]
    tK = np.array([[tIntr["fx"], 0, tIntr["cx"]], [0, tIntr["fy"], tIntr["cy"]], [0, 0, 1.0]])
    tT = (np.load(tTake / "t_ns.npy") - tMeta["t_start_monotonic_ns"]) / 1e9
    tHalf = tArgs.tag_size / 2
    tObj = np.array([[-tHalf, tHalf, 0], [tHalf, tHalf, 0], [tHalf, -tHalf, 0], [-tHalf, -tHalf, 0]], np.float64)
    tParams = cv2.aruco.DetectorParameters()
    tParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    tDet = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), tParams)
    tProto = tMeta["protocol"]
    tOut = {}
    for tName, tWord in PHASES.items():
        if tName not in tRigs:
            print(f"[{tName}] 리그 정보 없음 -- 먼저 --proto rig 녹화 + tag_rig_solve")
            continue
        tRigTags = {int(tId): np.array(tV) for tId, tV in tRigs[tName]["T_rig_tag"].items()}
        tPhase = next((tP for tI, tP in enumerate(tProto) if tP["text"].startswith(PIVOT_CUE) and tWord in tProto[tI - 1]["text"]),
                      None)
        if tPhase is None:
            print(f"[{tName}] 피벗 구간 없음")
            continue
        tRs, tTs = [], []
        for tFrame in np.where((tT >= tPhase["t0"] + CUE_START_TRIM_S) & (tT < tPhase["t0"] + tPhase["dur"] - CUE_END_TRIM_S))[0]:
            tImg = cv2.imread(str(tTake / f"color_{tFrame:05d}.jpg"), cv2.IMREAD_GRAYSCALE)
            tCorners, tIds, _ = tDet.detectMarkers(tImg)
            tEst = []
            for tC, tId in zip(tCorners if tIds is not None else [], tIds.ravel() if tIds is not None else []):
                tId = int(tId)
                if tId not in tRigTags:
                    continue
                tPx = tC.reshape(4, 2).astype(np.float64)
                if np.mean(np.linalg.norm(tPx - np.roll(tPx, 1, 0), axis=1)) < MIN_TAG_SIDE_PX:
                    continue
                tOk, tRv, tTv = cv2.solvePnP(tObj, tPx, tK, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not tOk:
                    continue
                tTm = np.eye(4)
                tTm[:3, :3] = cv2.Rodrigues(tRv)[0]
                tTm[:3, 3] = tTv.ravel()
                tEst.append(tTm @ np.linalg.inv(tRigTags[tId]))                 # T_C_rig
            if tEst:
                tQ = R.from_matrix(np.array([tE[:3, :3] for tE in tEst])).mean()
                tRs.append(tQ.as_matrix())
                tTs.append(np.median([tE[:3, 3] for tE in tEst], 0))
        if len(tRs) < MIN_FRAMES:
            print(f"[{tName}] 피벗 프레임 부족 ({len(tRs)})")
            continue
        tRs, tTs = np.array(tRs), np.array(tTs)
        tKeep = np.ones(len(tRs), bool)
        for tThr in GATES_M:
            tA = np.concatenate([np.c_[tRs[tKeep][tJ], -np.eye(3)] for tJ in range(tKeep.sum())])
            tB = -tTs[tKeep].reshape(-1)
            tX = np.linalg.lstsq(tA, tB, rcond=None)[0]
            tPRig, tPC = tX[:3], tX[3:]
            tRes = np.linalg.norm(np.einsum("nij,j->ni", tRs, tPRig) + tTs - tPC, axis=1)
            if (tRes < tThr).sum() < MIN_INLIERS:
                break
            tKeep = tRes < tThr
        tRk = R.from_matrix(tRs[tKeep])
        tSpread = float(2 * np.degrees(np.percentile((tRk * tRk.mean().inv()).magnitude(),
                                                     SPREAD_PERCENTILE)))   # ~ tilt range around the mean pose
        tRms, tP90 = float(np.sqrt(np.mean(tRes[tKeep] ** 2)) * 1000), float(np.percentile(tRes[tKeep], 90) * 1000)
        print(f"[{tName}] 피벗 프레임 {tKeep.sum()}/{len(tRs)} | 집게 끝 (리그 좌표, cm) {np.round(tPRig * 100, 2)} |p| {np.linalg.norm(tPRig) * 100:.1f} cm | "
              f"잔차 RMS {tRms:.1f} mm, p90 {tP90:.1f} mm | 기울기 범위 {tSpread:.0f}°")
        tOut[tName] = dict(p_rig_m=tPRig.tolist(), p_cam_m=tPC.tolist(), frames=int(tKeep.sum()), residual_rms_mm=tRms,
                           residual_p90_mm=tP90, tilt_spread_deg=tSpread, rig_ref=tRigs[tName]["ref"])
    (tTake / "tag_rig_pivot.json").write_text(json.dumps(tOut, indent=1, ensure_ascii=False))
    print(f"저장 {tTake / 'tag_rig_pivot.json'}")
    if tArgs.write and tOut:
        tPrev = json.loads(CONFIG_OUT.read_text()) if CONFIG_OUT.exists() else {}
        for tName, tV in tOut.items():
            tPrev[tName] = dict(T_rig_tag=tRigs[tName]["T_rig_tag"], ref=tRigs[tName]["ref"], tcp_in_rig_m=tV["p_rig_m"],
                                pivot=tV, rig_take=str(tArgs.rig), pivot_take=str(tTake))
        CONFIG_OUT.write_text(json.dumps(tPrev, indent=1, ensure_ascii=False))
        print(f"기록 {CONFIG_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
