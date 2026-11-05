"""Joint pivot over several tag_rig_record --proto pivot takes (2026-10-17): ONE jaw-tip position per UMI in its rig frame, one fixed desk
point per take (the dot may move between takes), progressively tightened inlier gate.  Per-frame rig poses from one PnP over every
visible tag corner (laundry_shared tag_rig_pose.rig_pose).  Also prints the per-take solution and the leave-one-take-out spread
(repeatability).  --write stores tcp_in_rig_m + pivot_joint into the --rig config.
    python -m laundry_tracking.calib.tag_rig_pivot_joint datasets/tag_rig/<take1> <take2> ... [--write]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from laundry_common import paths

WORD = {"UMI 오른손": "오른손", "UMI 왼손": "왼손"}
PIVOT_CUE = "집게 끝을 떼지"
CUE_START_TRIM_S = 1.0
CUE_END_TRIM_S = 0.5
MIN_TAG_SIDE_PX = 20
MAX_RMSE_PX = 2.0
MIN_TAKE_FRAMES = 30
GATES_M = (0.05, 0.025, 0.015, 0.010)
MIN_INLIERS = 40
DEFAULT_RIG = paths.CONFIGS / "calibration/umi_tag_rig_v002.json"


def pivotPoses(take: Path, name: str, rig: dict, size: float):
    """(R (n,3,3), t (n,3)) camera-frame rig poses of the pivot phase of one hand (was tag_rig_pivot_joint.poses)"""
    import cv2
    from laundry_shared.rebot_prep.parity.tag_rig_pose import rig_pose        # 10-19: one PnP over every visible corner (rotation jitter 1.78 -> 0.39 deg)
    tMeta = json.loads((take / "meta.json").read_text())
    tIntr = tMeta["intrinsics"]
    tK = np.array([[tIntr["fx"], 0, tIntr["cx"]], [0, tIntr["fy"], tIntr["cy"]], [0, 0, 1.0]])
    tT = (np.load(take / "t_ns.npy") - tMeta["t_start_monotonic_ns"]) / 1e9
    tProto = tMeta["protocol"]
    tPhase = next((tP for tI, tP in enumerate(tProto) if tP["text"].startswith(PIVOT_CUE) and WORD[name] in tProto[tI - 1]["text"]),
                  None)
    if tPhase is None:
        return np.zeros((0, 3, 3)), np.zeros((0, 3))
    tParams = cv2.aruco.DetectorParameters()
    tParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    tDet = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), tParams)
    tRigTags = {int(tId): np.array(tV) for tId, tV in rig[name]["T_rig_tag"].items()}
    tRs, tTs = [], []
    for tFrame in np.where((tT >= tPhase["t0"] + CUE_START_TRIM_S) & (tT < tPhase["t0"] + tPhase["dur"] - CUE_END_TRIM_S))[0]:
        tCorners, tIds, _ = tDet.detectMarkers(cv2.imread(str(take / f"color_{tFrame:05d}.jpg"), cv2.IMREAD_GRAYSCALE))
        tPose = rig_pose(tCorners, tIds, tRigTags, tK, size, min_side=MIN_TAG_SIDE_PX)
        if tPose is not None and tPose["rmse_px"] < MAX_RMSE_PX:
            tRs.append(tPose["T"][:3, :3])
            tTs.append(tPose["T"][:3, 3])
    return np.array(tRs), np.array(tTs)


def solve(blocks: list):
    """blocks: list of (Rs, ts).  Unknowns: p_rig (3) + one p_C per block (3 each) -> (p_rig, [p_C], rms mm, inliers per block)"""
    tKeeps = [np.ones(len(tB[0]), bool) for tB in blocks]
    tNb = len(blocks)
    for tThr in GATES_M:
        tRows, tRhs = [], []
        for tBi, ((tRs, tTs), tKeep) in enumerate(zip(blocks, tKeeps)):
            for tJ in np.where(tKeep)[0]:
                tArow = np.zeros((3, 3 + 3 * tNb))
                tArow[:, :3] = tRs[tJ]
                tArow[:, 3 + 3 * tBi: 6 + 3 * tBi] = -np.eye(3)
                tRows.append(tArow)
                tRhs.append(-tTs[tJ])
        tX = np.linalg.lstsq(np.concatenate(tRows), np.concatenate(tRhs), rcond=None)[0]
        tP = tX[:3]
        tRes = [np.linalg.norm(np.einsum("nij,j->ni", tRs, tP) + tTs - tX[3 + 3 * tBi: 6 + 3 * tBi],
                               axis=1) for tBi, (tRs, tTs) in enumerate(blocks)]
        tNew = [tR < tThr for tR in tRes]
        if sum(tK.sum() for tK in tNew) < MIN_INLIERS:
            break
        tKeeps = tNew
    tAll = np.concatenate([tR[tK] for tR, tK in zip(tRes, tKeeps)])
    tCentres = [tX[3 + 3 * tBi: 6 + 3 * tBi] for tBi in range(tNb)]
    return tP, tCentres, float(np.sqrt(np.mean(tAll ** 2)) * 1000), [int(tK.sum()) for tK in tKeeps]


def _buildParser() -> argparse.ArgumentParser:
    tAp = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tAp.add_argument("takes", nargs="+")
    tAp.add_argument("--rig", default=str(DEFAULT_RIG))
    tAp.add_argument("--tag-size", type=float, default=0.048)
    tAp.add_argument("--write", action="store_true")
    return tAp


def main(argv: list[str] | None = None) -> int:
    tArgs = _buildParser().parse_args(argv)
    tRig = json.loads(Path(tArgs.rig).read_text())
    tOut = {}
    for tName in WORD:
        tBlocks, tNames = [], []
        for tTakeArg in tArgs.takes:
            tRs, tTs = pivotPoses(Path(tTakeArg), tName, tRig, tArgs.tag_size)
            if len(tRs) >= MIN_TAKE_FRAMES:
                tBlocks.append((tRs, tTs))
                tNames.append(Path(tTakeArg).name)
            else:
                print(f"[{tName}] {Path(tTakeArg).name}: 프레임 부족 ({len(tRs)}) -- 제외")
        if not tBlocks:
            continue
        for (tRs, tTs), tNm in zip(tBlocks, tNames):
            tP1, _, tRms1, tN1 = solve([(tRs, tTs)])
            print(f"[{tName}] {tNm} 단독: 집게 끝 {np.round(tP1 * 100, 2)} cm | RMS {tRms1:.1f} mm | 사용 {tN1[0]}/{len(tRs)}")
        tP, _, tRms, tNs = solve(tBlocks)
        tLoo = []
        if len(tBlocks) > 2:
            tLoo = [np.linalg.norm(solve([tB for tJ, tB in enumerate(tBlocks) if tJ != tI])[0] - tP) * 1000 for tI in range(len(tBlocks))]
        print(f"[{tName}] 합동 ({len(tBlocks)}개 녹화): 집게 끝 {np.round(tP * 100, 2)} cm, |p| {np.linalg.norm(tP) * 100:.1f} cm | RMS {tRms:.1f} mm | "
              f"사용 {tNs} | 하나 빼고 다시 풀 때 변화 {np.round(tLoo, 1)} mm")
        tOut[tName] = dict(tcp_in_rig_m=tP.tolist(), residual_rms_mm=tRms, takes=tNames, frames=tNs,
                           leave_one_out_mm=[float(tX) for tX in tLoo])
    if tArgs.write and tOut:
        tCfg = json.loads(Path(tArgs.rig).read_text())
        for tName, tV in tOut.items():
            tCfg[tName]["tcp_in_rig_m"] = tV["tcp_in_rig_m"]
            tCfg[tName]["pivot_joint"] = tV
        Path(tArgs.rig).write_text(json.dumps(tCfg, indent=1, ensure_ascii=False))
        print("기록", tArgs.rig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
