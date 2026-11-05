"""TCP (jaw tip in the tag-rig frame) from CORNER TOUCHES on a board of known geometry (2026-10-20, GPT review: tag-only, before the IMU
smoother).  Each hold sample i: p_B_rig(i) + R_B_rig(i) t = corner_B(i).  One t per UMI by robust least squares (Huber, f_scale 5 mm).
Board frame from the Orbbec board pose (median over the take); hold names -> true corners by the board pose (best permutation).
Quality per hold (used as an automatic gate): mean visible UMI tags, multi-corner PnP RMSE, mean viewing angle, in-hold jitter of the
TCP point, single-tag fraction.  Evaluation: leave-one-corner-out (4 folds: solve on 3 corners, held-out corner error), fold-to-fold t
spread, and the full-data re-evaluation: same-corner spread across poses (main KPI), absolute bias, edges / diagonals.
Output: <EGO_DATA_ROOT>/datasets/tag_rig/touch_tcp_<take>.json; --write updates the --rig config.
    python -m laundry_tracking.calib.tag_rig_touch_tcp datasets/tag_rig/<corners take> [--write]
(was scripts/pose/tag_rig_touch_tcp.py of ego_collector; module path laundry_tracking.calib.tag_rig_touch_tcp unchanged by the 11-04 refactor)
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

from laundry_common import paths
from laundry_shared.rebot_prep.parity.tag_rig_corners import CORNERS, NAMES, board_pose
from laundry_shared.rebot_prep.parity.tag_rig_pose import rig_pose

DEFAULT_RIG = paths.CONFIGS / "calibration/umi_tag_rig_v002.json"
HOLD_CUE = "멈추세요"              # a hold cue reads '<hand>, <corner>, ... 멈추세요'
HOLD_WINDOW_S = (3.0, 0.1)         # samples from 3.0 s to 0.1 s before the end of a hold cue
MAX_RMSE_PX = 2.0
MAX_BOARD_FRAMES = 80
BOARD_EVERY = 3                    # board pose on every third frame
MIN_HOLD_SAMPLES = 5
SETTLED_TAIL = 0.6                 # 10-20: reference = median of the last 40 % of the hold window ...
SETTLED_RADIUS_M = 0.006           # ... and samples within 6 mm of it count as settled
BOARD_GAP = 0.3                    # board tag pitch = (1 + 0.3) x tag size
HUBER_F_SCALE = 0.005 / np.sqrt(30)
MS_PER_S = 1e3
HANDS = ("오른손", "왼손")
EDGE_PAIRS = [("왼쪽 아래", "오른쪽 아래"), ("오른쪽 아래", "오른쪽 위"), ("오른쪽 위", "왼쪽 위"), ("왼쪽 위", "왼쪽 아래"),
              ("왼쪽 아래", "오른쪽 위"), ("오른쪽 아래", "왼쪽 위")]


def _buildParser() -> argparse.ArgumentParser:
    tAp = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tAp.add_argument("take")
    tAp.add_argument("--rig", default=str(DEFAULT_RIG))
    tAp.add_argument("--tag-size", type=float, default=0.048)
    tAp.add_argument("--board-tag", type=float, default=0.05)
    tAp.add_argument("--board", default="3x4@300")
    tAp.add_argument("--gate-tags", type=float, default=1.5, help="min mean visible UMI tags per hold")
    tAp.add_argument("--gate-jitter-mm", type=float, default=15.0)
    tAp.add_argument("--write", action="store_true")
    tAp.add_argument("--fused", action="store_true",
                     help="10-20: rig poses from src/rebot_prep/parity/tag_rig_fuse.py (IMU orientation + tag centres)")
    tAp.add_argument("--depth", action="store_true", help="tag centre range from the aligned depth")
    tAp.add_argument("--rot", default="imu", choices=["imu", "tag"])
    return tAp


def main(argv: list[str] | None = None) -> int:
    tArgs = _buildParser().parse_args(argv)
    tTake = Path(tArgs.take)
    tMeta = json.loads((tTake / "meta.json").read_text())
    tCfg = json.loads(Path(tArgs.rig).read_text())
    tIntr = tMeta["intrinsics"]
    tK = np.array([[tIntr["fx"], 0, tIntr["cx"]], [0, tIntr["fy"], tIntr["cy"]], [0, 0, 1.0]])
    tT = (np.load(tTake / "t_ns.npy") - tMeta["t_start_monotonic_ns"]) / 1e9
    tDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tParams = cv2.aruco.DetectorParameters()
    tParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    tDet = cv2.aruco.ArucoDetector(tDict, tParams)
    tParams1 = cv2.aruco.DetectorParameters()
    tParams1.markerBorderBits = 1
    tParams1.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    tDet1 = cv2.aruco.ArucoDetector(tDict, tParams1)
    tBc, tRest = tArgs.board.split("x")
    tBr, tBid0 = tRest.split("@")
    tBc, tBr, tBid0 = int(tBc), int(tBr), int(tBid0)
    tW = (tBc + BOARD_GAP * (tBc - 1)) * tArgs.board_tag
    tHh = (tBr + BOARD_GAP * (tBr - 1)) * tArgs.board_tag
    tTruth = {"왼쪽 아래": np.array([0, 0, 0.0]), "오른쪽 아래": np.array([tW, 0, 0.0]), "오른쪽 위": np.array([tW, tHh, 0.0]),
              "왼쪽 위": np.array([0, tHh, 0.0])}
    tRig = {tH: {int(tKk): np.array(tV) for tKk, tV in tCfg[tN]["T_rig_tag"].items()} for tH, tN in NAMES.items()}
    tTcp0 = {tH: np.array(tCfg[tN]["tcp_in_rig_m"]) for tH, tN in NAMES.items()}
    tFused = {}
    if tArgs.fused:
        from laundry_shared.rebot_prep.parity.tag_rig_fuse import fuse_take
        for tHand in HANDS:
            tFr = fuse_take(tTake, tHand, tCfg, tArgs.tag_size, use_depth=tArgs.depth, rot_source=tArgs.rot)
            if tFr is not None:
                tFused[tHand] = {int(round((tTn / 1e9) * MS_PER_S)): (tTm, tN) for tTn, tTm, tN in zip(tFr["t_ns"],
                        tFr["T_C_rig"], tFr["ntags"])}
        print("[터치 TCP] IMU 융합 자세 사용: " + ", ".join(f"{tHand} {len(tV)}프레임" for tHand, tV in tFused.items()))
    tHolds, tBoards = [], []
    for tP in tMeta["protocol"]:
        if HOLD_CUE not in tP["text"] or "," not in tP["text"]:
            continue
        tHand, tCorner = [tX.strip() for tX in tP["text"].split(",")[:2]]
        tS = []
        for tFrame in np.where((tT >= tP["t0"] + tP["dur"] - HOLD_WINDOW_S[0]) & (tT < tP["t0"] + tP["dur"] - HOLD_WINDOW_S[1]))[0]:
            tGray = cv2.imread(str(tTake / f"color_{tFrame:05d}.jpg"), cv2.IMREAD_GRAYSCALE)
            tCs, tIds, _ = tDet.detectMarkers(tGray)
            if tArgs.fused:
                tKey = int(round((tT[tFrame] + tMeta["t_start_monotonic_ns"] / 1e9) * MS_PER_S))
                tFv = tFused.get(tHand, {}).get(tKey)
                if tFv is not None:
                    tS.append(dict(T=tFv[0], ntag=int(tFv[1]), rmse=0.0, view=0.0))
                if len(tBoards) < MAX_BOARD_FRAMES and tFrame % BOARD_EVERY == 0:
                    tB = board_pose(tGray, tK, tArgs.board_tag, tDet1, tBc, tBr, tBid0)
                    if tB is not None:
                        tBoards.append(tB)
                continue
            tPose = rig_pose(tCs, tIds, tRig[tHand], tK, tArgs.tag_size)
            if tPose is not None and tPose["rmse_px"] < MAX_RMSE_PX:
                tRay = tPose["T"][:3, 3] / np.linalg.norm(tPose["T"][:3, 3])
                tViews = [float(np.degrees(np.arccos(abs((tPose["T"] @ tRig[tHand][tI])[:3, 2] @ tRay)))) for tI in tPose["tags"]]
                tS.append(dict(T=tPose["T"], ntag=len(tPose["tags"]), rmse=tPose["rmse_px"], view=float(np.mean(tViews))))
            if len(tBoards) < MAX_BOARD_FRAMES and tFrame % BOARD_EVERY == 0:
                tB = board_pose(tGray, tK, tArgs.board_tag, tDet1, tBc, tBr, tBid0)
                if tB is not None:
                    tBoards.append(tB)
        tHolds.append(dict(hand=tHand, corner=tCorner, text=tP["text"], S=tS))
    tTb = np.eye(4)
    tTb[:3, :3] = R.from_matrix(np.array([tB[:3, :3] for tB in tBoards])).mean().as_matrix()
    tTb[:3, 3] = np.median([tB[:3, 3] for tB in tBoards], 0)
    tTbi = np.linalg.inv(tTb)
    print(f"[터치 TCP] {tTake.name}: 정지 {len(tHolds)}회, 보드 자세 {len(tBoards)}프레임 | 보드 {tW * 1000:.0f} x {tHh * 1000:.0f} mm")
    tReport = {}
    for tHand in HANDS:
        tHs = [tH for tH in tHolds if tH["hand"] == tHand and len(tH["S"]) >= MIN_HOLD_SAMPLES]
        if not tHs:
            continue
        for tH in tHs:                                              # board-frame rig poses + quality
            tH["RB"] = np.array([(tTbi @ tS["T"])[:3, :3] for tS in tH["S"]])
            tH["pB"] = np.array([(tTbi @ tS["T"])[:3, 3] for tS in tH["S"]])
            tTipB = tH["pB"] + np.einsum("nij,j->ni", tH["RB"], tTcp0[tHand])
            # 10-20: keep only the frames where the UMI had SETTLED (the operator often reaches the cross late -> the first part of the
            # window was still moving: one hold drifted 37 mm then sat within 1-2 mm). Reference = median of the last 40 % of the window.
            tRef = np.median(tTipB[int(SETTLED_TAIL * len(tTipB)):], 0)
            tKeep = np.linalg.norm(tTipB - tRef, axis=1) < SETTLED_RADIUS_M
            if tKeep.sum() >= MIN_HOLD_SAMPLES:
                tH["RB"], tH["pB"], tH["S"], tTipB = tH["RB"][tKeep], tH["pB"][tKeep], [tS for tS, tKk in zip(tH["S"],
                        tKeep) if tKk], tTipB[tKeep]
            tH["jit"] = float(np.linalg.norm(tTipB - np.median(tTipB, 0), axis=1).std() * 1000)
            tH["settled"] = int(tKeep.sum())
            tH["ntag"] = float(np.mean([tS["ntag"] for tS in tH["S"]]))
            tH["rmse"] = float(np.median([tS["rmse"] for tS in tH["S"]]))
            tH["view"] = float(np.median([tS["view"] for tS in tH["S"]]))
            tH["single"] = float(np.mean([tS["ntag"] == 1 for tS in tH["S"]]))
        # name -> true corner by the board pose, using the current t
        tMeans = {tC: np.mean([np.median(tH["pB"] + np.einsum("nij,j->ni", tH["RB"], tTcp0[tHand]), 0) for tH in tHs if tH["corner"] == tC], 0)
                  for tC in CORNERS if any(tH["corner"] == tC for tH in tHs)}
        tBest = min(itertools.permutations(CORNERS),
                    key=lambda perm: sum(np.linalg.norm(tMeans[c][:2] - tTruth[q][:2]) for c, q in zip(CORNERS,
                            perm) if c in tMeans))
        tCmap = dict(zip(CORNERS, tBest))
        print(f"== {tHand} UMI: 정지 {len(tHs)}회 | 모서리 대응 " + ", ".join(f"{tC}→{tQ}" for tC, tQ in tCmap.items()))
        for tH in tHs:
            tH["gate"] = tH["ntag"] >= tArgs.gate_tags and tH["jit"] <= tArgs.gate_jitter_mm
            print(f"   {tH['text'][:26]:26s} 태그 {tH['ntag']:.1f} (단일 {tH['single']:.0%}) rmse {tH['rmse']:.2f}px 각도 {tH['view']:.0f}° "
                  f"흔들림 {tH['jit']:5.1f}mm → {'사용' if tH['gate'] else '제외'}")
        tUsed = [tH for tH in tHs if tH["gate"]]
        if len({tH["corner"] for tH in tUsed}) < 2:
            print("   사용할 정지가 부족 -- 건너뜀")
            continue

        def _solve(sub, hand=tHand, cmap=tCmap):
            tRb = np.concatenate([tH["RB"] for tH in sub])
            tPb = np.concatenate([tH["pB"] for tH in sub])
            tTgt = np.concatenate([np.tile(tTruth[cmap[tH["corner"]]], (len(tH["S"]), 1)) for tH in sub])
            tWts = np.concatenate([np.full(len(tH["S"]),
                                           1 / np.sqrt(len(tH["S"]))) for tH in sub])          # each hold weighs the same
            return least_squares(lambda x: ((tPb + np.einsum("nij,j->ni", tRb, x) - tTgt) * tWts[:, None]).ravel(), tTcp0[hand],
                                 loss="huber", f_scale=HUBER_F_SCALE).x

        def _evaluate(x, sub, cmap=tCmap):
            tPts = {tH["text"]: np.median(tH["pB"] + np.einsum("nij,j->ni", tH["RB"], x), 0) for tH in sub}
            tPer = {tC: np.array([tPts[tH["text"]] for tH in sub if tH["corner"] == tC]) for tC in CORNERS}
            tPer = {tC: tV for tC, tV in tPer.items() if len(tV)}
            tSpread = {tC: (max(np.linalg.norm(tU - tV) for tU, tV in itertools.combinations(tPp, 2)) * 1000 if len(tPp) > 1 else 0.0)
                       for tC, tPp in tPer.items()}
            tErr = {tC: (tPp.mean(0) - tTruth[cmap[tC]]) * 1000 for tC, tPp in tPer.items()}
            return tSpread, tErr, {tC: tPp.mean(0) for tC, tPp in tPer.items()}
        tXAll = _solve(tUsed)
        tSp0, tEr0, _ = _evaluate(tTcp0[tHand], tUsed)
        tSp1, tEr1, tM1 = _evaluate(tXAll, tUsed)
        print(f"   TCP 기존 {np.round(tTcp0[tHand] * 100, 2)} cm → 터치 보정 {np.round(tXAll * 100, 2)} cm (변화 {np.linalg.norm(tXAll - tTcp0[tHand]) * 1000:.1f} mm)")
        print(f"   [전체 재평가] 같은 모서리 자세별 최대 차이(mm): 기존 " + ", ".join(f"{tSp0[tC]:.0f}" for tC in tSp0) + " → 보정 " + ", ".join(f"{tSp1[tC]:.0f}" for tC in tSp1))
        print(f"   [전체 재평가] 절대 오차 크기(mm): 기존 " + ", ".join(f"{np.linalg.norm(tEr0[tC]):.0f}" for tC in tEr0) + " → 보정 "
              + ", ".join(f"{np.linalg.norm(tEr1[tC]):.0f}" for tC in tEr1)
              + f" | 편향 기존 {np.round(np.mean(list(tEr0.values()), 0), 1)} → 보정 {np.round(np.mean(list(tEr1.values()), 0), 1)}")
        if len(tM1) == 4:
            print("   [전체 재평가] 변/대각선 오차: " + ", ".join(f"{(np.linalg.norm(tM1[tX] - tM1[tY]) / np.linalg.norm(tTruth[tCmap[tX]] - tTruth[tCmap[tY]]) - 1) * 100:+.1f}%"
                                                     for tX, tY in EDGE_PAIRS))
        tFolds = []
        for tC in CORNERS:                                          # leave-one-corner-out
            tTrain = [tH for tH in tUsed if tH["corner"] != tC]
            tTest = [tH for tH in tUsed if tH["corner"] == tC]
            if len(set(tH["corner"] for tH in tTrain)) < 2 or not tTest:
                continue
            tX = _solve(tTrain)
            tSpT, tErT, _ = _evaluate(tX, tTest)
            tFolds.append(tX)
            print(f"   [빼고 검증] {tC} 제외 → TCP {np.round(tX * 100, 2)} cm | 그 모서리 절대 오차 {np.linalg.norm(tErT[tC]):.1f} mm {np.round(tErT[tC], 1)} | "
                  f"자세별 최대 차이 {tSpT[tC]:.1f} mm")
        if tFolds:
            print(f"   fold 간 TCP 흩어짐: 최대 {max(np.linalg.norm(tF - tXAll) for tF in tFolds) * 1000:.1f} mm (전체 해 기준)")
        tReport[tHand] = dict(tcp_touch_m=tXAll.tolist(), tcp_prev_m=tTcp0[tHand].tolist(), used_holds=len(tUsed),
                              total_holds=len(tHs), loco_tcp_m=[tF.tolist() for tF in tFolds], spread_prev_mm=tSp0,
                              spread_new_mm=tSp1)
    tOut = paths.TAG_RIG / f"touch_tcp_{tTake.name}.json"         # was <ego_collector>/datasets/tag_rig (the module's repository root)
    tOut.write_text(json.dumps(tReport, indent=1, ensure_ascii=False))
    print("저장", tOut)
    if tArgs.write:
        tCfg = json.loads(Path(tArgs.rig).read_text())
        for tHand, tV in tReport.items():
            tN = NAMES[tHand]
            tCfg[tN]["tcp_pivot_m"] = tCfg[tN].get("tcp_pivot_m", tCfg[tN]["tcp_in_rig_m"])
            tCfg[tN]["tcp_in_rig_m"] = tV["tcp_touch_m"]
            tCfg[tN]["touch_tcp"] = dict(take=tTake.name, **{tKk: tV[tKk] for tKk in ("used_holds", "total_holds", "loco_tcp_m")})
        Path(tArgs.rig).write_text(json.dumps(tCfg, indent=1, ensure_ascii=False))
        print("기록", tArgs.rig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
