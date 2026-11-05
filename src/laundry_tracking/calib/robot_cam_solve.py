"""Robot base <-> global Orbbec solve (2026-10-20, GPT plan step 5).  Unknowns T_B_C (camera in the robot base) and T_E_tag (tag 220 on
the gripper, E = <side>_tcp of the URDF) from the robot_cam_poses holds + a tag_rig_record --proto robotcal take (same host monotonic
clock).  Chain per hold i: T_C_tag(i) = T_C_B T_B_E(i) T_E_tag.
  1. per hold: tag corners in every Orbbec frame inside the hold window (0.2 s trimmed), median over frames (>= --min-frames)
  2. init: flip-free tag-centre fit (+ cv2.calibrateRobotWorldHandEye as a second start)
  3. refine: reprojection of the 4 tag corners of every hold, Huber 2 px, + the Orbbec DEPTH at the tag (median in the quad,
     --depth-sigma-mm; 10-20 synthetic: at ~1 m a 48 mm tag's PnP depth is too noisy to start from) (optional --fit-size: tag size,
     the FK gives metric scale)
  4. held-out: --folds k-fold (default 5) -> predicted tag centre vs the PnP of the held-out holds (mm, deg) and reprojection px
    python -m laundry_tracking.calib.robot_cam_solve datasets/tag_rig/<take> datasets/robot_cam/<run> [--write]
    python -m laundry_tracking.calib.robot_cam_solve --synthetic                # self-test on simulated data (known answer)
--write -> configs/laundry/calibration/robot_cam_v001.json (T_B_C, T_E_tag, stats).
Absolute imports only: tests/laundry/test_layout_smoke.py also runs this file by path (--synthetic).
(was scripts/pose/robot_cam_solve.py of ego_collector; module path laundry_tracking.calib.robot_cam_solve unchanged by the 11-04 refactor)
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
from laundry_shared.rebot_prep.parity.tag_rig_pose import DepthStore, corners_obj, quad_depth_m

DEFAULT_OUT = paths.CONFIGS / "calibration/robot_cam_v001.json"
HOLD_TRIM_S = 0.2
MIN_DEPTH_SAMPLES = 3
INIT_ROUNDS = 30
INLIER_ROT_DEG = 15                # IPPE rotation candidates within 15 deg of the voted one are averaged
MIN_MAIN_HOLDS = 4
HUBER_F_SCALE = 2.0
PASS_REPROJ_PX = 3                 # verdict: held-out reprojection median < 3 px ...
PASS_LATERAL_MM = 10               # ... and lateral p95 < 10 mm
SYN_SEED = 1
SYN_HOLDS = 32
SYN_TRIES = 100
SYN_PX_NOISE = 0.4
SYN_FK_NOISE_M = 0.0007
SYN_DEPTH_NOISE_M = 0.004
SYN_SIZE_BIAS = 1.04
SYN_PASS_T_MM, SYN_PASS_R_DEG, SYN_PASS_TAG_MM = 10, 0.5, 5


def poseFromRotvec(rv, t) -> np.ndarray:
    """4x4 from a rotation vector and a translation (was robot_cam_solve.T_of)"""
    tT = np.eye(4)
    tT[:3, :3] = R.from_rotvec(rv).as_matrix()
    tT[:3, 3] = t
    return tT


def invertPose(T: np.ndarray) -> np.ndarray:
    """rigid 4x4 inverse (was robot_cam_solve.inv)"""
    tTi = np.eye(4)
    tTi[:3, :3] = T[:3, :3].T
    tTi[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return tTi


def project(T_C_tag: np.ndarray, K: np.ndarray, size: float) -> np.ndarray:
    """the 4 tag corners in pixels"""
    tP = (T_C_tag[:3, :3] @ corners_obj(size).T).T + T_C_tag[:3, 3]
    tUv = (K @ tP.T).T
    return tUv[:, :2] / tUv[:, 2:3]


def pnp(px: np.ndarray, K: np.ndarray, size: float):
    """IPPE tag pose or None"""
    tOk, tRv, tTv = cv2.solvePnP(corners_obj(size), px.astype(np.float64), K, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return poseFromRotvec(tRv.ravel(), tTv.ravel()) if tOk else None


def collectHolds(take: Path, run: Path, tags, minFrames: int, trimS: float = HOLD_TRIM_S, link: int = 6):
    """per hold and tag: median corners, jitter, FK pose, depth -> (observations, K, take meta) (was robot_cam_solve.collect)"""
    tMeta = json.loads((take / "meta.json").read_text())
    tIntr = tMeta["intrinsics"]
    tK = np.array([[tIntr["fx"], 0, tIntr["cx"]], [0, tIntr["fy"], tIntr["cy"]], [0, 0, 1.0]])
    tT = np.load(take / "t_ns.npy")
    tPoses = [json.loads(tL) for tL in (run / "poses.jsonl").read_text().splitlines() if tL.strip()]
    tParams = cv2.aruco.DetectorParameters()
    tParams.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    tDet = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), tParams)
    tObs = []
    tDepthStore = DepthStore(take)
    for tP in tPoses:
        tKs = np.where((tT >= tP["t_hold0_ns"] + trimS * 1e9) & (tT <= tP["t_hold1_ns"] - trimS * 1e9))[0]
        if not len(tKs):
            print(f"  포즈 {tP['i']:2d}: 녹화 프레임 없음 (녹화기와 시간이 안 겹침?)")
            continue
        tPer = {tG: [] for tG in tags}
        tDep = {tG: [] for tG in tags}
        for tFrame in tKs:
            tCorners, tIds, _ = tDet.detectMarkers(cv2.imread(str(take / f"color_{tFrame:05d}.jpg"), cv2.IMREAD_GRAYSCALE))
            tDm = None
            for tC, tId in zip(tCorners if tIds is not None else [], tIds.ravel() if tIds is not None else []):
                if int(tId) in tPer:
                    tPer[int(tId)].append(tC.reshape(4, 2))
                    tDm = tDepthStore.mm(tFrame) if tDm is None else tDm
                    tZ = quad_depth_m(tDm, tC)
                    if tZ is not None:
                        tDep[int(tId)].append(tZ)
        for tG, tL in tPer.items():
            if len(tL) < minFrames:
                continue
            tA = np.array(tL)
            tPx = np.median(tA, 0)
            tJit = float(np.sqrt(((tA - tPx) ** 2).sum(-1).mean()))
            tObs.append(dict(pose=tP["i"], tag=tG, px=tPx, jitter_px=tJit, n=len(tL),
                             T_B_E=np.array(tP["T_B_E"] if link == 6 else tP[f"T_B_E_link{link}"]),
                             depth_m=float(np.median(tDep[tG])) if len(tDep[tG]) >= MIN_DEPTH_SAMPLES else None,
                             side_px=float(np.mean(np.linalg.norm(tPx - np.roll(tPx, 1, 0), axis=1)))))
        if not any(tO["pose"] == tP["i"] for tO in tObs):
            print(f"  포즈 {tP['i']:2d}: 태그 {list(tags)} 안 보임 (프레임 {len(tKs)})")
    return tObs, tK, tMeta


def pnpBoth(px: np.ndarray, K: np.ndarray, size: float) -> list:
    """both IPPE solutions (small tags flip: the 20 px synthetic case gave 300-400 mm hand-eye init errors) (was robot_cam_solve.pnp2)"""
    _, tRvs, tTvs, _ = cv2.solvePnPGeneric(corners_obj(size), px.astype(np.float64), K, np.zeros(5),
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
    return [poseFromRotvec(tR.ravel(), tT.ravel()) for tR, tT in zip(tRvs, tTvs)]


def tagCentre(o: dict, K: np.ndarray, size: float) -> np.ndarray:
    """tag centre in the camera: pixel ray x Orbbec depth when there is depth, else the PnP translation (was robot_cam_solve.p3)"""
    if o.get("depth_m"):
        return np.linalg.inv(K) @ np.r_[o["px"].mean(0), 1.0] * o["depth_m"]
    return pnp(o["px"], K, size)[:3, 3]


def kabsch(P: np.ndarray, Q: np.ndarray):
    """R, t with Q ~ R P + t"""
    tMp, tMq = P.mean(0), Q.mean(0)
    tU, _, tVt = np.linalg.svd((P - tMp).T @ (Q - tMq))
    tD = np.diag([1, 1, np.sign(np.linalg.det(tVt.T @ tU.T))])
    tRm = tVt.T @ tD @ tU.T
    return tRm, tMq - tRm @ tMp


def initPositions(obsMain: list, K: np.ndarray, size: float):
    """flip-free init (10-20): tag CENTRES only (PnP translation is fine even when the rotation flips).
    p_C(i) = R_CB (p_BE(i) + R_BE(i) t_Et) + t_CB : alternate Kabsch (R_CB, t_CB) and linear LS (t_Et, t_CB); then R_Et by inlier voting
    over both IPPE solutions of every hold (was robot_cam_solve.init_positions)"""
    tPc = np.array([tagCentre(tO, K, size) for tO in obsMain])
    tTbe = np.array([tO["T_B_E"] for tO in obsMain])
    tTet = np.zeros(3)
    for _ in range(INIT_ROUNDS):
        tRcb, tTcb = kabsch(tTbe[:, :3, 3] + np.einsum("nij,j->ni", tTbe[:, :3, :3], tTet), tPc)
        tA = np.concatenate([np.concatenate([tRcb @ tT[:3, :3], np.eye(3)], 1) for tT in tTbe])
        tB = np.concatenate([tPc[tI] - tRcb @ tTbe[tI, :3, 3] for tI in range(len(tTbe))])
        tSol = np.linalg.lstsq(tA, tB, rcond=None)[0]
        tTet = tSol[:3]
    tRcb, tTcb = kabsch(tTbe[:, :3, 3] + np.einsum("nij,j->ni", tTbe[:, :3, :3], tTet), tPc)
    tCand = [tTbe[tI, :3, :3].T @ tRcb.T @ tT[:3, :3] for tI, tO in enumerate(obsMain) for tT in pnpBoth(tO["px"], K, size)]
    tBest = max(tCand, key=lambda Rc: sum(np.degrees(R.from_matrix(Rc.T @ Rq).magnitude()) < INLIER_ROT_DEG for Rq in tCand))
    tInl = [tRq for tRq in tCand if np.degrees(R.from_matrix(tBest.T @ tRq).magnitude()) < INLIER_ROT_DEG]
    T_C_B = np.eye(4)
    T_C_B[:3, :3] = tRcb
    T_C_B[:3, 3] = tTcb
    T_E_tag = np.eye(4)
    T_E_tag[:3, :3] = R.from_matrix(np.array(tInl)).mean().as_matrix()
    T_E_tag[:3, 3] = tTet
    return T_C_B, T_E_tag


class RobotCamError(ValueError):
    """Too few holds see the main tag to solve the camera."""


def solve(obs: list, K: np.ndarray, size: float, fitSize: bool = False, depthSigmaMm: float = 6.0) -> dict:
    tTags = sorted({tO["tag"] for tO in obs})
    tMain = max(tTags, key=lambda g: sum(tO["tag"] == g for tO in obs))
    tObsMain = [tO for tO in obs if tO["tag"] == tMain]
    if len(tObsMain) < MIN_MAIN_HOLDS:
        raise RobotCamError(f"태그 {tMain} 가 보인 포즈가 {len(tObsMain)}개 -- 최소 4개 (권장 25+)")
    tInits = [initPositions(tObsMain, K, size)]
    try:                                             # classic AX=ZB as a second start (fine with big tags, flips with small ones)
        tAR, tAt, tBR, tBt = [], [], [], []
        for tO in tObsMain:
            tTtc = invertPose(pnp(tO["px"], K, size))
            tTeb = invertPose(tO["T_B_E"])
            tAR.append(tTtc[:3, :3])
            tAt.append(tTtc[:3, 3])
            tBR.append(tTeb[:3, :3])
            tBt.append(tTeb[:3, 3])
        tRx, tTx, tRz, tTz = cv2.calibrateRobotWorldHandEye(tAR, tAt, tBR, tBt, method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH)
        tX = np.eye(4)
        tX[:3, :3] = tRx
        tX[:3, 3] = tTx.ravel()
        tZ = np.eye(4)
        tZ[:3, :3] = tRz
        tZ[:3, 3] = tTz.ravel()
        tInits.append((tX, invertPose(tZ)))
    except cv2.error:
        pass
    tBest = None
    for T_C_B0, tTm0 in tInits:
        T_E_tag0 = {tMain: tTm0}
        for tG in tTags:                             # extra tags: average of T_E_B T_B_C T_C_tag over their holds
            if tG == tMain:
                continue
            tTs = [invertPose(tO["T_B_E"]) @ invertPose(T_C_B0) @ pnp(tO["px"], K, size) for tO in obs if tO["tag"] == tG]
            tT = np.eye(4)
            tT[:3, :3] = R.from_matrix(np.array([tX[:3, :3] for tX in tTs])).mean().as_matrix()
            tT[:3, 3] = np.mean([tX[:3, 3] for tX in tTs], 0)
            T_E_tag0[tG] = tT
        tRes = refine(obs, K, size, fitSize, tTags, T_C_B0, T_E_tag0, depthSigmaMm)
        if tBest is None or tRes["cost"] < tBest["cost"]:
            tBest = tRes
    return tBest


def refine(obs: list, K: np.ndarray, size: float, fitSize: bool, tags: list, T_C_B0: np.ndarray, T_E_tag0: dict,
           depthSigmaMm: float = 6.0) -> dict:
    def _unpack(x):
        tTcb = poseFromRotvec(x[:3], x[3:6])
        tTe = {tG: poseFromRotvec(x[6 + 6 * tJ:9 + 6 * tJ], x[9 + 6 * tJ:12 + 6 * tJ]) for tJ, tG in enumerate(tags)}
        return tTcb, tTe, (size * np.exp(x[-1]) if fitSize else size)

    def _residual(x):
        tTcb, tTe, tSz = _unpack(x)
        tOut = []
        for tO in obs:
            tT = tTcb @ tO["T_B_E"] @ tTe[tO["tag"]]
            tOut.append((project(tT, K, tSz) - tO["px"]).ravel())
            if tO.get("depth_m") and depthSigmaMm > 0:
                tOut.append([(tT[2, 3] - tO["depth_m"]) * 1000 / depthSigmaMm])   # 1 sigma ~ 1 px
        return np.concatenate(tOut)
    tTagParams = [np.r_[R.from_matrix(T_E_tag0[tG][:3, :3]).as_rotvec(), T_E_tag0[tG][:3, 3]] for tG in tags]
    tX0 = np.concatenate([R.from_matrix(T_C_B0[:3, :3]).as_rotvec(), T_C_B0[:3, 3]] + tTagParams + ([[0.0]] if fitSize else []))
    tSol = least_squares(_residual, tX0, loss="huber", f_scale=HUBER_F_SCALE, x_scale="jac")
    T_C_B, tTe, tSz = _unpack(tSol.x)
    tPer = np.array([np.sqrt(((project(T_C_B @ tO["T_B_E"] @ tTe[tO["tag"]], K,
                                       tSz) - tO["px"]) ** 2).sum(-1).mean()) for tO in obs])
    tDz = np.array([(T_C_B @ tO["T_B_E"] @ tTe[tO["tag"]])[2, 3] - tO["depth_m"] for tO in obs if tO.get("depth_m")]) * 1000
    return dict(T_C_B=T_C_B, T_B_C=invertPose(T_C_B), T_E_tag=tTe, size=tSz, rms_px=tPer, depth_res_mm=tDz, init_T_C_B=T_C_B0,
                cost=float(tSol.cost))


def heldOut(obs: list, K: np.ndarray, size: float, folds: int, fitSize: bool, depthSigmaMm: float = 6.0) -> list:
    """k-fold over hold poses: predicted vs measured tag of the held-out holds (was robot_cam_solve.heldout)"""
    tPoses = sorted({tO["pose"] for tO in obs})
    tOut = []
    for tF in range(folds):
        tTest = set(tPoses[tF::folds])
        tTrain = [tO for tO in obs if tO["pose"] not in tTest]
        if len({tO["pose"] for tO in tTrain}) < 3:
            continue
        S = solve(tTrain, K, size, fitSize, depthSigmaMm)
        for tO in obs:
            if tO["pose"] not in tTest or tO["tag"] not in S["T_E_tag"]:
                continue
            tTp = S["T_C_B"] @ tO["T_B_E"] @ S["T_E_tag"][tO["tag"]]
            tPm = tagCentre(tO, K, S["size"])
            tTm = min(pnpBoth(tO["px"], K, S["size"]),
                      key=lambda T: R.from_matrix(tTp[:3, :3].T @ T[:3, :3]).magnitude())   # the IPPE twin closest
            tPx = float(np.sqrt(((project(tTp, K, S["size"]) - tO["px"]) ** 2).sum(-1).mean()))
            tDp = tTp[:3, 3] - tPm
            tRay = tPm / np.linalg.norm(tPm)
            tOut.append(dict(pose=tO["pose"], px=tPx, mm=float(np.linalg.norm(tDp) * 1e3),
                             lateral_mm=float(np.linalg.norm(tDp - tRay * (tDp @ tRay)) * 1e3),
                             deg=float(np.degrees(R.from_matrix(tTp[:3, :3].T @ tTm[:3, :3]).magnitude()))))
    return tOut


def report(S: dict, obs: list, H: list) -> None:
    tT = S["T_B_C"]
    tZCam = tT[:3, 2]
    print(f"\n카메라 위치 (로봇 베이스 기준) {np.round(tT[:3, 3], 3)} m, 거리 {np.linalg.norm(tT[:3, 3]):.2f} m, 시선 방향 {np.round(tZCam, 2)}")
    for tG, tTe in S["T_E_tag"].items():
        print(f"태그 {tG} (그리퍼 TCP 기준) 위치 {np.round(tTe[:3, 3] * 100, 1)} cm, 회전 rotvec {np.round(np.degrees(R.from_matrix(tTe[:3, :3]).as_rotvec()), 1)}°")
    print(f"태그 크기 {S['size'] * 1000:.1f} mm | 재투영 오차 중앙 {np.median(S['rms_px']):.2f}px, p95 {np.percentile(S['rms_px'], 95):.2f}px (관측 {len(obs)}개)" +
          (f" | depth 잔차 중앙 {np.median(np.abs(S['depth_res_mm'])):.1f} mm (depth 있는 관측 {len(S['depth_res_mm'])})" if len(S["depth_res_mm"]) else " | depth 없음"))
    tWorst = np.argsort(-S["rms_px"])[:3]
    print("  가장 나쁜 포즈:", ", ".join(f"{obs[tW]['pose']}({S['rms_px'][tW]:.1f}px, 태그 {obs[tW]['side_px']:.0f}px)" for tW in tWorst))
    if H:
        def _col(key):
            return np.array([tH[key] for tH in H])
        print(f"검증 (k-fold, 학습에 안 쓴 포즈 {len(H)}개): 재투영 중앙 {np.median(_col('px')):.2f}px p95 {np.percentile(_col('px'), 95):.2f}px | "
              f"태그 위치 오차 중앙 {np.median(_col('mm')):.1f} mm (시선 수직 {np.median(_col('lateral_mm')):.1f} mm, p95 {np.percentile(_col('lateral_mm'), 95):.1f}) | "
              f"회전 중앙 {np.median(_col('deg')):.2f}°")
        print("  (위치 오차 = 예측 vs 측정(depth 광선, 없으면 PnP); 시선 방향은 depth 잡음 포함이라 시선 수직 mm 과 px 이 판정 기준)")
        tOk = np.median(_col("px")) < PASS_REPROJ_PX and np.percentile(_col("lateral_mm"), 95) < PASS_LATERAL_MM
        print("판정:", "PASS" if tOk else "확인 필요", "(기준: 검증 재투영 중앙 < 3px, 시선 수직 p95 < 10 mm)")


def synthetic(fitSize: bool, noDepth: bool = False) -> bool:
    """self-test on simulated holds with a known answer -> PASS"""
    tRng = np.random.default_rng(SYN_SEED)
    tK = np.array([[690.0, 0, 640], [0, 690, 360], [0, 0, 1]])
    tSize = 0.048
    tEye, tTgt = np.array([0.45, 0.6, 0.75]), np.array([0.45, -0.3, 0.25])          # camera ~1.3 m away, looking at the workspace
    tZ = tTgt - tEye
    tZ /= np.linalg.norm(tZ)
    tX = np.cross(tZ, [0, 0, 1.0])
    tX /= np.linalg.norm(tX)
    tY = np.cross(tZ, tX)
    T_B_C = np.eye(4)
    T_B_C[:3, :3] = np.c_[tX, tY, tZ]
    T_B_C[:3, 3] = tEye
    T_E_tag = poseFromRotvec(np.radians([5, -80, 3]), [0.0, 0.03, -0.06])
    tObs = []
    for tI in range(SYN_HOLDS):
        T_B_E = np.eye(4)
        T_B_E[:3, 3] = tTgt + tRng.uniform(-1, 1, 3) * [0.08, 0.08, 0.06]
        for _ in range(SYN_TRIES):                      # rotation excitation, keep the tag facing the camera
            tRr = R.from_rotvec(tRng.normal(size=3) * np.radians(15)).as_matrix()
            T_B_E[:3, :3] = tRr @ T_B_C[:3, :3] @ R.from_euler("x", 180, degrees=True).as_matrix() @ invertPose(T_E_tag)[:3, :3]
            T_C_tag = invertPose(T_B_C) @ T_B_E @ T_E_tag
            if T_C_tag[2, 2] < -0.5:
                break
        tPx = project(T_C_tag, tK, tSize) + tRng.normal(0, SYN_PX_NOISE, (4, 2))
        T_B_E_meas = T_B_E.copy()
        T_B_E_meas[:3, 3] += tRng.normal(0, SYN_FK_NOISE_M, 3)                        # FK / joint noise ~1 mm
        tObs.append(dict(pose=tI, tag=220, px=tPx, jitter_px=SYN_PX_NOISE, n=20, T_B_E=T_B_E_meas,
                         depth_m=None if noDepth else T_C_tag[2, 3] + tRng.normal(0, SYN_DEPTH_NOISE_M),
                         side_px=float(np.mean(np.linalg.norm(tPx - np.roll(tPx, 1, 0), axis=1)))))
    S = solve(tObs, tK, tSize * (SYN_SIZE_BIAS if fitSize else 1.0), fitSize)
    H = heldOut(tObs, tK, tSize, 5, fitSize)
    report(S, tObs, H)
    tErrT = np.linalg.norm(S["T_B_C"][:3, 3] - T_B_C[:3, 3]) * 1e3
    tErrR = np.degrees(R.from_matrix(S["T_B_C"][:3, :3].T @ T_B_C[:3, :3]).magnitude())
    tErrG = np.linalg.norm(S["T_E_tag"][220][:3, 3] - T_E_tag[:3, 3]) * 1e3
    print(f"\n[합성 시험] 정답 대비 T_B_C {tErrT:.1f} mm / {tErrR:.2f}°, T_E_tag {tErrG:.1f} mm, 태그 크기 {S['size'] * 1000:.2f} mm (정답 48.00)")
    tOk = tErrT < SYN_PASS_T_MM and tErrR < SYN_PASS_R_DEG and tErrG < SYN_PASS_TAG_MM
    print("합성 시험", "PASS" if tOk else "FAIL")
    return tOk


def _check(obs: list, K: np.ndarray, args) -> None:
    """10-20 (fisheye mounted on the wrist): is the saved T_B_C still valid?  T_B_C FIXED, only T_E_tag refit (the tag may have moved)"""
    tCal = json.loads(Path(args.out).read_text())
    T_C_B = np.array(tCal["T_C_B"])
    tG0 = obs[0]["tag"]
    tTs = [invertPose(tO["T_B_E"]) @ invertPose(T_C_B) @ min(pnpBoth(tO["px"], K, args.tag_size),
                                                             key=lambda T: 0) for tO in obs if tO["tag"] == tG0]
    tT0 = np.eye(4)
    tT0[:3, :3] = R.from_matrix(np.array([tX[:3, :3] for tX in tTs])).mean().as_matrix()
    tT0[:3, 3] = np.median([tX[:3, 3] for tX in tTs], 0)

    def _residual(x):
        tTe = poseFromRotvec(x[:3], x[3:])
        tOut = []
        for tO in obs:
            tT = T_C_B @ tO["T_B_E"] @ tTe
            tOut.append((project(tT, K, args.tag_size) - tO["px"]).ravel())
            if tO.get("depth_m"):
                tOut.append([(tT[2, 3] - tO["depth_m"]) * 1000 / args.depth_sigma_mm])
        return np.concatenate(tOut)
    tSol = least_squares(_residual, np.r_[R.from_matrix(tT0[:3, :3]).as_rotvec(), tT0[:3, 3]], loss="huber",
                         f_scale=HUBER_F_SCALE)
    tTe = poseFromRotvec(tSol.x[:3], tSol.x[3:])
    tPx = [np.sqrt(((project(T_C_B @ tO["T_B_E"] @ tTe, K, args.tag_size) - tO["px"]) ** 2).sum(-1).mean()) for tO in obs]
    tLat = []
    for tO in obs:
        tTp = T_C_B @ tO["T_B_E"] @ tTe
        tPm = tagCentre(tO, K, args.tag_size)
        tDp = tTp[:3, 3] - tPm
        tRay = tPm / np.linalg.norm(tPm)
        tLat.append(np.linalg.norm(tDp - tRay * (tDp @ tRay)) * 1e3)
    tTeOld = np.array(tCal["T_E_tag"][str(tG0)])
    print(f"[점검] 저장된 T_B_C 고정, 태그 위치만 다시 맞춤: 재투영 중앙 {np.median(tPx):.2f}px (p95 {np.percentile(tPx, 95):.2f}), "
          f"시선 수직 오차 중앙 {np.median(tLat):.1f} mm (p95 {np.percentile(tLat, 95):.1f})")
    print(f"       태그 위치 변화 {np.linalg.norm(tTe[:3, 3] - tTeOld[:3, 3]) * 1000:.1f} mm / {np.degrees(R.from_matrix(tTe[:3, :3].T @ tTeOld[:3, :3]).magnitude()):.1f}°")
    print("판정:", "그대로 사용 가능" if np.median(tPx) < PASS_REPROJ_PX and np.percentile(tLat,
                                                                                  95) < PASS_LATERAL_MM else "다시 캘리브레이션 필요 (로봇 또는 Orbbec 이 움직였을 수 있음)")


def _buildParser() -> argparse.ArgumentParser:
    tAp = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tAp.add_argument("take", nargs="?")
    tAp.add_argument("run", nargs="?")
    tAp.add_argument("--tags", type=int, nargs="+", default=[220])
    tAp.add_argument("--tag-size", type=float, default=0.050)
    tAp.add_argument("--fit-size", action="store_true")
    tAp.add_argument("--min-frames", type=int, default=5)
    tAp.add_argument("--folds", type=int, default=5)
    tAp.add_argument("--synthetic", action="store_true")
    tAp.add_argument("--no-depth", action="store_true")
    tAp.add_argument("--depth-sigma-mm", type=float, default=6.0)
    tAp.add_argument("--write", action="store_true")
    tAp.add_argument("--check", action="store_true", help="저장된 캘리브레이션이 아직 맞는지 점검 (짧은 포즈 녹화로)")
    tAp.add_argument("--tag-link", type=int, default=6, choices=[4, 5, 6],
                     help="태그가 붙은 부품: 6=그리퍼(j6 뒤), 5=손목(j5~j6 사이), 4=손목(j4~j5 사이)")
    tAp.add_argument("--out", default=str(DEFAULT_OUT))
    return tAp


def main(argv: list[str] | None = None) -> int:
    tArgs = _buildParser().parse_args(argv)
    if tArgs.synthetic:
        return 0 if synthetic(tArgs.fit_size, tArgs.no_depth) else 1
    tTake, tRun = Path(tArgs.take), Path(tArgs.run)
    tObs, tK, tMeta = collectHolds(tTake, tRun, tArgs.tags, tArgs.min_frames, link=tArgs.tag_link)
    print(f"관측: 포즈 {len({tO['pose'] for tO in tObs})}개 (depth 있음 {sum(bool(tO['depth_m']) for tO in tObs)}), 태그별 {[(tG, sum(tO['tag'] == tG for tO in tObs)) for tG in tArgs.tags]}, "
          f"태그 크기 중앙 {np.median([tO['side_px'] for tO in tObs]):.0f}px, 정지 중 코너 흔들림 중앙 {np.median([tO['jitter_px'] for tO in tObs]):.2f}px")
    if tArgs.no_depth:
        for tO in tObs:
            tO["depth_m"] = None
    if tArgs.check:
        _check(tObs, tK, tArgs)
        return 0
    try:
        S = solve(tObs, tK, tArgs.tag_size, tArgs.fit_size, tArgs.depth_sigma_mm)
        H = heldOut(tObs, tK, tArgs.tag_size, tArgs.folds, tArgs.fit_size, tArgs.depth_sigma_mm)
    except RobotCamError as tErr:
        print(tErr, file=sys.stderr)
        return 1
    report(S, tObs, H)
    if tArgs.write:
        tRunMeta = json.loads((tRun / "plan.json").read_text())
        Path(tArgs.out).write_text(json.dumps(dict(T_B_C=S["T_B_C"].tolist(), T_C_B=S["T_C_B"].tolist(), T_E_tag={str(tG): tT.tolist() for tG, tT in S["T_E_tag"].items()},
                                                   ee_frame=f"{tRunMeta['side']}_tcp" + ("" if tArgs.tag_link == 6 else f" with joints > {tArgs.tag_link} zeroed"),
                                                   tag_link=tArgs.tag_link, side=tRunMeta["side"], tag_size_m=S["size"], intrinsics=tMeta["intrinsics"],
                                                   rms_px_median=float(np.median(S["rms_px"])), heldout=H, take=str(tTake), run=str(tRun)),
                                              indent=1, ensure_ascii=False))
        print("기록", tArgs.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
