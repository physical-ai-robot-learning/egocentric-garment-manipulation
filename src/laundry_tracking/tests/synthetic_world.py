"""Synthetic recording world for the laundry_tracking characterization tests (test helper, no test cases).

One take of two UMI units seen by a fixed Orbbec camera above a table: rendered colour / depth frames (two dark spheres = the rig
bodies), AprilTag-style tag trajectories (noisy truth with dropouts), raw tag rotations, wrist IMUs (gyro + specific force, with a
bias, host / device clocks), wrist-camera SLAM segments (MASt3R raw layout and the generic visual npz), jaw states, idle-trim rows,
derived assignment / production trajectories, calibration files and the per-take start seeds.  Everything is deterministic.

`writeWorld(root)` lays the files out under root/{ego,train,ckpt,desktop} with the production layout; `worldEnv(root)` returns the
EGO_* environment that points laundry_common.paths at it.  `syntheticArm(arm)` builds the in-memory per-arm dict that
fgo_data.load_arm would return (for the in-process tests)."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

TAKE = "20261030_120000"
HANDEYE_TAKE = "20261022_121938"          # one 10-22 fused take: the source of the mean hand-eye of handeye_1022
T_START_NS = 1_000_000_000_000
FPS = 15.0
DURATION_S = 30.0
DEMOS = ((2.0, 12.0), (16.0, 12.0))          # (t0, dur) of the two '자유:' protocol cues
IMU_HZ = 200.0
SLAM_HZ = 30.0
SLAM_DT_NS = 20_000_000
WIDTH, HEIGHT = 848, 480                 # the QA renderer assumes the 848 x 480 Orbbec colour stream
FX = FY = 560.0
CX, CY = 424.0, 240.0
CAM_POS = np.array([0.40, 0.0, 0.80])
BODY_R_M = 0.045
GRAVITY = 9.81
ARMS = ("left", "right")
RIG_OF_ARM = {"left": "right", "right": "left"}          # swapped on purpose: rig labels must never be confused with arm labels
CAM_OF_ARM = {"left": "wrist_left", "right": "wrist"}
START = {"left": np.array([0.40, 0.17, 0.09]), "right": np.array([0.40, -0.17, 0.09])}
BASE_ROTVEC = {"left": np.array([0.3, -0.2, 0.4]), "right": np.array([-0.25, 0.3, -0.5])}
GYRO_BIAS = {"right": np.array([0.012, -0.008, 0.005]), "left": np.array([-0.006, 0.010, 0.004])}   # by rig
OFFSETS_MM = {"right": [-79.9, -53.9, 58.6], "left": [83.5, -65.2, 59.2]}                          # body_offset_by_rig_v1
RX = R.from_euler("xyz", [10.0, -80.0, 5.0], degrees=True)
TX = np.array([0.0, 0.05, -0.08])
# tag dropouts per arm: (t0_s, t1_s, ntags) relative to the take start
TAG_GAPS = {"left": ((4.5, 5.5, 0), (14.0, 14.6, 1), (17.2, 17.5, 0)), "right": ((6.0, 6.4, 1), (16.0, 17.0, 0), (7.5, 8.3, 0))}
SLAM_SEGMENTS_S = ((1.5, 13.5, False), (13.0, 19.0, True), (18.5, 29.0, False))   # (t0, t1, lost at the end) per wrist camera
JAW_EVENTS_S = {"left": (4.0, 6.0, 15.0), "right": (5.0, 13.5, 18.0)}


def _envelope(tS: np.ndarray) -> np.ndarray:
    tOut = np.zeros_like(tS)
    for tT0, tDur in DEMOS:
        tA, tB = tT0 + 1.0, tT0 + tDur - 1.0
        tIn = (tS >= tA) & (tS <= tB)
        tOut[tIn] = np.sin(np.pi * (tS[tIn] - tA) / (tB - tA)) ** 2
    return tOut


def bodyCentre(arm: str, tS: np.ndarray) -> np.ndarray:
    tE = _envelope(tS)[:, None]
    tPhase = 0.0 if arm == "left" else 1.3
    tMove = np.stack([0.07 * np.sin(2 * np.pi * 0.35 * tS + tPhase), 0.05 * np.sin(2 * np.pi * 0.27 * tS + 2 * tPhase),
                      0.04 * np.sin(2 * np.pi * 0.45 * tS + tPhase)], 1)
    return START[arm] + tE * tMove


def tcpRotation(arm: str, tS: np.ndarray) -> R:
    tE = _envelope(tS)[:, None]
    tPhase = 0.4 if arm == "left" else 2.0
    tWobble = np.stack([0.35 * np.sin(2 * np.pi * 0.3 * tS + tPhase), 0.25 * np.sin(2 * np.pi * 0.22 * tS),
                        0.30 * np.sin(2 * np.pi * 0.4 * tS + 2 * tPhase)], 1)
    return R.from_rotvec(np.repeat(BASE_ROTVEC[arm][None], len(tS), 0)) * R.from_rotvec(tE * tWobble)


def tcpPosition(arm: str, tS: np.ndarray) -> np.ndarray:
    tOffset = np.array(OFFSETS_MM[RIG_OF_ARM[arm]]) / 1000
    return bodyCentre(arm, tS) - tcpRotation(arm, tS).apply(tOffset)


def baseFromCamera() -> np.ndarray:
    tT = np.eye(4)
    tT[:3, :3] = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    tT[:3, 3] = CAM_POS
    return tT


def frameTimesNs() -> np.ndarray:
    tN = int(DURATION_S * FPS)
    tJitter = np.round(np.sin(np.arange(tN) * 1.7) * 1e6).astype(np.int64)
    return T_START_NS + np.round(np.arange(tN) / FPS * 1e9).astype(np.int64) + tJitter


def tagCounts(arm: str, tS: np.ndarray) -> np.ndarray:
    tN = np.full(len(tS), 2, int)
    for tA, tB, tCount in TAG_GAPS[arm]:
        tN[(tS >= tA) & (tS < tB)] = tCount
    return tN


def tagTrajectory(arm: str):
    """(t_ns, ntags, p_tag, R_tag matrices, p_B production, R_B_tcp production) on the Orbbec frames"""
    tT = frameTimesNs()
    tS = (tT - T_START_NS) / 1e9
    tRng = np.random.default_rng(11 if arm == "left" else 12)
    tN = tagCounts(arm, tS)
    tP = tcpPosition(arm, tS)
    tR = tcpRotation(arm, tS)
    tPTag = tP + tRng.normal(0, 0.002, tP.shape)
    tRTag = (tR * R.from_rotvec(tRng.normal(0, np.radians(0.5), (len(tS), 3)))).as_matrix()
    tPProd = tP + tRng.normal(0, 0.001, tP.shape)
    tRProd = (tR * R.from_rotvec(tRng.normal(0, np.radians(0.3), (len(tS), 3)))).as_matrix()
    return tT, tN, tPTag, tRTag, tPProd, tRProd


def imuSamples(arm: str):
    """(t_ns truth, device_us, host_ns, gyro (n,3) rad/s, acc (n,3) m/s^2) in the rig frame"""
    tT = T_START_NS - int(0.5e9) + np.round(np.arange(int((DURATION_S + 1.0) * IMU_HZ)) / IMU_HZ * 1e9).astype(np.int64)
    tS = (tT - T_START_NS) / 1e9
    tH = 1.0 / IMU_HZ
    tR0, tR1 = tcpRotation(arm, tS - tH / 2), tcpRotation(arm, tS + tH / 2)
    tOmega = (tR0.inv() * tR1).as_rotvec() / tH
    tP = [tcpPosition(arm, tS + tD) for tD in (-tH, 0.0, tH)]
    tAcc = (tP[0] - 2 * tP[1] + tP[2]) / tH ** 2 + np.array([0.0, 0.0, GRAVITY])
    tF = tcpRotation(arm, tS).inv().apply(tAcc)
    tRng = np.random.default_rng(21 if arm == "left" else 22)
    tGyro = tOmega + GYRO_BIAS[RIG_OF_ARM[arm]] + tRng.normal(0, 0.002, tOmega.shape)
    tF = tF + tRng.normal(0, 0.02, tF.shape)
    tDevice = ((tT - T_START_NS) // 1000 + 7_000_000).astype(np.int64)
    tHost = tT + 2_000_000 + np.abs(tRng.normal(0, 3e5, len(tT))).astype(np.int64)
    return tT, tDevice, tHost, tGyro.astype(np.float32), tF.astype(np.float32)


def cameraPoses(arm: str, tS: np.ndarray):
    """wrist camera centre / rotation in the base frame at orbbec-clock seconds tS (TCP = centre + R_B_cam tX)"""
    tRbc = tcpRotation(arm, tS) * RX.inv()
    return tcpPosition(arm, tS) - tRbc.apply(TX), tRbc


def slamSegments(arm: str):
    """per segment: (frame times on the wrist clock, p in the segment world, R camera->segment world, lost_end)"""
    tOut = []
    for tI, (tA, tB, tLost) in enumerate(SLAM_SEGMENTS_S):
        tTw = T_START_NS + np.round(np.arange(tA, tB, 1 / SLAM_HZ) * 1e9).astype(np.int64)
        tS = (tTw - SLAM_DT_NS - T_START_NS) / 1e9
        tPc, tRc = cameraPoses(arm, tS)
        tWorld = R.from_rotvec([0.2 * tI, -0.3, 0.1 + 0.5 * tI])
        tScale = 0.6 + 0.2 * tI
        tRng = np.random.default_rng(31 + tI + (0 if arm == "left" else 10))
        tP = tWorld.apply(tPc + np.array([0.1 * tI, -0.2, 0.05])) / tScale + tRng.normal(0, 0.0005, tPc.shape)
        tOut.append((tTw, tP, tWorld * tRc, tLost))
    return tOut


def syntheticArm(arm: str) -> dict:
    """the dict fgo_data.load_arm(take, arm) returns for this world (IMU already on the orbbec clock, hand-eye set, SLAM segments)"""
    tT, tN, tPTag, tRTag, tPProd, tRProd = tagTrajectory(arm)
    tTi, _, tHost, tGyro, tAcc = imuSamples(arm)
    tRig = RIG_OF_ARM[arm]
    tD = dict(take=TAKE, arm=arm, rig=tRig, cam=CAM_OF_ARM[arm], t=tT, ntags=tN, p_tag=tPTag, p_fused=tPProd, R_fused=tRProd,
              slam_ok=np.zeros(len(tT), bool), fused_valid=tN > 0)
    tD["R_tag"] = np.where((tN > 0)[:, None, None], tRTag, np.nan)
    tD["n_tagrot"] = tN.copy()
    tD["imu_t"] = tTi + 2_000_000
    tD["gyro"] = tGyro.astype(float)
    tD["acc"] = tAcc.astype(float)
    tD["RX"] = RX
    tD["tX"] = TX.copy()
    tD["handeye_src"] = "synthetic"
    tSegs = []
    for tI, (tTw, tP, tR, _) in enumerate(slamSegments(arm)):
        tSegs.append(dict(t=tTw - SLAM_DT_NS, p=tP, R=tR, start=int(tI * 1000)))
    tD["segs"] = tSegs
    tD["no_slam"] = False
    return tD


# ------------------------------------------------------------------------------------------------ rendering
def _render(tPositions: dict) -> tuple[np.ndarray, np.ndarray]:
    tTbc = baseFromCamera()
    tV, tU = np.mgrid[0:HEIGHT, 0:WIDTH]
    tDirC = np.stack([(tU - CX) / FX, (tV - CY) / FY, np.ones_like(tU, float)], -1)
    tDirB = tDirC @ tTbc[:3, :3].T
    tO = tTbc[:3, 3]
    tDepth = -tO[2] / tDirB[..., 2]                                    # table plane z = 0 (camera z == ray parameter)
    tGray = np.full((HEIGHT, WIDTH), 200.0)
    tGray += 20 * np.sin(tU / 7.0) * np.cos(tV / 9.0)
    for tC in tPositions.values():
        tOc = tO - tC
        tB = (tDirB * tOc).sum(-1)
        tA = (tDirB * tDirB).sum(-1)
        tDisc = tB ** 2 - tA * ((tOc * tOc).sum() - BODY_R_M ** 2)
        tHit = tDisc > 0
        tS = np.where(tHit, (-tB - np.sqrt(np.maximum(tDisc, 0))) / tA, np.inf)
        tCloser = tHit & (tS < tDepth)
        tDepth = np.where(tCloser, tS, tDepth)
        tGray = np.where(tCloser, 15.0, tGray)
    return np.clip(tGray, 0, 255).astype(np.uint8), np.round(tDepth * 1000).astype(np.uint16)


TAG_PAIR_OFFSET_M = 0.06          # tag 201 sits 6 cm along the rig x axis from tag 200 (both 48 mm, facing the camera)


def _tagImage() -> np.ndarray:
    """tags 200 and 201 of the right UMI rig, fronto-parallel 0.336 m in front of the camera (80 px = 48 mm at f = 560)"""
    import cv2
    tDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tImg = np.full((HEIGHT, WIDTH), 255, np.uint8)
    tImg[200:280, 300:380] = cv2.aruco.generateImageMarker(tDict, 200, 80)
    tImg[200:280, 400:480] = cv2.aruco.generateImageMarker(tDict, 201, 80)
    return tImg


# ------------------------------------------------------------------------------------------------ files
def _writeJson(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False))


def worldEnv(root: Path, fusionWork: Path) -> dict:
    root = Path(root)
    return dict(EGO_DATA_ROOT=str(root / "ego"), EGO_TRAIN_ROOT=str(root / "train"), EGO_CKPT_ROOT=str(root / "ckpt"),
                EGO_DESKTOP=str(root / "desktop"), EGO_FUSION_WORK=str(fusionWork), TAGRIG_STATE_DIR=str(root / "train/deploy"))


def protocol() -> list[dict]:
    return [dict(text="준비", t0=0.0, dur=2.0), dict(text="자유: 접기", t0=DEMOS[0][0], dur=DEMOS[0][1]),
            dict(text="쉬기", t0=14.0, dur=2.0), dict(text="자유: 펴기", t0=DEMOS[1][0], dur=DEMOS[1][1])]


IDLE_TRIM_FIELDS = ["take", "demo", "demo_len_s", "keep_start_s", "keep_end_s", "cut_head_s", "cut_tail_s", "kept_s", "pose_src"]


def idleTrimRows() -> list[dict]:
    return [dict(take=TAKE, demo=tI, demo_len_s=12.0, keep_start_s=0.6, keep_end_s=11.4, cut_head_s=0.6, cut_tail_s=0.6, kept_s=10.8, pose_src="tag_only")
            for tI in range(len(DEMOS))]


def writeIdleTrim(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as tHandle:
        tWriter = csv.DictWriter(tHandle, fieldnames=IDLE_TRIM_FIELDS)
        tWriter.writeheader()
        tWriter.writerows(rows)


def writeWorld(root: Path, fusionWork: Path, withTrims: bool = True) -> None:
    """fusionWork = the EGO_FUSION_WORK root the code under test uses (the production layout below it)"""
    import cv2
    root = Path(root)
    tEgo, tTrain = root / "ego", root / "train"
    tTake = tEgo / "datasets/tag_rig" / TAKE
    tTake.mkdir(parents=True, exist_ok=True)
    tWork = Path(fusionWork)
    (root / "ckpt").mkdir(parents=True, exist_ok=True)
    (root / "desktop").mkdir(parents=True, exist_ok=True)
    tTbc = baseFromCamera()
    tCal = dict(T_B_C=tTbc.tolist(), T_C_B=np.linalg.inv(tTbc).tolist())
    for tVer in range(1, 6):
        _writeJson(tEgo / f"configs/calibration/robot_cam_v00{tVer}.json", tCal)
    tImuCfg = {"UMI 오른손": dict(imu=dict(R_I_rig=np.eye(3).tolist(), offset_ms=0.0)),
               "UMI 왼손": dict(imu=dict(R_I_rig=np.eye(3).tolist(), offset_ms=0.0))}
    _writeJson(tEgo / "configs/calibration/umi_tag_rig_v001.json", tImuCfg)
    tShift = np.eye(4)
    tShift[0, 3] = TAG_PAIR_OFFSET_M
    tRigCfg = {"UMI 오른손": dict(T_rig_tag={"200": np.eye(4).tolist(), "201": tShift.tolist()}),
               "UMI 왼손": dict(T_rig_tag={"210": np.eye(4).tolist()})}
    _writeJson(tEgo / "configs/calibration/umi_tag_rig_v002.json", tRigCfg)
    tT = frameTimesNs()
    _writeJson(tTake / "meta.json",
               dict(t_start_monotonic_ns=T_START_NS, protocol=protocol(), intrinsics=dict(fx=FX, fy=FY, cx=CX, cy=CY),
                    frames=len(tT), depth_scale=1.0, tag_ids=[200, 210]))
    np.save(tTake / "t_ns.npy", tT)
    tS = (tT - T_START_NS) / 1e9
    tCentres = {tArm: bodyCentre(tArm, tS) for tArm in ARMS}
    for tK in range(len(tT)):
        tGray, tDepth = _render({tArm: tCentres[tArm][tK] for tArm in ARMS})
        cv2.imwrite(str(tTake / f"color_{tK:05d}.jpg"), cv2.cvtColor(tGray, cv2.COLOR_GRAY2BGR))
        cv2.imwrite(str(tTake / f"depth_{tK:05d}.png"), tDepth)
    tTagRot = dict(t_ns=tT)
    for tArm in ARMS:
        tRig = RIG_OF_ARM[tArm]
        tTt, tN, tPTag, tRTag, tPProd, tRProd = tagTrajectory(tArm)
        np.savez(tTake / f"umi_traj_{tRig}.npz", t_ns=tTt, ntags=tN, p_B=tPProd, R_B_tcp=tRProd, p_B_tag=tPTag)
        tTagRot[f"R_{tRig}"] = np.where((tN > 0)[:, None, None], tRTag, np.nan)
        tTagRot[f"n_{tRig}"] = tN
        tTagRot[f"rmse_{tRig}"] = np.where(tN > 0, 0.5, np.nan)
        _, tDevice, tHost, tGyro, tAcc = imuSamples(tArm)
        np.savez(tTake / f"imu_{tRig}.npz", device_us=tDevice, host_ns=tHost, gyro=tGyro, acc=tAcc)
        tCam = tTake / CAM_OF_ARM[tArm]
        tCam.mkdir(exist_ok=True)
        tJawT = T_START_NS + np.round(np.arange(0, DURATION_S, 0.1) * 1e9).astype(np.int64)
        tState = np.zeros(len(tJawT), int)
        for tE in JAW_EVENTS_S[tArm]:
            tState[(tJawT - T_START_NS) / 1e9 >= tE] ^= 1
        np.savez(tCam / "umi_jaw.npz", t_ns=tJawT, state=tState)
        tWristT = T_START_NS + np.round(np.arange(0, DURATION_S, 1.0) * 1e9).astype(np.int64)
        np.save(tCam / "t_ns.npy", tWristT)
        for tI in range(len(tWristT)):
            cv2.imwrite(str(tCam / f"wrist_{tI:05d}.jpg"), np.full((60, 80, 3), 40 + 5 * tI, np.uint8))
        _writeSlam(tEgo, tArm, tWork)
    (tEgo / "datasets/m3_runs/_fgo_1023/tagrot").mkdir(parents=True, exist_ok=True)
    np.savez(tEgo / "datasets/m3_runs/_fgo_1023/tagrot" / f"{TAKE}.npz", **tTagRot)
    # derived (production) take + one 10-22 fused take for the mean hand-eye
    tAsg = dict(rig_of_arm=RIG_OF_ARM, cam_of_arm=CAM_OF_ARM)
    _writeJson(tTrain / "data/tag_rig_derived" / TAKE / "assignment.json", tAsg)
    _writeJson(tTrain / "data/tag_rig_derived" / TAKE / "segments.json", [])
    _writeJson(tTrain / "data/tag_rig_derived" / f"{HANDEYE_TAKE}_fused" / "assignment.json", tAsg)
    for tArm in ARMS:
        tTt, tN, tPTag, tRTag, tPProd, tRProd = tagTrajectory(tArm)
        np.savez(tTrain / "data/tag_rig_derived" / TAKE / f"umi_traj_{tArm}.npz", t_ns=tTt, ntags=tN, p_B=tPProd, R_B_tcp=tRProd,
                 source_rig_label=np.array(RIG_OF_ARM[tArm]))
        np.savez(tTrain / "data/tag_rig_derived" / f"{HANDEYE_TAKE}_fused" / f"umi_traj_{tArm}.npz", handeye_R=RX.as_matrix(),
                 handeye_t=TX)
    writeIdleTrim(tTrain / "reports/idle_trim.csv", idleTrimRows() if withTrims else [])
    # fusion working tree: start seeds, rigid body models, pooled hand-eye
    tV2b = tWork / "orbbec/v2b"
    _writeJson(tV2b / "start_seed_v1.json", {tArm: START[tArm].round(3).tolist() for tArm in ARMS})
    tGolden = np.pi * (3 - np.sqrt(5))
    tI = np.arange(400)
    tZ = 1 - 2 * (tI + 0.5) / 400
    tModel = np.stack([np.sqrt(1 - tZ ** 2) * np.cos(tGolden * tI), np.sqrt(1 - tZ ** 2) * np.sin(tGolden * tI), tZ],
                      1) * BODY_R_M
    (tV2b / "out").mkdir(parents=True, exist_ok=True)
    for tRig in ("left", "right"):
        np.savez(tV2b / "out" / f"rig_model_{tRig}.npz", pts=tModel)
    (tWork / "out").mkdir(parents=True, exist_ok=True)
    for tCamName in ("wrist", "wrist_left"):
        np.save(tWork / "out" / f"RX_pooled_{tCamName}.npy", RX.as_quat())
    # one tag-rotation take for tag_rot_extract: three frames, the middle one shows tag 200
    tTagTake = tEgo / "datasets/tag_rig" / "20261030_130000"
    tTagTake.mkdir(parents=True, exist_ok=True)
    _writeJson(tTagTake / "meta.json", dict(t_start_monotonic_ns=T_START_NS, protocol=protocol(), intrinsics=dict(fx=FX, fy=FY,
            cx=CX, cy=CY)))
    np.save(tTagTake / "t_ns.npy", T_START_NS + np.arange(3, dtype=np.int64) * 66_666_667)
    for tK in range(3):
        cv2.imwrite(str(tTagTake / f"color_{tK:05d}.jpg"), _tagImage() if tK == 1 else np.full((HEIGHT, WIDTH), 255, np.uint8))


def _writeSlam(tEgo: Path, arm: str, tWork: Path) -> None:
    """MASt3R raw layout (manifest windows + segment csvs, frames.json) and the generic visual npz for one wrist camera"""
    tCam = CAM_OF_ARM[arm]
    tTag = f"umi_{TAKE}" + ("" if tCam == "wrist" else "_left")
    tRun = tEgo / "datasets/m3_runs" / tTag
    tSegs = slamSegments(arm)
    tAllT = np.unique(np.concatenate([tSeg[0] for tSeg in tSegs]))
    tIndex = {int(tT): tI for tI, tT in enumerate(tAllT)}
    _writeJson(tEgo / "datasets/m3_frames" / tTag / "frames.json",
               dict(frames=[dict(i=tI, t_ns=int(tT)) for tI, tT in enumerate(tAllT)]))
    tWinDir = tRun / "windows/w000/segs"
    tWinDir.mkdir(parents=True, exist_ok=True)
    tManifestSegs = []
    tVis = dict(t_ns=[], p=[], q_xyzw=[], segment_id=[])
    for tI, (tTw, tP, tR, tLost) in enumerate(tSegs):
        tS0 = tIndex[int(tTw[0])]
        tQ = tR.as_quat()
        tRows = []
        for tJ in range(len(tTw)):
            tIsLost = tLost and tJ == len(tTw) - 1
            tRows.append(dict(frame_idx=tJ, q_x=tQ[tJ, 0], q_y=tQ[tJ, 1], q_z=tQ[tJ, 2], q_w=tQ[tJ, 3], x=tP[tJ, 0], y=tP[tJ, 1],
                              z=tP[tJ, 2], is_lost="True" if tIsLost else "False", state="TRACKING" if tJ else "INIT",
                              sim3_scale=1.0 + 0.01 * np.sin(tJ / 9), is_keyframe="True" if tJ % 5 == 0 else "False"))
        tName = f"{tS0:06d}_seg{tI}.csv"
        for tDir in (tWinDir, tRun / "segs"):
            tDir.mkdir(parents=True, exist_ok=True)
            with open(tDir / tName, "w", newline="") as tHandle:
                tWriter = csv.DictWriter(tHandle, fieldnames=list(tRows[0]))
                tWriter.writeheader()
                tWriter.writerows(tRows)
        tManifestSegs.append(dict(csv=f"segs/{tName}", cut=len(tTw) - (1 if tLost else 0), start=tS0))
        tVis["t_ns"].append(tTw)
        tVis["p"].append(tP)
        tVis["q_xyzw"].append(tQ)
        tVis["segment_id"].append(np.full(len(tTw), tI))
    _writeJson(tRun / "segs/manifest.json", dict(windows=[dict(sub="w000", t0_ns=int(tAllT[0]), t1_ns=int(tAllT[-1]), w=0, i0=0)],
                                                 segments=tManifestSegs))
    np.savez(tWork / f"visual_{TAKE}_{arm}.npz", **{tKey: np.concatenate(tVal) for tKey, tVal in tVis.items()})


def fakeTracks(packDemoDir: Path) -> None:
    """stand-in for CoTracker: every query pixel is lifted to the base frame at its query frame, attached to the nearest body
    (rigid with its rotation), and projected into every frame; visibility 1 from the query frame on, NaN before"""
    tMeta = json.loads((packDemoDir / "meta.json").read_text())
    tT = np.array(tMeta["t_ns"], np.int64)
    tS = (tT - T_START_NS) / 1e9
    tTbc = baseFromCamera()
    tTcb = np.linalg.inv(tTbc)
    tQ = np.array(tMeta["queries"], float)
    tTracks = np.full((len(tT), len(tQ), 2), np.nan)
    tVis = np.zeros((len(tT), len(tQ)))
    tCentres = {tArm: bodyCentre(tArm, tS) for tArm in ARMS}
    tRots = {tArm: tcpRotation(tArm, tS) for tArm in ARMS}
    import cv2
    tTake = Path(os.environ["EGO_DATA_ROOT"]) / "datasets/tag_rig" / tMeta["take"]
    for tN, (tJ, tU, tV) in enumerate(tQ):
        tJ = int(tJ)
        tDepth = cv2.imread(str(tTake / f"depth_{tMeta['ks'][tJ]:05d}.png"), -1).astype(np.float32) / 1000
        tZ = float(tDepth[int(tV), int(tU)])
        tXc = np.array([(tU - CX) / FX * tZ, (tV - CY) / FY * tZ, tZ])
        tXb = tTbc[:3, :3] @ tXc + tTbc[:3, 3]
        tArm = min(ARMS, key=lambda a: np.linalg.norm(tCentres[a][tJ] - tXb))
        tLocal = tRots[tArm][tJ].inv().apply(tXb - tCentres[tArm][tJ])
        tWorld = np.full((len(tT), 3), np.nan)
        tWorld[tJ:] = tCentres[tArm][tJ:] + tRots[tArm][tJ:].apply(tLocal)
        tVisN = np.zeros(len(tT))
        tVisN[tJ:] = 1.0
        # failure modes for the solver / gates: hand out of view, stale points, drifting outliers, L/R collision
        tKind = tMeta["query_kind"][tN]
        if tMeta["demo"] == 0 and tArm == "right":
            tVisN[30:46] = 0.0
            if tKind == 0:
                tVisN[30:] = 0.0                     # the seed points never come back: re-acquisition from reserve points
        if tMeta["demo"] == 0 and tArm == "left" and tJ <= 55:
            tWorld[55:101] = tWorld[55]
        if tMeta["demo"] == 1 and tArm == "left" and tKind == 0 and tN % 5 in (0, 1, 2):
            tAngle = 2.4 * tN
            tDrift = np.clip(np.arange(len(tT)) - 40, 0, None)[:, None] * 0.002
            tWorld += tDrift * np.array([np.cos(tAngle), np.sin(tAngle), 0.3])
        if tMeta["demo"] == 1 and tArm == "right":
            tWorld[50:61] += tCentres["left"][50:61] - tCentres["right"][50:61] + np.array([0.02, 0.0, 0.0])
            tVisN[-14:] = 0.0                        # leaves the view at the end of the window: edge-gap trim
        tCamPts = tWorld @ tTcb[:3, :3].T + tTcb[:3, 3]
        tTracks[:, tN, 0] = FX * tCamPts[:, 0] / tCamPts[:, 2] + CX
        tTracks[:, tN, 1] = FY * tCamPts[:, 1] / tCamPts[:, 2] + CY
        tVis[:, tN] = tVisN
    np.savez(packDemoDir / "tracks.npz", tracks=tTracks.astype(np.float32), vis=tVis.astype(np.float32))
