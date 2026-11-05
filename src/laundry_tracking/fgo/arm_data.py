"""Per take / physical arm data access for the pose stage (was _fgo_1023/tools/fgo_data.py, 10-23).

Inputs (laundry_common.paths): the recorded take (datasets/tag_rig/<take>: meta.json, umi_traj_<rig>.npz, imu_<rig>.npz), the derived
assignment (data/tag_rig_derived/<take>_fused), raw per-frame tag rotations (m3_runs/_fgo_1023/tagrot/<take>.npz), the IMU calibration
(configs/calibration/umi_tag_rig_v001.json), raw MASt3R segments (m3_runs/umi_<take>[_left]/segs + m3_frames/.../frames.json) and the
idle-trim table (reports/idle_trim.csv).
Output: the per-arm record `D` (a dict mirroring those files: orbbec-grid tag + fused pose, raw tag rotation, IMU in the TCP frame
on the orbbec clock, hand-eye, visual segments) and the demo windows of a take.

All times are on the orbbec clock: SLAM pose at orbbec time t is the wrist pose at t + SLAM_DT_NS (chain_v1 convention); IMU = host
time of the device clock fit + the calibrated offset (as tag_rig_fuse).
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from laundry_common import paths

from ..constants import NS_PER_MS, NS_PER_S

M3_RUNS = paths.DATA_ROOT / "datasets/m3_runs"
M3_FRAMES = paths.DATA_ROOT / "datasets/m3_frames"
FGO_DIR = M3_RUNS / "_fgo_1023"
ARCHIVE = paths.CKPT_ROOT / "tag_rig_raw"          # 10-26: raw takes 10-16..10-23 moved to the PortableSSD
# The 10-22 fused takes whose chain_v1 hand-eye is averaged for takes without their own fusion (meanHandEye1022).
HANDEYE_TAKES = ("20261022_121938", "20261022_123614", "20261022_124754", "20261022_130007", "20261022_134315", "20261022_143329",
                 "20261022_144321", "20261022_160308", "20261022_161756", "20261022_170651", "20261023_113550", "20261023_114817",
                 "20261023_123747")
RIG_NAMES = {"right": "UMI 오른손", "left": "UMI 왼손"}
SLAM_DT_NS = 20e6                  # chain_v1: SLAM pose at orbbec time t is the wrist pose at t + 20 ms
FREE_CUE_PREFIX = "자유:"          # protocol cue of one free demo window
SLAM_OK_STATES = ("INIT", "TRACKING")
MIN_SEGMENT_FRAMES = 10            # shorter SLAM / visual segments carry no usable shape
MIN_QUAT_NORM = 1e-3               # below: an all-zero (invalid) quaternion row


class ArmDataError(RuntimeError):
    """A take's files are inconsistent with each other or a required input is missing."""


@dataclass(frozen=True)
class DemoWindow:
    """One kept demo of a take (idle_trim.csv row + the protocol cue), host monotonic ns."""
    mDemo: int
    mA0: int          # cue start
    mW0: int          # kept window start
    mW1: int          # kept window end
    mKeptS: float


def _takeMeta(take: str) -> Path | None:
    tFile = paths.TAG_RIG / take / "meta.json"
    if not tFile.exists():
        tFile = ARCHIVE / take / "meta.json"
    return tFile if tFile.exists() else None


def loadTrimWindows() -> dict[str, list[DemoWindow]]:
    """{take: [DemoWindow per idle_trim.csv row]} for every take whose meta.json is found (was fgo_data.trims)"""
    tRowsByTake = {}
    for tRow in csv.DictReader(open(paths.IDLE_TRIM_CSV)):
        tRowsByTake.setdefault(tRow["take"], []).append(tRow)
    tOut = {}
    for tTake, tRows in tRowsByTake.items():
        tMetaFile = _takeMeta(tTake)
        if tMetaFile is None:
            continue
        tMeta = json.loads(tMetaFile.read_text())
        tT0 = int(tMeta["t_start_monotonic_ns"])
        tCues = [tP for tP in tMeta["protocol"] if tP["text"].startswith(FREE_CUE_PREFIX)]
        tOut[tTake] = [DemoWindow(mDemo=int(tRow["demo"]), mA0=tT0 + int(tCues[int(tRow["demo"])]["t0"] * NS_PER_S),
                                  mW0=tT0 + int((tCues[int(tRow["demo"])]["t0"] + float(tRow["keep_start_s"])) * NS_PER_S),
                                  mW1=tT0 + int((tCues[int(tRow["demo"])]["t0"] + float(tRow["keep_end_s"])) * NS_PER_S),
                                  mKeptS=float(tRow["kept_s"]))
                       for tRow in tRows]
    return tOut


def demoWindow(take: str, demo: int) -> DemoWindow:
    return {tW.mDemo: tW for tW in loadTrimWindows()[take]}[demo]


def loadAssignment(take: str) -> tuple[dict, dict, bool]:
    """(rig_of_arm, cam_of_arm, fused): the derived <take>_fused/assignment.json, else the rig on the +y side is the left arm"""
    tFile = paths.DERIVED / f"{take}_fused/assignment.json"
    if tFile.exists():
        tAsg = json.loads(tFile.read_text())
        return tAsg["rig_of_arm"], tAsg["cam_of_arm"], True
    tMedianY = {}
    for tSide in ("left", "right"):
        tZ = np.load(paths.TAG_RIG / take / f"umi_traj_{tSide}.npz")
        tMedianY[tSide] = float(np.median(tZ["p_B"][tZ["ntags"] > 0, 1]))
    tRigOf = {("left" if tMedianY[tSide] > 0 else "right"): tSide for tSide in tMedianY}
    if set(tRigOf) != {"left", "right"}:
        raise ArmDataError(f"{take}: both rigs on one side of the base, median y {tMedianY}")
    return tRigOf, {"left": "wrist_left", "right": "wrist"}, False


def slamTag(take: str, cam: str) -> str:
    """MASt3R run name of a wrist camera: umi_<take> (wrist) / umi_<take>_left (wrist_left)"""
    return f"umi_{take}" + ("" if cam == "wrist" else "_left")


def loadSlamSegments(manifest, framesJson) -> list[dict]:
    """merged MASt3R segments -> [dict(t (orbbec clock), p, R camera->segment world, start)] (was fgo_data.load_segments)"""
    tFrameTime = {tF["i"]: tF["t_ns"] for tF in json.loads(Path(framesJson).read_text())["frames"]}
    tSegs = []
    for tSeg in json.loads(Path(manifest).read_text())["segments"]:
        tCsv = Path(manifest).parent / Path(tSeg["csv"]).name
        if not tCsv.exists() or tSeg["cut"] < MIN_SEGMENT_FRAMES:
            continue
        tT, tP, tQ = [], [], []
        for tRow in csv.DictReader(open(tCsv)):
            tK = int(tRow["frame_idx"])
            if tK >= tSeg["cut"] or tRow["state"] not in SLAM_OK_STATES:
                continue
            tQuat = [float(tRow[tC]) for tC in ("q_x", "q_y", "q_z", "q_w")]
            tXyz = [float(tRow["x"]), float(tRow["y"]), float(tRow["z"])]
            if not np.all(np.isfinite(tQuat + tXyz)) or np.linalg.norm(tQuat) < MIN_QUAT_NORM or (tSeg["start"] + tK) not in tFrameTime:
                continue
            tT.append(tFrameTime[tSeg["start"] + tK])
            tP.append(tXyz)
            tQ.append(tQuat)
        if len(tT) >= MIN_SEGMENT_FRAMES:
            tTimes = np.array(tT, np.int64)
            tOrder = np.argsort(tTimes, kind="stable")
            tTimes, tKeep = np.unique(tTimes[tOrder], return_index=True)
            tOrder = tOrder[tKeep]
            tSegs.append(dict(t=tTimes - int(SLAM_DT_NS), p=np.array(tP)[tOrder], R=R.from_quat(np.array(tQ)[tOrder]),
                              start=tSeg["start"]))
    return tSegs


def loadArm(take: str, arm: str, withSlam: bool = True) -> dict:
    """the per-arm record D of a take (was fgo_data.load_arm); see the module docstring for the clock conventions"""
    from laundry_shared.handumi_collector.pose.timing import fit_device_to_host
    tRigOf, tCamOf, tFused = loadAssignment(take)
    tRig = tRigOf[arm]
    tCam = tCamOf[arm]
    if tFused:
        tU = dict(np.load(paths.DERIVED / f"{take}_fused/umi_traj_{arm}.npz", allow_pickle=True))
    else:
        tU = dict(np.load(paths.TAG_RIG / take / f"umi_traj_{tRig}.npz", allow_pickle=True))
    tT = tU["t_ns"].astype(np.int64)
    tN = tU["ntags"]
    D = dict(take=take, arm=arm, rig=tRig, cam=tCam, raw=tU, t=tT, ntags=tN, p_tag=tU.get("p_B_tag", tU["p_B"]),
             p_fused=tU["p_B"], R_fused=tU["R_B_tcp"], slam_ok=tU.get("slam_ok", np.zeros(len(tT), bool)).astype(bool),
             fused_valid=tU.get("fused_valid", tN > 0).astype(bool))
    # raw tag rotation (PnP only) on the same orbbec frames
    tTagRot = np.load(FGO_DIR / "tagrot" / f"{take}.npz")
    if not (np.array_equal(tTagRot["t_ns"].astype(np.int64), tT) or len(tTagRot["t_ns"]) == len(tT)):
        raise ArmDataError(f"{take}: tagrot frames do not match umi_traj_{tRig}")
    D["R_tag"] = tTagRot[f"R_{tRig}"]
    D["n_tagrot"] = tTagRot[f"n_{tRig}"]
    # IMU on the orbbec clock (same mapping as tag_rig_fuse), expressed in the TCP (= rig) frame
    tCfg = json.loads((paths.DATA_CALIB / "umi_tag_rig_v001.json").read_text())[RIG_NAMES[tRig]]
    tImu = np.load(paths.TAG_RIG / take / f"imu_{tRig}.npz")
    tRigFromImu = np.array(tCfg["imu"]["R_I_rig"])
    tImuT = fit_device_to_host(tImu["device_us"],
                               tImu["host_ns"]).to_host_ns(tImu["device_us"]).astype(np.int64) + int(tCfg["imu"]["offset_ms"] * NS_PER_MS)
    tOrder = np.argsort(tImuT, kind="stable")
    tImuT = tImuT[tOrder]
    D["imu_t"] = tImuT
    D["gyro"] = tImu["gyro"][tOrder].astype(float) @ tRigFromImu          # v_rig = RI^T v_I
    D["acc"] = tImu["acc"][tOrder].astype(float) @ tRigFromImu
    D["segs"] = []
    D["no_slam"] = True
    # 10-23 fix: hand-eye always set -- own chain_v1 fit if fused, else the 10-22 mean for that camera (any visual front-end needs it)
    if tFused:
        D["RX"] = R.from_matrix(tU["handeye_R"])
        D["tX"] = np.array(tU["handeye_t"])
        D["handeye_src"] = "own_fused"
    else:
        D["RX"], D["tX"] = meanHandEye1022(tCam)
        D["handeye_src"] = "mean_1022_" + tCam
    if withSlam and slamReady(take, tCam):
        tTag = slamTag(take, tCam)
        D["segs"] = loadSlamSegments(M3_RUNS / tTag / "segs/manifest.json", M3_FRAMES / tTag / "frames.json")
        D["no_slam"] = len(D["segs"]) == 0
    return D


def slamReady(take: str, cam: str) -> bool:
    """merged MASt3R manifest present and (for service-processed takes) all windows collected"""
    tTag = slamTag(take, cam)
    tManifest = M3_RUNS / tTag / "segs/manifest.json"
    if not tManifest.exists() or not (M3_FRAMES / tTag / "frames.json").exists():
        return False
    return not json.loads(tManifest.read_text()).get("missing_windows")


def meanHandEye1022(cam: str) -> tuple[R, np.ndarray]:
    """10-23 takes have no chain_v1 fusion: camera->TCP hand-eye = rotation mean / translation median over the 10-22 fused arms
    with that camera (was fgo_data.handeye_1022)"""
    tRs, tTs = [], []
    for tTake in HANDEYE_TAKES:
        tFile = paths.DERIVED / f"{tTake}_fused/assignment.json"
        if not tFile.exists():
            continue
        tAsg = json.loads(tFile.read_text())
        for tArm, tC in tAsg["cam_of_arm"].items():
            if tC == cam:
                tU = np.load(paths.DERIVED / f"{tTake}_fused/umi_traj_{tArm}.npz")
                tRs.append(tU["handeye_R"])
                tTs.append(tU["handeye_t"])
    return R.from_matrix(np.array(tRs)).mean(), np.median(np.array(tTs), 0)


def setVisual(D: dict, npz, timeOffsetNs: int = 0, minConf: float | None = None) -> dict:
    """replace D["segs"] by a generic visual front-end (was fgo_data.set_visual): npz with t_ns (n,) on the WRIST-camera clock (as
    m3_frames/frames.json; the -20 ms SLAM_DT_NS is applied here like for MASt3R), p (n,3) camera centre in the segment world,
    q_xyzw (n,4) camera->world rotation, segment_id (n,), optional conf (n,) (frames with conf < minConf dropped).
    Scale / world of every segment are free (estimated by the graph)."""
    tZ = np.load(npz)
    tT = tZ["t_ns"].astype(np.int64) - int(SLAM_DT_NS) + int(timeOffsetNs)
    tIds = tZ["segment_id"]
    tKeep = np.ones(len(tT), bool)
    if minConf is not None and "conf" in tZ.files:
        tKeep &= tZ["conf"] >= minConf
    tSegs = []
    for tK in np.unique(tIds):
        tM = tKeep & (tIds == tK)
        if tM.sum() < MIN_SEGMENT_FRAMES:
            continue
        tOrder = np.argsort(tT[tM])
        tTimes, tU = np.unique(tT[tM][tOrder], return_index=True)
        tOrder = tOrder[tU]
        tSegs.append(dict(t=tTimes, p=tZ["p"][tM][tOrder], R=R.from_quat(tZ["q_xyzw"][tM][tOrder]), start=int(tK)))
    if "RX" not in D:
        raise ArmDataError("setVisual: D has no hand-eye RX/tX (load it with arm_data.loadArm)")
    D["segs"] = tSegs
    D["no_slam"] = len(tSegs) == 0
    return D


def exportVisual(D: dict, npz) -> None:
    """write D["segs"] (e.g. MASt3R) in the generic format of setVisual (wrist clock) (was fgo_data.export_visual)"""
    tT = np.concatenate([tS["t"] + int(SLAM_DT_NS) for tS in D["segs"]])
    tP = np.concatenate([tS["p"] for tS in D["segs"]])
    tQ = np.concatenate([tS["R"].as_quat() for tS in D["segs"]])
    tIds = np.concatenate([np.full(len(tS["t"]), tI) for tI, tS in enumerate(D["segs"])])
    np.savez(npz, t_ns=tT, p=tP, q_xyzw=tQ, segment_id=tIds)
