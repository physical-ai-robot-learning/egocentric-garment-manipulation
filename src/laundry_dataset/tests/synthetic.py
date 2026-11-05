"""Synthetic recording / training tree for the laundry_dataset tests: three tag-rig takes with two free demos each, the files the
dataset stage reads (meta.json, camera clocks, rig trajectories tag-only + fused, IMUs, wrist jaw labels, optional jpgs), the
calibration files, an idle-trim table and an arm-assigned derived copy. Deterministic (fixed seeds, analytic motion)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as Rot

T0_NS = 1_000_000_000_000
REC_S = 21.0
TRAJ_HZ = 15.0
CAM_HZ = 15.0
WRIST_HZ = 30.0
IMU_HZ = 100.0
TAKES = ("20261022_124754", "20261023_113550", "20261023_114817")
PROTOCOL = [{"text": "준비", "t0": 0.0, "dur": 2.0}, {"text": "자유: 펼치기", "t0": 2.0, "dur": 9.0},
            {"text": "정리", "t0": 11.0, "dur": 0.5}, {"text": "자유: 접기", "t0": 11.5, "dur": 9.0, "ended_early": True}]
# rig label -> base-frame y sign: the labels are swapped (rig 'right' is the robot LEFT arm, wears camera 'wrist_left')
RIG_Y = {"right": 0.22, "left": -0.22}
# rig label -> tag gaps (s) and the fused gaps (a subset)
TAG_GAPS = {"right": [(6.0, 6.8), (12.4, 12.55)], "left": [(9.5, 11.5), (14.0, 14.3)]}
FUSED_GAPS = {"right": [(6.0, 6.8)], "left": [(9.6, 11.4)]}
JAW_CLOSED = {"wrist_left": [(5.0, 7.5), (13.0, 15.0)], "wrist": [(4.0, 6.0), (16.0, 18.5)]}
JAW_INVALID = {"wrist_left": [(7.4, 7.7)], "wrist": [(16.9, 17.0)]}
R_U_E = Rot.from_euler("z", 90, degrees=True).as_matrix()
R_U_E_LEFT = Rot.from_euler("z", -90, degrees=True).as_matrix()


def _envelope(t):
    """motion amplitude: still for the first ~3 s of each demo window and after ~8 s"""
    tFirst = np.clip((np.sin(np.pi * (t - 2.0) / 9.0)) * 1.5, 0, 1) ** 2
    return tFirst + np.clip(np.sin(np.pi * (t - 11.5) / 9.0) * 1.5, 0, 1) ** 2


def _inRuns(t, runs):
    tOut = np.zeros(len(t), bool)
    for tA, tB in runs:
        tOut |= (t >= tA) & (t < tB)
    return tOut


def _gapLen(t, gap):
    tOut = np.zeros(len(t))
    tX = np.r_[0, gap.astype(int), 0]
    for tS, tE in zip(np.flatnonzero(np.diff(tX) == 1), np.flatnonzero(np.diff(tX) == -1)):
        tOut[tS:tE] = t[min(tE, len(t) - 1)] - t[max(tS - 1, 0)]
    return tOut


def rigPose(rig, t, phase=0.0):
    tE = _envelope(t)
    tSign = 1.0 if rig == "right" else -1.0
    tP = np.stack([0.30 + 0.08 * tE * np.sin(0.7 * t + phase), RIG_Y[rig] + 0.05 * tE * np.sin(0.5 * t + phase) * tSign,
                   0.035 + 0.04 * tE * np.sin(0.9 * t + phase)], 1)
    tF = 1.0 if rig == "right" else 2.3
    tRv = np.stack([0.3 * tE * np.sin(0.4 * tF * t), 0.2 * tE * np.cos(0.3 * t), 0.4 * tE * np.sin(0.2 * tF * t + phase)], 1)
    tR = (Rot.from_euler("y", 180, degrees=True) * Rot.from_rotvec(tRv)).as_matrix()
    return tP, tR


def writeTake(take: Path, seed: int, images: bool = False, fused: bool = True):
    take.mkdir(parents=True, exist_ok=True)
    tRng = np.random.default_rng(seed)
    (take / "meta.json").write_text(json.dumps({"t_start_monotonic_ns": T0_NS, "protocol": PROTOCOL}, ensure_ascii=False))
    (take / "segments.json").write_text("{}")
    tCam = np.arange(0.013, REC_S, 1 / CAM_HZ)
    np.save(take / "t_ns.npy", (T0_NS + tCam * 1e9).astype(np.int64))
    tTraj = np.arange(0.02, REC_S, 1 / TRAJ_HZ)
    tTrajNs = (T0_NS + tTraj * 1e9).astype(np.int64)
    for tRig in ("right", "left"):
        tP, tR = rigPose(tRig, tTraj, phase=0.3 * seed)
        tP = tP + tRng.normal(0, 0.0005, tP.shape)
        tTagGap = _inRuns(tTraj, TAG_GAPS[tRig])
        tFusedGap = _inRuns(tTraj, FUSED_GAPS[tRig])
        tNtags = np.where(tTagGap, 0, 1 + (np.arange(len(tTraj)) % 3)).astype(np.int64)
        np.savez(take / f"umi_traj_{tRig}.npz", t_ns=tTrajNs, p_B=tP, R_B_tcp=tR, ntags=tNtags)
        if fused:
            np.savez(take / f"umi_traj_{tRig}_fused.npz", t_ns=tTrajNs, p_B=tP, R_B_tcp=tR, ntags=tNtags, fused_valid=~tFusedGap,
                     gap_len_s=_gapLen(tTraj, tFusedGap))
        tImu = np.arange(0.0, REC_S, 1 / IMU_HZ)
        tGyro = np.stack([np.interp(tImu, tTraj, _envelope(tTraj)) * np.sin(3 * tImu + tK) * 1.2 for tK in range(3)], 1)
        tGyro = tGyro + tRng.normal(0, 0.01, tGyro.shape)
        np.savez(take / f"imu_{tRig}.npz", host_ns=(T0_NS + tImu * 1e9).astype(np.int64), gyro=tGyro)
    tWrist = np.arange(0.007, REC_S, 1 / WRIST_HZ)
    for tCamName in ("wrist", "wrist_left"):
        tDir = take / tCamName
        tDir.mkdir(exist_ok=True)
        np.save(tDir / "t_ns.npy", (T0_NS + tWrist * 1e9).astype(np.int64))
        tClosed = _inRuns(tWrist, JAW_CLOSED[tCamName])
        tState = (~tClosed).astype(np.int64)
        tValid = (~_inRuns(tWrist, JAW_INVALID[tCamName])).astype(np.int64)
        tGc = np.clip(np.where(tClosed, 0.1, 0.85) + 0.05 * np.sin(2 * tWrist), 0, 1)
        tEdges = np.flatnonzero(np.diff(tState) != 0) + 1
        np.savez(tDir / "umi_jaw.npz", t_ns=(T0_NS + tWrist * 1e9).astype(np.int64), state=tState, valid=tValid, g_cont=tGc,
                 g_is_aperture=np.array(1), ev_t=(T0_NS + tWrist[tEdges] * 1e9).astype(np.int64))
        if images:
            for tI in range(len(tWrist)):
                cv2.imwrite(str(tDir / f"wrist_{tI:05d}.jpg"), _image(tI, 1 if tCamName == "wrist" else 2))
    if images:
        for tI in range(len(tCam)):
            cv2.imwrite(str(take / f"color_{tI:05d}.jpg"), _image(tI, 0))


def _image(i, k):
    tY, tX = np.mgrid[0:36, 0:64]
    tImg = np.stack([(tX * 4 + i * 3) % 256, (tY * 7 + i * 5 + k * 40) % 256, ((tX + tY) * 2 + k * 90) % 256], -1)
    return tImg.astype(np.uint8)


def writeCalibration(calibDir: Path):
    calibDir.mkdir(parents=True, exist_ok=True)
    tRight = calibDir / "umi_gripper_align_v001.json"
    tLeft = calibDir / "umi_gripper_align_left_v001.json"
    tRight.write_text(json.dumps({"R_U_E": R_U_E.tolist()}))
    tLeft.write_text(json.dumps({"R_U_E": R_U_E_LEFT.tolist()}))
    tJaw = calibDir / "jaw_kpt_calib_v002.json"
    tJaw.write_text(json.dumps({"left": {"threshold_px": 191.0}, "right": {"threshold_px": 150.0}}))
    return {"right": tRight, "left": tLeft}, tJaw


ASSIGNMENT = {"rig_of_arm": {"left": "right", "right": "left"}, "cam_of_arm": {"left": "wrist_left", "right": "wrist"},
              "rig_labels_swapped": True, "cam_labels_swapped": True, "flow_corr": {"right|wrist_left": 0.9}}


def writeDerived(take: Path, derivedTake: Path, poseSrc: str):
    derivedTake.mkdir(parents=True, exist_ok=True)
    (derivedTake / "assignment.json").write_text(json.dumps(dict(ASSIGNMENT, take=str(take), pose_src=poseSrc or "tag_only")))
    for tArm, tRig in ASSIGNMENT["rig_of_arm"].items():
        tZ = dict(np.load(take / f"umi_traj_{tRig}{poseSrc}.npz", allow_pickle=True))
        tZ["source_rig_label"] = np.array(tRig)
        tZ["pose_src"] = np.array(poseSrc or "tag_only")
        np.savez(derivedTake / f"umi_traj_{tArm}.npz", **tZ)


IDLE_TRIM = {0: (0.5, 8.5), 1: (0.3, 8.0)}


def writeIdleTrim(path: Path, takes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as tFh:
        tW = csv.DictWriter(tFh, fieldnames=["take", "demo", "demo_len_s", "keep_start_s", "keep_end_s", "cut_head_s",
                                             "cut_tail_s", "kept_s", "pose_src"])
        tW.writeheader()
        for tTake in takes:
            for tDemo, (tA, tB) in IDLE_TRIM.items():
                tW.writerow(dict(take=tTake, demo=tDemo, demo_len_s=9.0, keep_start_s=tA, keep_end_s=tB, cut_head_s=tA,
                                 cut_tail_s=round(9.0 - tB, 1), kept_s=round(tB - tA, 1), pose_src="fused"))


# robotize_ik command-line defaults (the argparse Namespace the IK stage and the deploy receive)
ROBOTIZE_DEFAULTS = dict(demos="", workers=1, arm_gap=0.12, rounds=6, ramp=10, smooth=5, q_smooth=5, ik_step=7.5, ori_cost=0.1,
                         roll_thr=115.0,
                         roll_relax=1.5, table_z=0.003, tip_clear=0.005, tip_clear_left_extra=0.005, ik_mode="seq", reseed_n=5,
                         reseed_mm=50.0,
                         seeds=8, bridge_window_s=1.0, rejoin_window_s=2.0, jaw_half=0.020, level_max=0.0, force=False)


class FakeIk:
    """stands in for the Pink QP solver (no QP solver in the test environment): a few damped least-squares position steps on the
    arm's own pinocchio FK, each joint step clipped to the solver's max_joint_delta_deg. Deterministic."""

    def __init__(self, pk, stepDeg, iters):
        self.mPk, self.mStep, self.mIters = pk, np.radians(stepDeg), iters

    def ik(self, q, T):
        q = np.array(q, np.float64)
        for _ in range(self.mIters):
            tP = self.mPk.fk(q)[:3, 3]
            tE = T[:3, 3] - tP
            tJ = np.empty((3, 6))
            for tJoint in range(6):
                tDq = np.zeros(6)
                tDq[tJoint] = 1e-5
                tJ[:, tJoint] = (self.mPk.fk(q + tDq)[:3, 3] - tP) / 1e-5
            tD = tJ.T @ np.linalg.solve(tJ @ tJ.T + 1e-3 * np.eye(3), tE)
            q = q + np.clip(tD, -self.mStep, self.mStep)
        return q, float(np.linalg.norm(self.mPk.fk(q)[:3, 3] - T[:3, 3])), 0.0
