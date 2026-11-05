"""Prep stage of the LeRobot builders: robot arm / wrist camera assignment of a take's two UMI rigs, the arm-labelled derived copy
and the IK robotization run (dataset stage; was the prep part of tag_rig_to_lerobot.py).

The rig labels in umi_traj_<side>.npz come from the tag ids (R 200-202 / L 210-212) + their IMU. On 2026-10-22 these labels are
SWAPPED w.r.t. the robot (right-labelled rig at base +y = robot LEFT side) and w.r.t. the wrist cameras (cam 'wrist' moves with
rig 'left'). So the converter never trusts labels:
  robot arm  := sign of the rig's median base-frame y on tag-seen frames (+y = LEFT arm, reBot base convention)
  wrist cam  := the camera whose optical-flow magnitude correlates best with that rig's angular speed (2x2 test, vote over all
  demos)
  gripper    := umi_jaw.npz of that camera
10-22 user decision: the assignment is automatic for all future takes, so the two independent checks must agree; any disagreement
(ArmAssignmentError) = something changed physically -> stop, never guess.
prepTake writes <derived>/<take><pose_src>/assignment.json + umi_traj_<arm>.npz and runs the robotizer there: robotize_ik (ik_v1,
FINAL) through the IK interpreter, or the legacy umi_robotize_bi --sep 0.06 (never for training).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

import cv2
import numpy as np

from laundry_common import paths as egopaths

from .takes import demoWindows

EGO = egopaths.DATA_ROOT
EGO_PY = EGO / ".venv/bin/python"
IK_PY = egopaths.IK_PYTHON
ROBOTIZE_ARGS = ["--clear", "0", "--sep", "0.06", "--min-reach", "0"]   # legacy umi_robotize_bi settings
IK_WORKERS = 6
ROBOTIZE_MODULE = "laundry_dataset.robotize_ik"
SIDES = ("right", "left")
WRIST_CAMS = ("wrist", "wrist_left")
EXPECTED_CAM_OF_ARM = {"left": "wrist_left", "right": "wrist"}   # cameras are labelled by the hand that wears them
FLOW_SIZE = (160, 90)                # optical flow on a downscaled grey frame
FLOW_STEP = 3                        # every 3rd wrist frame
FLOW_MIN_SAMPLES = 50                # flow / angular-speed samples needed for one correlation
FARNEBACK = (None, 0.5, 2, 9, 2, 5, 1.1, 0)
N_PAIRINGS = len(SIDES) * len(WRIST_CAMS)   # full 2x2 rig x camera correlation of one demo
MIN_VOTES, MIN_AGREEMENT = 2, 0.8    # >= 2 demos with a full 2x2, >= 80 % voting the same way
MIN_DT_S = 1e-3
NS_PER_S = 1e9
STDERR_TAIL = 500
IK_LOG_TAIL = 2000


class ArmAssignmentError(RuntimeError):
    """The y-sign and the optical-flow pairing of a take disagree: something changed physically."""


def _angSpeed(z):
    tT = z["t_ns"].astype(np.int64)
    tR = z["R_B_tcp"]
    tDr = np.einsum("nji,njk->nik", tR[:-1], tR[1:])
    tAngle = np.arccos(np.clip((np.trace(tDr, axis1=1, axis2=2) - 1) / 2, -1, 1))
    return tT[1:], tAngle / np.maximum(np.diff(tT) / NS_PER_S, MIN_DT_S)


def _flowMag(camDir, tLo, tHi, step=2):
    """median optical-flow magnitude between consecutive used wrist frames in [tLo, tHi] -> (t_ns, magnitude)"""
    tT = np.load(camDir / "t_ns.npy")
    tSel = np.flatnonzero((tT >= tLo) & (tT <= tHi))[::step]
    tPrev, tOut, tUsed = None, [], []
    for tI in tSel:
        tImg = cv2.imread(str(camDir / f"wrist_{tI:05d}.jpg"))
        if tImg is None:
            continue   # 10-27: 152818 lost ~70 wrist jpgs (recorder drop) -> skip, flow is a median over the window
        tGrey = cv2.cvtColor(cv2.resize(tImg, FLOW_SIZE), cv2.COLOR_BGR2GRAY)
        if tPrev is None:
            tFlow = 0.0
        else:
            tFlow = float(np.median(np.linalg.norm(cv2.calcOpticalFlowFarneback(tPrev, tGrey, *FARNEBACK), axis=2)))
        tOut.append(tFlow)
        tPrev = tGrey
        tUsed.append(tI)
    return tT[tUsed][1:], np.array(tOut[1:])


def assignArms(take) -> dict:
    """-> dict(rig_of_arm, cam_of_arm, diagnostics) (was assign_arms)"""
    tZ = {tS: np.load(take / f"umi_traj_{tS}.npz") for tS in SIDES}
    tYmed = {tS: float(np.median(tZ[tS]["p_B"][tZ[tS]["ntags"] > 0, 1])) for tS in tZ}
    if np.sign(tYmed["right"]) == np.sign(tYmed["left"]):
        raise ValueError(f"rigs on the same side of the base (median y {tYmed})")
    tArmOfRig = {tS: ("left" if tYmed[tS] > 0 else "right") for tS in tZ}
    # per-demo 2x2 correlation of wrist optical-flow magnitude vs rig angular speed; decide by a vote over ALL demos
    # (124754: single 40 s windows were ambiguous, but crossed > straight in 10/10 demos)
    tAng = {tS: _angSpeed(tZ[tS]) for tS in tZ}
    tCorr = {}
    tVotes = []
    for tD in demoWindows(take):
        tC1 = {}
        for tC in WRIST_CAMS:
            tTc, tFlow = _flowMag(take / tC, tD.mT0Ns, tD.mT1Ns, step=FLOW_STEP)
            for tS in tZ:
                tTa, tA = tAng[tS]
                tM = (tTc >= tTa[0]) & (tTc <= tTa[-1])
                if tM.sum() > FLOW_MIN_SAMPLES:
                    tC1[(tS, tC)] = float(np.corrcoef(tFlow[tM], np.interp(tTc[tM], tTa, tA))[0, 1])
        if len(tC1) == N_PAIRINGS:
            tCrossed = tC1[("right", "wrist_left")] + tC1[("left", "wrist")]
            tStraight = tC1[("right", "wrist")] + tC1[("left", "wrist_left")]
            tVotes.append(tCrossed - tStraight)
            for tK, tV in tC1.items():
                tCorr.setdefault(tK, []).append(tV)
    tVotes = np.array(tVotes)
    tCorr = {tK: float(np.mean(tV)) for tK, tV in tCorr.items()}
    tAgree = max(np.mean(tVotes > 0), np.mean(tVotes < 0)) if len(tVotes) else 0.0
    if len(tVotes) < MIN_VOTES or tAgree < MIN_AGREEMENT:
        raise ValueError(f"camera<->rig pairing ambiguous: votes {np.round(tVotes, 2).tolist()}")
    tStraight, tCrossed = (0.0, 1.0) if np.mean(tVotes > 0) >= MIN_AGREEMENT else (1.0, 0.0)
    tCamOfRig = {"right": "wrist", "left": "wrist_left"} if tStraight > tCrossed else {"right": "wrist_left", "left": "wrist"}
    tOut = dict(rig_of_arm={tArmOfRig[tS]: tS for tS in tZ}, cam_of_arm={tArmOfRig[tS]: tCamOfRig[tS] for tS in tZ},
                rig_median_y_m=tYmed, flow_corr={f"{tS}|{tC}": round(tV, 3) for (tS, tC), tV in tCorr.items()},
                rig_labels_swapped=bool(tArmOfRig["right"] == "left"), cam_labels_swapped=bool(tCamOfRig["right"] != "wrist"),
                pairing_votes_crossed_minus_straight=np.round(tVotes, 3).tolist(), pairing_agreement=float(tAgree))
    if {*tOut["rig_of_arm"]} != {"left", "right"}:
        raise ValueError(f"bad assignment {tOut}")
    # y-sign (rig -> arm) and optical flow (rig -> camera) must put the 'wrist_left' camera on the robot LEFT arm
    if tOut["cam_of_arm"] != EXPECTED_CAM_OF_ARM:
        raise ArmAssignmentError(f"[{take.name}] ASSIGNMENT CONFLICT: y-sign says rig_of_arm={tOut['rig_of_arm']} but optical "
                                 f"flow pairs cam_of_arm={tOut['cam_of_arm']} (corr {tOut['flow_corr']}); check the rigs/cameras "
                                 "before converting")
    return tOut


def srcEnv() -> dict:
    """environment of the IK sub-process: src/ on PYTHONPATH (robotize_ik imports laundry_shared / laundry_common from this
    repo)"""
    tPath = os.pathsep.join(tP for tP in (str(egopaths.SRC_ROOT), os.environ.get("PYTHONPATH", "")) if tP)
    return {**os.environ, "PYTHONPATH": tPath}


def prepTake(take, derivedRoot, poseSrc: str = "", robotize: str = "ik_v1", force: bool = False) -> dict:
    """arm assignment + arm-labelled copy (once) and the robotizer run (every call; robotize_ik skips demos already done) -> the
    assignment dict + robotize mode (was prep_take; poseSrc "" = tag-only umi_traj_<rig>.npz, "_fused" = MASt3R + AprilTag
    fusion)"""
    tD = derivedRoot / (take.name + poseSrc)
    tAsgFile = tD / "assignment.json"
    if not (tAsgFile.exists() and not force and all((tD / f"umi_traj_{tA}.npz").exists() for tA in ("left", "right"))):
        tD.mkdir(parents=True, exist_ok=True)
        tAsg = assignArms(take)
        for tArm, tRig in tAsg["rig_of_arm"].items():
            tZ = dict(np.load(take / f"umi_traj_{tRig}{poseSrc}.npz", allow_pickle=True))
            tZ["source_rig_label"] = np.array(tRig)
            tZ["pose_src"] = np.array(poseSrc or "tag_only")
            np.savez(tD / f"umi_traj_{tArm}.npz", **tZ)
        shutil.copy(take / "segments.json", tD / "segments.json")
        tAsg.update(take=str(take), pose_src=poseSrc or "tag_only", created=time.strftime("%Y-%m-%d %H:%M:%S"))
        tAsgFile.write_text(json.dumps(tAsg, indent=1, ensure_ascii=False))
    tAsg = json.loads(tAsgFile.read_text())
    # legacy, never for training
    if robotize == "sep" and not all((tD / f"umi_traj_{tA}_robot.npz").exists() for tA in ("left", "right")):
        tR = subprocess.run([str(EGO_PY), str(EGO / "scripts/umi_robotize_bi.py"), str(tD), *ROBOTIZE_ARGS], capture_output=True,
                            text=True)
        if tR.returncode:
            raise RuntimeError(f"robotize failed: {tR.stderr[-STDERR_TAIL:]}")
        (tD / "robotize.log").write_text(tR.stdout)
    if robotize == "ik_v1":                                  # robotize_ik per demo (skips demos already done)
        tR = subprocess.run([str(IK_PY), "-m", ROBOTIZE_MODULE, str(take), str(tD), "--workers", str(IK_WORKERS)],
                            capture_output=True, text=True,
                            env=srcEnv())
        with (tD / "robotize_ik.log").open("a") as tLog:
            tLog.write(tR.stdout + tR.stderr[-IK_LOG_TAIL:])
        if tR.returncode:
            raise RuntimeError(f"robotize_ik failed: {tR.stderr[-STDERR_TAIL:]}")
    tAsg["robotize"] = robotize
    return tAsg
