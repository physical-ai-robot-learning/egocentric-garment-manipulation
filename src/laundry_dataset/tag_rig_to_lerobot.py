#!/usr/bin/env python3
"""[2026-10-22] v1 builder: AprilTag-rig UMI laundry-folding takes (ego_collector tag_rig_record --proto task --wrist) ->
LeRobotDataset in the Agent A contract, same layout as datasets/umi_to_lerobot.py (20 Hz grid, absolute next-frame action targets,
contract proprio, 3 cameras), shared by the Multi-Task DiT (training/train_dit.py) and ACT (training/train_act.py) launchers
(dataset stage). The current builder is tag_rig_unsplit; this one is kept for the v1 sets and owns the source META both write.

Inputs per take (READ-ONLY; nothing under ego_collector/datasets is written):
  meta.json                 t_start_monotonic_ns + protocol; every entry whose text starts with '자유:' is one demo = one episode
  t_ns.npy, color_%05d.jpg  fixed Orbbec RGB (~15 Hz, 1280x720)                                   -> observation.images.global
  wrist/, wrist_left/       UMI fisheye wrist cams (~30 Hz) + umi_jaw.npz (g_cont 0 closed .. 1 open, valid, ev_t)
  umi_traj_{right,left}.npz scripts/umi_tcp_traj.py: jaw-tip TCP p_B, rig rotation R_B_tcp (robot base frame B), ntags

ARM ASSIGNMENT (prep stage, per take; arm_assignment.prepTake -> <derived>/<take>/assignment.json): the converter never trusts the
rig labels; see arm_assignment. orientation := R_B_E = R_B_tcp @ R_U_E(rig label) (calibration.GRIPPER_ALIGN, as umi_replay_bi
does).
  The arm-labelled trajectories are copied to <derived>/<take>/umi_traj_<arm>.npz and robotized THERE (robotize_ik, ik_v1).
  (Running the legacy robotizer on the raw labels pushes each hand ~23 cm across the midline on these takes -> those _robot.npz
  are invalid.)

Per 20 Hz frame (grid inside the demo window clipped to the recorded data):
  observation.images.{global,left_wrist,right_wrist}  nearest frame (<= 50 ms global / 40 ms wrist), all 384x216 (16:9, ONE shape:
  DiT)
  observation.state  contract proprio 30D (episode-anchor poses, grippers, T_LR valid = 1: both rigs are in the same base frame B)
  action             ABSOLUTE next-frame target [L pose9 | R pose9 | gL | gR]; training builds inv(T(t)) @ T(t+k)
  pose_abs           current-frame absolute [L pose9 | R pose9 | gL | gR]
  quality            [tag_seen_L, tag_seen_R, robotize_L, robotize_R, jaw_valid_L, jaw_valid_R, jaw_event_L, jaw_event_R]
  weight             suggested per-frame loss weight: 1.0 good | 0.5 medium (robotize 1 or a rig without tags) |
                     0.1 bad (robotize 2, or jaw keypoints not detected within +-0.5 s of a jaw open/close event). NOT used by the
                     stock DiT / ACT losses yet; stored so a weighted loss / frame filter can be added without re-conversion.
  src_index          [global, left_wrist, right_wrist] source frame indices; img_residual_ms likewise (provenance / QA)
Frames inside a no-tag run > 1 s (either arm) are MASKED (episode split there, like any invalid run).
Frames whose pose bracket > 150 ms, jaw sample > 60 ms away or image residual above tolerance are invalid; an episode is split at
invalid runs (segments < --min-seconds dropped; same rule as umi_to_lerobot).
Split: by EPISODE (take/demo key), seeded shuffle, 80/10/10 over ALL candidate demos -> <out>_{train,val,test}; meta/split.json in
each root + episode_table.csv. --clean-min-tag 0.6 keeps only episodes whose per-arm tag-seen fraction is >= 0.6 (split
unchanged).
usage: tag_rig_to_lerobot.py --out data/lerobot/tagrig_laundry_v1 --takes TAKE_DIR ... [--clean-min-tag 0.6] [--max-demos N]
    python -m laundry_dataset.tag_rig_to_lerobot --out ... --takes ...
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import pathlib
import shutil
import sys
import time

import numpy as np

from laundry_common import paths as egopaths
from laundry_shared.contracts.action_contract import ACTION_DT_S, T_to_pose9, make_T
from laundry_shared.umi_to_lerobot import proprio_rows_lr
from laundry_shared.xvla_core import abs_action_rows, nearest_frame, resample_pose, valid_segments

from . import calibration
from .arm_assignment import ROBOTIZE_ARGS, ArmAssignmentError, prepTake
from .export_common import (CAMS, IMG_WH, MAX_JAW_GAP_S, MAX_SRC_GAP_S, TASK, baseFeatures, createDataset, loadImage, rampedJaw,
                            splitAssign,
                            tailFeatures)
from .takes import demoKey, demoWindows

MIN_SECONDS = 5.0                 # QA 5: demo window < 5 s -> drop
MIN_SEG_S = 2.0                   # 10-22: masked pieces >= 2 s (40 frames > the 32-step chunk) are kept as LeRobot episodes
IMG_TOL_S = {"global": 0.05, "left_wrist": 0.04, "right_wrist": 0.04}
JAW_EVENT_WIN_S = 0.5
TAG_GAP_MASK_S = 1.0              # QA 1: a no-tag run > 1 s -> those frames weight 0.1
# QA 1 (user 10-22: 0.3, was 0.6): drop the episode when < 30 % of its active frames are valid and outside long tag gaps
MIN_GOOD_FRAC = 0.3
FLICKER_TOGGLES_PER_S = 2         # QA 3: > 2 jaw state toggles inside any 1 s window -> flag
# QA 6: idle head/tail trim (fixed 45 s demos end ~20 s idle with the UMIs on the table)
IDLE_SPEED_MPS, IDLE_MARGIN_S = 0.02, 1.0
IDLE_SMOOTH_FRAMES = 20           # 1 s speed smoothing of the idle trim
# 10-22 reviewer rule (fused pose quality, per episode); jump = neighbour-inconsistent step, contact = +-0.5 s around grasp
# (close) events: TRAIN (weight 1.0) / DOWNWEIGHT (0.3) / EXCLUDE
JUMP_MM, JUMP_DEG, JUMP_BIG_MM = 30.0, 15.0, 100.0          # 20 Hz frame-to-frame after a 3-frame position smoothing
# 10-22 auto-decision: grasp-instant robotize displacement median 136 mm (midline y-compression, --sep 0.06); 60 mm excl would
# drop 69/75
CONTACT_DOWN_MM, CONTACT_EXCL_MM = 60.0, 150.0
JUMPS_EXCL = 3                                              # >= 3 jumps (or any > 100 mm) -> exclude; 1-2 -> downweight
COVERAGE_WARN, GAP_WARN_S = 0.95, 2.0
DOWN_W = 0.3
PAIR_MARGIN = 0.15
JUMP_NEIGHBOURS = (-2, -1, 1, 2)  # a jump = a step inconsistent with the mean of these neighbour steps
JUMP_EDGE_FRAMES = 3              # no jump test on the first / last 3 frames
STITCH_WIN_FRAMES = 5             # jump "near a stitch" = within +-2 frames of a fusion segment change
GRASP_HALF_FRAMES = 10            # contact window = +-0.5 s around each jaw close
JAW_OPEN = 0.5
WEIGHT_BAD, WEIGHT_MEDIUM, WEIGHT_GOOD = 0.1, 0.5, 1.0
ROBOTIZE_MEDIUM = 1               # robotize class 1 (20-50 mm) = medium
ROBOTIZE_BAD = 2                  # robotize class 2 (>= 50 mm) = bad
GAP_MARGIN_M = 0.005              # link-gap mask below ARM_GAP_M minus this
QUALITY_COLUMNS = ("tag_L", "tag_R", "rob_L", "rob_R", "jv_L", "jv_R", "ev_L", "ev_R")   # build columns behind QUALITY_NAMES
QUALITY_NAMES = ["tag_seen_L", "tag_seen_R", "robotize_L", "robotize_R", "jaw_valid_L", "jaw_valid_R", "jaw_event_L",
                 "jaw_event_R"]
# grasp-window gate (robotized vs fused TCP, max per window): <20 ok, 20-50 weight 0.5, >50 mask
GRASP_OK_MM, GRASP_WARN_MM, GRASP_W = 20.0, 50.0, 0.5
IK_FAIL_MM, ARM_GAP_M = 10.0, 0.12
# fused files: if one of these per-sample masks exists it replaces ntags > 0 for gap masking
GAP_KEYS = ("fused_valid", "valid", "ok")
ARMS = (("left", "L"), ("right", "R"))
NS_PER_S = 1e9
MS_PER_NS = 1e6
MM_PER_M = 1000
PCT = 100
FLAG_ON = 0.5                     # 0/1 float flag columns (tag seen, jaw valid, jaw event): set
MIN_WINDOW_S = 1e-6              # guards the active-fraction division of an empty window


def sourceMeta(task: str = TASK) -> dict:
    """source_meta.json content of both builders (was the module dict META; `instruction` = the task string)"""
    return dict(data_source="UMI", source_robot="HANDUMI_TAGRIG_X2",
                skill="LAUNDRY_FOLDING (full task; not a contracts.skill_contract.Skill yet)",
                instruction=task,
                pose_source="AprilTag rig (Orbbec) + IMU rotation fill (scripts/umi_tcp_traj.py) -> umi_robotize_bi.py "
                            f"{' '.join(ROBOTIZE_ARGS)} (robot_feasible_tcp; p_B_human kept in the source only)",
                pose_frame="reBot base B (configs/calibration/robot_cam_v001.json), both arms; T_LR valid",
                tcp_point="UMI jaw-tip touch TCP (umi_tag_rig_v001 tcp_in_rig_m)",
                tcp_axes="reBot gripper EEF axes E: R_B_E = R_B_tcp(rig) @ R_U_E (umi_gripper_align[_left]_v001), i.e. the "
                         "orientation umi_replay_bi commands; NOT the camera-aligned dataset-TCP axes of the older UMIalign sets",
                T_LR_valid=True,
                gripper="action/state gripper = BINARY umi_jaw `state` (1 = open, 0 = closed) with a 0.1 s linear ramp at "
                        "transitions; continuous g_cont stored in gripper_cont only. Deploy: > 0.5 open, else full close",
                images=f"global = Orbbec RGB, wrists = raw UMI fisheye; all resized to {IMG_WH[0]}x{IMG_WH[1]} "
                       "(16:9, no stretch)",
                qa="automatic per episode (episode_table.csv qa_decision / qa_flags / reject_reason): tag gap > 1 s -> weight "
                   "0.1; < 60 % good frames -> drop; < 5 s -> drop; idle head/tail (both arms < 2 cm/s, 1 s smoothing, 1 s "
                   "margin) trimmed; no open->close event / flicker > 2 toggles per s -> flag; left arm median y must be > 0 "
                   "(else LR_SWAP flag). IK / joint-speed / workspace-limit feasibility is NOT checked here (umi_replay_bi "
                   "--offline does that); robotize class 2 (>= 50 mm) -> weight 0.1",
                episode_rule=dict(jump=f"frame-to-frame > {JUMP_MM:.0f} mm or > {JUMP_DEG:.0f} deg at 20 Hz (3-frame position "
                                       "smoothing), active frames",
                                  exclude=f">= {JUMPS_EXCL} jumps, any jump > {JUMP_BIG_MM:.0f} mm, or contact (gripper closed) "
                                          f"robotize displacement >= {CONTACT_EXCL_MM:.0f} mm",
                                  downweight=f"1-2 jumps or contact robotize displacement >= {CONTACT_DOWN_MM:.0f} mm -> weight "
                                             f"{DOWN_W} (used in the loss)",
                                  warn=f"fused coverage < {COVERAGE_WARN * PCT:.0f} %, longest gap >= {GAP_WARN_S} s"),
                weight_column="`weight` = EPISODE weight (1.0 / 0.3) multiplied into the per-sample loss "
                              "(training/tagrig_common.py); `weight_frame` = per-frame suggestion (not used)",
                rate_hz=1.0 / ACTION_DT_S,
                weight_rule="1.0 good | 0.5 robotize 1 or rig without tags | 0.1 robotize 2, tag gap > 1 s, or jaw undetected "
                            "within 0.5 s of a jaw event")


def contractVersion() -> str:
    return importlib.import_module("laundry_shared.contracts").CONTRACT_VERSION


def _runs(mask):
    tX = np.r_[0, mask.astype(int), 0]
    return np.flatnonzero(np.diff(tX) == 1), np.flatnonzero(np.diff(tX) == -1)


def _relRotDeg(R):
    tDr = np.einsum("nji,njk->nik", R[:-1], R[1:])
    return np.degrees(np.arccos(np.clip((np.trace(tDr, axis1=1, axis2=2) - 1) / 2, -1, 1)))


# ---------------------------------------------------------------- per-demo grid
def _loadInputs(take, dtake, asg, demo, robotize):
    tCam = np.load(take / "t_ns.npy").astype(np.int64)
    tTrajs = {tA: np.load(dtake / (f"umi_traj_{tA}.npz" if robotize == "ik_v1" else f"umi_traj_{tA}_robot.npz"),
                          allow_pickle=True)
              for tA in ("left", "right")}
    tIk = None
    if robotize == "ik_v1":
        tFile = dtake / "ik_v1" / f"demo{demo.mIdx:02d}.npz"
        if not tFile.exists():
            raise FileNotFoundError(f"ik_v1 robotization missing ({tFile.name})")
        tIk = np.load(tFile)
    tJawFiles = {tA: (take / asg["cam_of_arm"][tA] / "umi_jaw.npz") for tA in ("left", "right")}
    for tA, tP in tJawFiles.items():
        if not tP.exists():
            raise FileNotFoundError(f"missing {tP.parent.name}/umi_jaw.npz ({tA} arm)")
    tJaws = {tA: np.load(tP) for tA, tP in tJawFiles.items()}
    for tA, tJ in tJaws.items():
        if "g_is_aperture" not in tJ.files or int(tJ["g_is_aperture"]) != 1:
            raise ValueError(f"{tA} arm jaw npz has g_is_aperture != 1 (not an aperture label)")
    tCams = {"global": tCam, "left_wrist": np.load(take / asg["cam_of_arm"]["left"] / "t_ns.npy").astype(np.int64),
             "right_wrist": np.load(take / asg["cam_of_arm"]["right"] / "t_ns.npy").astype(np.int64)}
    return tCam, tTrajs, tIk, tJaws, tCams


def _arm(out, a, s, z, J, IK, tg, lo, hi):
    """per-arm columns of the grid (pose, jaw, tracked flag, coverage, robotize displacement / class) into `out`"""
    tTs = (z["t_ns"].astype(np.int64) - lo) / NS_PER_S
    tK, _ = nearest_frame(tTs, tg)
    # robot-feasible label = FK(q) of the ik_v1 joint path (already reBot EEF axes)
    if IK is not None:
        tTi = (IK["t_ns"].astype(np.int64) - lo) / NS_PER_S
        tT, _, tV, _ = resample_pose(tTi, make_T(IK[f"{a}_R"], IK[f"{a}_p"]), np.zeros(len(tTi)), tg, max_gap=MAX_SRC_GAP_S)
        tKi, _ = nearest_frame(tTi, tg)
        tPh = np.stack([np.interp(tg, tTi, IK[f"{a}_p_h"][:, tJ]) for tJ in range(3)], 1)
        out[f"Ph_{a}"] = tPh
        out[f"ikmm_{s}"] = IK[f"{a}_ik_mm"][tKi]
        out[f"lim_{s}"] = IK[f"{a}_lim"][tKi]
        out[f"grasp_{s}"] = IK[f"{a}_grasp"][tKi]
        out["gap_m"] = IK["gap"][tKi]
    else:
        tTSrc = make_T(z["R_B_tcp"] @ calibration.rigToEef(str(z["source_rig_label"])), z["p_B"])
        tT, _, tV, _ = resample_pose(tTs, tTSrc, np.zeros(len(tTs)), tg, max_gap=MAX_SRC_GAP_S)
    tTj = (J["t_ns"].astype(np.int64) - lo) / NS_PER_S
    tGc = np.clip(np.interp(tg, tTj, J["g_cont"].astype(float)), 0, 1)
    tKj, tRj = nearest_frame(tTj, tg)
    # box filter of width 0.1 s = linear ramp
    tG = rampedJaw(J, tTj, tg, nearest_frame)
    tState = J["state"][tKj].astype(int)
    tWin = int(round(1.0 / ACTION_DT_S))
    tToggles = np.r_[0, np.abs(np.diff(tState))]
    tToggles1s = np.convolve(tToggles, np.ones(tWin), "same") if len(tToggles) >= tWin else tToggles
    out[f"close_events_{s}"] = int(np.sum(np.diff(tState) < 0))
    out[f"flicker_{s}"] = bool(np.max(tToggles1s, initial=0) > FLICKER_TOGGLES_PER_S)
    out[f"gc_{a}"] = tGc
    tEv = (np.asarray(J["ev_t"], np.int64) - lo) / NS_PER_S if "ev_t" in J.files else np.zeros(0)
    tNearEv = np.zeros(len(tg), bool)
    for tE in tEv:
        tNearEv |= np.abs(tg - tE) <= JAW_EVENT_WIN_S
    out[f"T_{a}"] = tT
    out[f"g_{a}"] = tG
    out[f"ymed_{s}"] = float(np.median(tT[:, 1, 3]))
    # "tracked" flag: fused validity mask if present, else tag seen
    tGapKey = next((tKey for tKey in GAP_KEYS if tKey in z.files), None)
    if tGapKey is None and "slam_ok" in z.files:
        tTracked = z["slam_ok"].astype(bool) | (z["ntags"] > 0)              # pre-final fused files
    else:
        tTracked = (z[tGapKey] > 0) if tGapKey else (z["ntags"] > 0)
    out[f"tag_{s}"] = tTracked[tK].astype(np.float32)
    tIn = (z["t_ns"] >= lo) & (z["t_ns"] <= hi)
    tVv = tTracked[tIn]
    tTt = z["t_ns"][tIn] / NS_PER_S
    tG0, tG1 = _runs(~tVv)
    out[f"cov_{s}"] = float(tVv.mean()) if len(tVv) else 0.0
    out[f"maxgap_{s}"] = float(max([tTt[min(tE, len(tTt) - 1)] - tTt[max(tB - 1, 0)] for tB, tE in zip(tG0, tG1)], default=0.0))
    if IK is not None:
        out[f"disp_{s}"] = np.linalg.norm(tT[:, :3, 3] - out[f"Ph_{a}"], axis=1) * MM_PER_M
    else:
        out[f"disp_{s}"] = np.linalg.norm(z["p_B"] - z["p_B_human"],
                                          axis=1)[tK] * MM_PER_M if "p_B_human" in z.files else np.zeros(len(tK))
    tSeg = z["seg_src"][tK] if "seg_src" in z.files else np.zeros(len(tK), int)
    out[f"segchg_{s}"] = np.r_[False, tSeg[1:] != tSeg[:-1]]
    if IK is not None:
        out[f"rob_{s}"] = np.digitize(out[f"disp_{s}"], [GRASP_OK_MM, GRASP_WARN_MM]).astype(np.float32)
    else:
        out[f"rob_{s}"] = z["robotize_quality"][tK].astype(np.float32)
    out[f"jv_{s}"] = J["valid"][tKj].astype(np.float32)
    out[f"ev_{s}"] = tNearEv.astype(np.float32)
    out["valid"] &= tV & (tRj <= MAX_JAW_GAP_S)


def _jumps(out, a, s, act):
    """neighbour-inconsistent steps, stitch jumps, robotize displacement stats and the grasp-instant (contact) maximum into
    `out`"""
    tP = out[f"Ph_{a}"] if f"Ph_{a}" in out else out[f"T_{a}"][:, :3, 3]
    tPs = np.copy(tP)                 # jumps: on the FUSED (human) path; Ps[1:-1] = (P[:-2] + P[1:-1] + P[2:]) / 3
    tDr = np.r_[0, _relRotDeg(out[f"T_{a}"][:, :3, :3])]
    # 10-22: an ABSOLUTE 30 mm/step (0.6 m/s) is the p95 of normal folding motion (tag-only and fused alike, max 77 mm, smooth) ->
    # a jump is a step INCONSISTENT with its neighbours: |dp(t) - mean(dp(t-2..t+2) without t)| > 30 mm, or a rotation step > 15
    # deg
    tV = np.diff(tPs, axis=0, prepend=tPs[:1])
    tVn = np.zeros_like(tV)
    for tO in JUMP_NEIGHBOURS:
        tVn += np.roll(tV, tO, axis=0)
    tDev = np.linalg.norm(tV - tVn / len(JUMP_NEIGHBOURS), axis=1) * MM_PER_M
    tDev[:JUMP_EDGE_FRAMES] = 0
    tDev[-JUMP_EDGE_FRAMES:] = 0
    tJump = act & ((tDev > JUMP_MM) | (tDr > JUMP_DEG))
    tNearStitch = np.convolve(out[f"segchg_{s}"].astype(int), np.ones(STITCH_WIN_FRAMES), "same") > 0
    out[f"jumps_{s}"] = int(tJump.sum())
    out[f"stitch_jumps_{s}"] = int((tJump & tNearStitch).sum())
    out[f"jump_max_mm_{s}"] = float(tDev[act].max()) if act.any() else 0.0
    # contact = the GRASP instant (+-0.5 s around open->close events): the jaw is closed ~90 % of the time, so "closed" != contact
    tOpen = (out[f"g_{a}"] > JAW_OPEN).astype(int)
    tContact = np.zeros(len(tOpen), bool)
    for tC in np.flatnonzero(np.diff(tOpen) < 0) + 1:
        tContact[max(tC - GRASP_HALF_FRAMES, 0): tC + GRASP_HALF_FRAMES + 1] = True
    tContact &= act
    tD = out[f"disp_{s}"]
    out[f"disp_mean_{s}"] = float(tD[act].mean()) if act.any() else 0.0
    out[f"disp_max_{s}"] = float(tD[act].max()) if act.any() else 0.0
    out[f"disp_contact_max_{s}"] = float(tD[tContact].max()) if tContact.any() else 0.0


def _graspGate(out):
    """ik_v1: per grasp window max |robotized - fused| -> mask (> 50 mm) / weight 0.5 (20-50 mm); IK and link-gap masks"""
    for tA, tS in ARMS:
        for tG0, tG1 in zip(*_runs(out[f"grasp_{tS}"].astype(bool))):
            if not out["valid"][tG0:tG1].any():
                continue                                                  # grasp inside masked / idle frames: not counted
            tMax = float(out[f"disp_{tS}"][tG0:tG1].max())
            out["grasp_total"] += 1
            out["grasp_max_mm"].append(round(tMax))
            if tMax > GRASP_WARN_MM:
                out["grasp_fail"] += 1
                out["valid"][tG0:tG1] = False                                   # mask the grasp segment (+-0.5 s)
            elif tMax >= GRASP_OK_MM:
                out["grasp_warn"] += 1
                out["grasp_w"][tG0:tG1] = np.minimum(out["grasp_w"][tG0:tG1], GRASP_W)
        out[f"ikfail_{tS}"] = out[f"ikmm_{tS}"] > IK_FAIL_MM
        out["valid"] &= ~out[f"ikfail_{tS}"]                                   # free space: mask only IK / limit failures
    out["gapfail"] = out["gap_m"] < ARM_GAP_M - GAP_MARGIN_M
    out["valid"] &= ~out["gapfail"]


def buildDemo(take, dtake, asg, demo, robotize: str = "ik_v1", tagGapS: float = TAG_GAP_MASK_S) -> dict:
    """one demo on the 20 Hz grid with every QA column (was build_demo; robotize / tagGapS were module globals)"""
    tCam, tTrajs, tIK, tJaws, tCams = _loadInputs(take, dtake, asg, demo, robotize)
    tLo = max(demo.mT0Ns, *(int(tTrajs[tA]["t_ns"][0]) for tA in tTrajs), tCam[0])
    tHi = min(demo.mT1Ns, *(int(tTrajs[tA]["t_ns"][-1]) for tA in tTrajs), tCam[-1])
    if tHi - tLo < MIN_SECONDS * NS_PER_S:
        raise ValueError(f"demo window inside the recording only {(tHi - tLo) / NS_PER_S:.1f} s (take truncated)")
    tGrid = tLo + (np.arange(int((tHi - tLo) / NS_PER_S / ACTION_DT_S) + 1) * ACTION_DT_S * NS_PER_S).astype(np.int64)
    tTg = (tGrid - tLo) / NS_PER_S
    tOut = dict(t=tTg, t_ns=tGrid, valid=np.ones(len(tGrid), bool), window_s=(tHi - tLo) / NS_PER_S)
    for tA, tS in ARMS:
        _arm(tOut, tA, tS, tTrajs[tA], tJaws[tA], tIK, tTg, tLo, tHi)
    for tC in CAMS:
        tIdx, tRes = nearest_frame(tCams[tC].astype(np.float64), tGrid.astype(np.float64))
        tOut[f"img_{tC}"] = tIdx
        tOut[f"res_{tC}"] = tRes / MS_PER_NS
        tOut["valid"] &= tRes <= IMG_TOL_S[tC] * NS_PER_S
    tLongGap = np.zeros(len(tTg), bool)
    for tS in ("L", "R"):
        for tP0, tP1 in zip(*_runs(tOut[f"tag_{tS}"] < FLAG_ON)):
            if (tP1 - tP0) * ACTION_DT_S > tagGapS:
                tLongGap[tP0:tP1] = True
    tOut["long_gap"] = tLongGap
    tSpeed = np.zeros(len(tTg))
    for tA in ("left", "right"):
        tV = np.r_[0, np.linalg.norm(np.diff(tOut[f"T_{tA}"][:, :3, 3], axis=0), axis=1)] / ACTION_DT_S
        tSpeed = np.maximum(tSpeed, np.convolve(tV, np.ones(IDLE_SMOOTH_FRAMES) / IDLE_SMOOTH_FRAMES, "same"))
    tActive = np.flatnonzero(tSpeed > IDLE_SPEED_MPS)
    tM = int(IDLE_MARGIN_S / ACTION_DT_S)
    tKeep = np.zeros(len(tTg), bool)
    if len(tActive):
        tKeep[max(tActive[0] - tM, 0): tActive[-1] + tM + 1] = True
    tOut["idle_head_s"] = float(np.argmax(tKeep) * ACTION_DT_S) if tKeep.any() else tOut["window_s"]
    tOut["idle_tail_s"] = float((len(tKeep) - 1 - np.flatnonzero(tKeep)[-1]) * ACTION_DT_S) if tKeep.any() else 0.0
    tOut["valid_pre_mask"] = tOut["valid"].copy()
    tOut["valid"] &= ~tLongGap & tKeep      # QA 1: MASK (episode split there) -- stock losses ignore `weight`
    tAct = tOut["valid"]
    for tA, tS in ARMS:
        _jumps(tOut, tA, tS, tAct)
    tOut["grasp_w"] = np.ones(len(tTg), np.float32)
    tOut["grasp_total"] = tOut["grasp_fail"] = tOut["grasp_warn"] = 0
    tOut["grasp_max_mm"] = []
    if tIK is not None:
        _graspGate(tOut)
    tRob = np.maximum(tOut["rob_L"], tOut["rob_R"])
    # jaw keypoints not detected within +-0.5 s of a jaw event
    tJawMissed = ((tOut["jv_L"] < FLAG_ON) & (tOut["ev_L"] > 0)) | ((tOut["jv_R"] < FLAG_ON) & (tOut["ev_R"] > 0))
    tBad = (tRob >= ROBOTIZE_BAD) | tLongGap | tJawMissed
    tMed = (tRob >= ROBOTIZE_MEDIUM) | (tOut["tag_L"] < FLAG_ON) | (tOut["tag_R"] < FLAG_ON)
    tOut["weight"] = np.where(tBad, WEIGHT_BAD, np.where(tMed, WEIGHT_MEDIUM, WEIGHT_GOOD)).astype(np.float32)
    tOut["cam_dir"] = {"global": take, "left_wrist": take / asg["cam_of_arm"]["left"],
                       "right_wrist": take / asg["cam_of_arm"]["right"]}
    return tOut


# ---------------------------------------------------------------- CLI
def buildParser() -> argparse.ArgumentParser:
    tAp = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tAp.add_argument("--out", required=True)
    tAp.add_argument("--takes", nargs="+", required=True)
    tAp.add_argument("--derived", default=str(egopaths.DERIVED), help="arm-labelled + robotized trajectories (prep stage)")
    tAp.add_argument("--seed", type=int, default=0)
    tAp.add_argument("--clean-min-tag", type=float, default=0.0)
    tAp.add_argument("--max-demos", type=int, default=0, help="per take, for quick tests")
    tAp.add_argument("--overwrite", action="store_true")
    tAp.add_argument("--prep-only", action="store_true")
    tAp.add_argument("--force-prep", action="store_true")
    tAp.add_argument("--task", default=TASK, help="task / instruction string stored per frame (DiT text conditioning)")
    tAp.add_argument("--jaw-newer-than", default="",
                     help="YYYY-mm-dd HH:MM: reject episodes whose umi_jaw.npz is older (stale labels)")
    tAp.add_argument("--qa-only", default="", help="write only the pass-1 QA table to this CSV and exit")
    tAp.add_argument("--pose-src", default="", choices=["", "_fused"],
                     help="'' = tag-only umi_traj_<rig>.npz (smoke/audit); _fused = MASt3R+tag fusion (FINAL)")
    tAp.add_argument("--robotize", default="ik_v1", choices=["ik_v1", "sep"])
    tAp.add_argument("--split-file", default="",
                     help="frozen split json (meta/split.json of an earlier build) -> identical episode lists")
    tAp.add_argument("--tag-gap-s", type=float, default=TAG_GAP_MASK_S)
    tAp.add_argument("--min-good-frac", type=float, default=MIN_GOOD_FRAC)
    return tAp


def _pct(x, w):
    return round(float(np.mean(x[w])) * PCT, 1) if w.any() else 0.0


def _passOne(t, d, asgs, derived, a, jawMin, built) -> dict:
    """pass 1 of one candidate demo: grid + QA (no frames written) -> its episode-table row; eligible demos go into `built`"""
    tKey = demoKey(t.name, d.mIdx)
    tRow = dict(episode_key=tKey, take=t.name, demo=d.mIdx, split="", ended_early=d.mEndedEarly, status="", reject_reason="")
    try:
        if t.name not in asgs:
            raise ValueError("take prep failed")
        tAsg = asgs[t.name]
        tRow.update(rig_of_left_arm=tAsg["rig_of_arm"]["left"], cam_of_left_arm=tAsg["cam_of_arm"]["left"])
        for tCamDir in set(tAsg["cam_of_arm"].values()):
            tJawFile = t / tCamDir / "umi_jaw.npz"
            if tJawFile.exists() and tJawFile.stat().st_mtime < jawMin:
                raise ValueError(f"stale {tCamDir}/umi_jaw.npz (older than --jaw-newer-than)")
        tB = buildDemo(t, derived / (t.name + a.pose_src), tAsg, d, a.robotize, a.tag_gap_s)
        tSegs = valid_segments(tB["valid"], int(MIN_SEG_S / ACTION_DT_S))
        tW = tB["valid_pre_mask"]                    # every percentage below is over the frames valid before the gap / idle masks
        tRow.update(duration_s=round(tB["window_s"], 2), grid_frames=len(tB["t"]), valid_frames=int(tW.sum()),
                    n_segments=len(tSegs),
                    frames_written=int(sum(tY - tX for tX, tY in tSegs)),
                    tag_seen_L_pct=_pct(tB["tag_L"], tW), tag_seen_R_pct=_pct(tB["tag_R"], tW),
                    jaw_valid_L_pct=_pct(tB["jv_L"], tW),
                    jaw_valid_R_pct=_pct(tB["jv_R"], tW), robotize2_L_pct=_pct(tB["rob_L"] >= ROBOTIZE_BAD, tW),
                    robotize2_R_pct=_pct(tB["rob_R"] >= ROBOTIZE_BAD, tW),
                    robotize1_L_pct=_pct(tB["rob_L"] == ROBOTIZE_MEDIUM, tW),
                    robotize1_R_pct=_pct(tB["rob_R"] == ROBOTIZE_MEDIUM, tW), open_frac_L=_pct(tB["g_left"] > JAW_OPEN, tW),
                    open_frac_R=_pct(tB["g_right"] > JAW_OPEN, tW),
                    mean_weight=round(float(tB["weight"][tW].mean()), 3) if tW.any() else 0.0,
                    img_res_max_ms=round(float(max(tB[f"res_{tC}"][tW].max() for tC in CAMS)), 1) if tW.any() else None)
        tGood = tB["valid"]
        tFlags = []
        tRow.update(idle_head_trim_s=round(tB["idle_head_s"], 1), idle_tail_trim_s=round(tB["idle_tail_s"], 1))
        tActFrac = 1 - (tB["idle_head_s"] + tB["idle_tail_s"]) / max(tB["window_s"], MIN_WINDOW_S)
        tGoodInActive = tGood.sum() / max(tActFrac * len(tGood), 1)
        tRow.update(good_frac=round(float(tGood.mean()), 3), long_tag_gap_pct=round(float(tB["long_gap"].mean()) * PCT, 1),
                    close_events_L=tB["close_events_L"], close_events_R=tB["close_events_R"], ymed_L=round(tB["ymed_L"], 3),
                    ymed_R=round(tB["ymed_R"], 3))
        if tB["ymed_L"] <= 0 or tB["ymed_R"] >= 0:
            tFlags.append(f"LR_SWAP? left arm median y {tB['ymed_L']:+.3f}, right {tB['ymed_R']:+.3f}")
        for tS in ("L", "R"):
            if tB[f"close_events_{tS}"] == 0:
                tFlags.append(f"no open->close event {tS}")
            if tB[f"flicker_{tS}"]:
                tFlags.append(f"gripper flicker {tS} (> {FLICKER_TOGGLES_PER_S} toggles / 1 s)")
        if tRow["long_tag_gap_pct"] > 0:
            tFlags.append(f"tag gaps > 1 s on {tRow['long_tag_gap_pct']} % of frames -> masked")
        tRow["qa_flags"] = "; ".join(tFlags)
        tSegs = valid_segments(tB["valid"], int(MIN_SEG_S / ACTION_DT_S))       # after the grasp / IK / gap masks
        if not tSegs:
            raise ValueError(f"no valid segment >= {MIN_SEG_S:.0f} s")
        tRow["good_frac"] = round(float(tGoodInActive), 3)       # of the ACTIVE (non-idle) part
        if tGoodInActive < a.min_good_frac:
            raise ValueError(f"only {tGoodInActive * PCT:.0f} % of the active frames valid and outside > 1 s tag gaps "
                             f"(< {a.min_good_frac * PCT:.0f} %)")
        tFm = {f"{tK}_{tS}": tB[f"{tK}_{tS}"] for tS in ("L", "R")
               for tK in ("cov", "maxgap", "jumps", "stitch_jumps", "jump_max_mm", "disp_mean", "disp_max", "disp_contact_max")}
        tRow.update({tK: round(float(tV), 3) for tK, tV in tFm.items()})
        tJumps = tFm["jumps_L"] + tFm["jumps_R"]
        tContactMax = max(tFm["disp_contact_max_L"], tFm["disp_contact_max_R"])
        tBig = max(tFm["jump_max_mm_L"], tFm["jump_max_mm_R"]) > JUMP_BIG_MM
        if a.robotize == "ik_v1":
            tContactMax = 0.0                                     # contact handled by the per-grasp gate below
            tFree = np.r_[tB["disp_L"][~tB["grasp_L"].astype(bool)], tB["disp_R"][~tB["grasp_R"].astype(bool)]]
            tRow.update(grasps=tB["grasp_total"], grasp_warn=tB["grasp_warn"], grasp_fail=tB["grasp_fail"],
                        grasp_max_mm=" ".join(map(str, tB["grasp_max_mm"])),
                        min_link_gap_mm=round(float(tB["gap_m"].min()) * MM_PER_M, 1),
                        ik_fail_L_pct=round(float(tB["ikfail_L"].mean()) * PCT, 1),
                        ik_fail_R_pct=round(float(tB["ikfail_R"].mean()) * PCT, 1),
                        lim_L_pct=round(float(tB["lim_L"].mean()) * PCT, 1),
                        lim_R_pct=round(float(tB["lim_R"].mean()) * PCT, 1), free_disp_p50_mm=round(float(np.median(tFree)), 1))
            if tB["grasp_total"] and tB["grasp_fail"] > tB["grasp_total"] / 2:
                tRow.update(fused_decision="EXCLUDE", episode_weight=0.0)
                tRow["qa_flags"] = "; ".join(tFlags)
                raise ValueError(f"grasp gate EXCLUDE: {tB['grasp_fail']}/{tB['grasp_total']} grasps > {GRASP_WARN_MM:.0f} mm")
            if tB["grasp_fail"]:
                tFlags.append(f"{tB['grasp_fail']} grasp(s) > {GRASP_WARN_MM:.0f} mm masked")
            if tB["grasp_warn"]:
                tFlags.append(f"{tB['grasp_warn']} grasp(s) 20-50 mm -> weight {GRASP_W}")
        if min(tFm["cov_L"], tFm["cov_R"]) < COVERAGE_WARN:
            tFlags.append(f"fused coverage {min(tFm['cov_L'], tFm['cov_R']) * PCT:.0f} % < 95 %")
        if max(tFm["maxgap_L"], tFm["maxgap_R"]) >= GAP_WARN_S:
            tFlags.append(f"longest gap {max(tFm['maxgap_L'], tFm['maxgap_R']):.1f} s")
        if tJumps >= JUMPS_EXCL or tBig or tContactMax >= CONTACT_EXCL_MM:
            tWhy = [f"jumps {tJumps}" if tJumps >= JUMPS_EXCL else "", "jump > 100 mm" if tBig else "",
                    f"contact robotize {tContactMax:.0f} mm" if tContactMax >= CONTACT_EXCL_MM else ""]
            tRow.update(fused_decision="EXCLUDE", episode_weight=0.0)
            tRow["qa_flags"] = "; ".join(tFlags)
            raise ValueError("fused-quality EXCLUDE: " + ", ".join(tX for tX in tWhy if tX))
        if tJumps > 0 or tContactMax >= CONTACT_DOWN_MM:
            tRow.update(fused_decision="DOWNWEIGHT", episode_weight=DOWN_W)
            tFlags.append(f"downweight {DOWN_W}: jumps {tJumps}, contact robotize max {tContactMax:.0f} mm")
        else:
            tRow.update(fused_decision="TRAIN", episode_weight=1.0)
        tRow["qa_flags"] = "; ".join(tFlags)
        tB["episode_weight"] = tRow["episode_weight"]
        if a.clean_min_tag and min(tRow["tag_seen_L_pct"], tRow["tag_seen_R_pct"]) < a.clean_min_tag * PCT:
            raise ValueError(f"not clean: tag_seen L/R {tRow['tag_seen_L_pct']}/{tRow['tag_seen_R_pct']} % "
                             f"< {a.clean_min_tag * PCT:.0f} %")
        tRow["status"] = "eligible"
        built[tKey] = (t, tB, tSegs)
    except Exception as tEx:
        tRow.update(status="rejected", reject_reason=f"{type(tEx).__name__}: {tEx}", qa_decision="dropped")
    return tRow


def _writeEpisode(row, split, built, dss, roots, base, feats, task):
    """pass 2 of one row: its valid segments -> LeRobot episodes of its split (row status / reject_reason updated)"""
    tKey = row["episode_key"]
    tSplit = split.get(tKey, "")
    row["split"] = tSplit
    if row["status"] != "eligible":
        return
    if not tSplit:
        row.update(status="rejected", reject_reason="not in the frozen split", qa_decision="dropped")
        return
    try:
        tT, tB, tSegs = built[tKey]
        if tSplit not in dss:
            dss[tSplit] = createDataset(f"local/{base.name}_{tSplit}", roots[tSplit], feats)
        tDs = dss[tSplit]
        tEps = []
        for tX, tY in tSegs:
            tTL, tTR, tGL, tGR = tB["T_left"][tX:tY], tB["T_right"][tX:tY], tB["g_left"][tX:tY], tB["g_right"][tX:tY]
            tState = proprio_rows_lr(tTL, tTR, tGL, tGR)
            tAct = abs_action_rows(tTL, tTR, tGL, tGR)
            tPoseAbs = np.concatenate([T_to_pose9(tTL), T_to_pose9(tTR), tGL[:, None], tGR[:, None]], -1).astype(np.float32)
            tQuality = np.stack([tB[tK][tX:tY] for tK in QUALITY_COLUMNS], -1).astype(np.float32)
            for tJ, tI in enumerate(range(tX, tY)):
                tFrame = {f"observation.images.{tC}": loadImage(tB, tC, int(tB[f"img_{tC}"][tI])) for tC in CAMS}
                tFrame.update({"observation.state": tState[tJ], "action": tAct[tJ], "pose_abs": tPoseAbs[tJ],
                               "quality": tQuality[tJ],
                               "weight": np.array([tB["episode_weight"] * tB["grasp_w"][tI]], np.float32),
                               "weight_frame": tB["weight"][tI:tI + 1],
                               "gripper_cont": np.array([tB["gc_left"][tI], tB["gc_right"][tI]], np.float32),
                               "src_index": np.array([tB[f"img_{tC}"][tI] for tC in CAMS], np.int64),
                               "img_residual_ms": np.array([tB[f"res_{tC}"][tI] for tC in CAMS], np.float32), "task": task})
                tDs.add_frame(tFrame)
            tEps.append(tDs.meta.total_episodes)
            tDs.save_episode()
        row.update(status="converted", lerobot_episodes=" ".join(map(str, tEps)), qa_decision=row["fused_decision"])
    except Exception as tEx:
        row.update(status="rejected", reject_reason=f"write {type(tEx).__name__}: {tEx}", qa_decision="dropped")


def _writeTables(roots, rows, splitDoc, meta):
    tCols = list(dict.fromkeys(tK for tR in rows for tK in tR))
    for tS, tR in roots.items():
        tR.mkdir(parents=True, exist_ok=True)
        (tR / "meta").mkdir(exist_ok=True)
        with open(tR / "episode_table.csv", "w", newline="") as tFh:
            tWr = csv.DictWriter(tFh, fieldnames=tCols)
            tWr.writeheader()
            tWr.writerows(rows)     # full table (all splits) in every root
        (tR / "meta" / "split.json").write_text(json.dumps(dict(splitDoc, this_split=tS), indent=1))
        (tR / "source_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
        shutil.copy(tR / "episode_table.csv", tR / "conversion_manifest.csv")              # name used by training/smoke_train.sh


def main(argv: list[str] | None = None) -> int:
    tA = buildParser().parse_args(argv)
    tTask = tA.task
    tMeta0 = sourceMeta(tTask)
    try:
        tJawCal = calibration.checkJawCalib()
    except calibration.JawCalibrationError as tE:
        sys.exit(str(tE))
    tJawMin = time.mktime(time.strptime(tA.jaw_newer_than, "%Y-%m-%d %H:%M")) if tA.jaw_newer_than else 0.0
    tTakes = [pathlib.Path(tT).expanduser().resolve() for tT in tA.takes]
    tDerived = pathlib.Path(tA.derived).expanduser().resolve()
    tAsgs = {}
    for tT in tTakes:
        try:
            tAsgs[tT.name] = prepTake(tT, tDerived, tA.pose_src, tA.robotize, tA.force_prep)
            tG = tAsgs[tT.name]
            print(f"[prep] {tT.name}: rig_of_arm {tG['rig_of_arm']} cam_of_arm {tG['cam_of_arm']} "
                  f"rig_swapped={tG['rig_labels_swapped']} cam_swapped={tG['cam_labels_swapped']} corr {tG['flow_corr']}",
                  flush=True)
        except ArmAssignmentError as tE:
            raise SystemExit(str(tE))           # 10-22: an assignment conflict stops the whole conversion, never guess
        except Exception as tEx:
            print(f"[prep] {tT.name}: FAILED {type(tEx).__name__}: {tEx}", flush=True)
    if tA.prep_only:
        return 0
    tCands = [(tT, tD) for tT in tTakes for tD in demoWindows(tT)[: (tA.max_demos or None)]]
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401  (import check before pass 1, as before)
    tBase = pathlib.Path(tA.out).expanduser()
    tRoots = {tS: tBase.parent / f"{tBase.name}_{tS}" for tS in ("train", "val", "test")}
    for tR in tRoots.values():
        if tR.exists() and not tA.qa_only:
            if not tA.overwrite:
                sys.exit(f"{tR} exists (pass --overwrite)")
            shutil.rmtree(tR)
    tFeats = baseFeatures()
    tFeats.update({"quality": dict(dtype="float32", shape=(len(QUALITY_NAMES),), names=QUALITY_NAMES),
                   "weight": dict(dtype="float32", shape=(1,), names=["weight"]),
                   "weight_frame": dict(dtype="float32", shape=(1,), names=["weight_frame"])})
    tFeats.update(tailFeatures())
    tRows, tBuilt = [], {}
    for tT, tD in tCands:                                          # ---- pass 1: grid + QA (no frames written)
        tRows.append(_passOne(tT, tD, tAsgs, tDerived, tA, tJawMin, tBuilt))
    if tA.qa_only:
        tCols = list(dict.fromkeys(tK for tR in tRows for tK in tR))
        with open(tA.qa_only, "w", newline="") as tFh:
            tWr = csv.DictWriter(tFh, fieldnames=tCols)
            tWr.writeheader()
            tWr.writerows(tRows)
        print("qa-only:", sum(tR["status"] == "eligible" for tR in tRows), "eligible of", len(tRows))
        return 0
    # ---- split over the ELIGIBLE episodes (80/10/10, seeded) or a frozen split file
    tElig = [tR["episode_key"] for tR in tRows if tR["status"] == "eligible"]
    if tA.split_file:
        tFrozen = json.loads(pathlib.Path(tA.split_file).read_text())["episodes"]
        tSplit = {tK: tS for tS, tKs in tFrozen.items() for tK in tKs}
        tMissing = [tK for tK in tSplit if tK not in tElig]
        if tMissing:
            print("WARNING frozen-split episodes not eligible now:", tMissing, flush=True)
    else:
        tSplit = splitAssign(tElig, tA.seed)
    tDss = {}
    for tRow in tRows:
        _writeEpisode(tRow, tSplit, tBuilt, tDss, tRoots, tBase, tFeats, tTask)
        print(f"{tRow['episode_key']} [{tRow['split']}] {tRow['status']} {tRow['reject_reason']} "
              f"frames={tRow.get('frames_written')} "
              f"tag L/R {tRow.get('tag_seen_L_pct')}/{tRow.get('tag_seen_R_pct')} jaw L/R {tRow.get('jaw_valid_L_pct')}/"
              f"{tRow.get('jaw_valid_R_pct')} open L/R {tRow.get('open_frac_L')}/{tRow.get('open_frac_R')} "
              f"w {tRow.get('mean_weight')} | {tRow.get('qa_flags', '')}", flush=True)
    for tDs in tDss.values():
        tDs.finalize()
    tMethod = (("FROZEN " + tA.split_file) if tA.split_file
               else "episode-level, seeded, stratified across takes over the QA-eligible demos (take/demoNN), 85/8/7")
    tSplitDoc = dict(method=tMethod, seed=tA.seed, clean_min_tag=tA.clean_min_tag,
                     episodes={tS: sorted(tK for tK, tV in tSplit.items() if tV == tS) for tS in ("train", "val", "test")})
    tMeta = dict(tMeta0, pose_src=tA.pose_src or "tag_only", robotize=tA.robotize,
                 grasp_gate=dict(ok_mm=GRASP_OK_MM, warn_mm=GRASP_WARN_MM, warn_weight=GRASP_W,
                                 rule="per grasp window (+-0.5 s around each jaw close): max |robotized - fused| < 20 ok, "
                                      "20-50 weight 0.5, > 50 masked; > half of grasps masked -> drop; free space: mask only IK "
                                      "error > 10 mm; link gap < 0.115 m masked"),
                 converter="tag_rig_to_lerobot.py", contract_version=contractVersion(), takes=[tT.name for tT in tTakes],
                 assignments=tAsgs, jaw_calibration=tJawCal, quality_names=QUALITY_NAMES, img_tolerance_s=IMG_TOL_S,
                 max_src_gap_s=MAX_SRC_GAP_S,
                 tag_gap_mask_s=tA.tag_gap_s, min_good_frac=tA.min_good_frac)
    _writeTables(tRoots, tRows, tSplitDoc, tMeta)
    print("done", sum(tR["status"] == "converted" for tR in tRows), "converted,", sum(tR["status"] == "rejected" for tR in tRows),
          "rejected;",
          {tS: sum(1 for tR in tRows if tR["status"] == "converted" and tR["split"] == tS) for tS in tRoots})
    return 0


if __name__ == "__main__":
    sys.exit(main())
