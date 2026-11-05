#!/usr/bin/env python3
"""[2026-10-23] Tag-rig laundry dataset v2 (the current builder, dataset stage): ONE demo = ONE unsplit LeRobot episode over the
idle-trimmed window, with PER-ARM, PER-STEP loss masks instead of cutting (user requirement 10-23; v1 kept only frames where BOTH
arms were valid and split at every gap -> 2-6 s pieces).

Window   reports/idle_trim.csv (canonical, coordinator 10-23): demo start (`자유:` cue) + [keep_start_s, keep_end_s]
Pose     robotized ik_v1 (robotize_ik) over the whole window: fused / tag position where measured, the tracker's interpolated
         position + IMU-filled rotation elsewhere (umi_traj --imu-fill), then IK. Every frame stays in sequence.
Per frame, per arm (stored; computed by unsplit_build):
  measured[2]   the pose sample is measured (fused_valid, or ntags > 0 for tag-only) -- else filled
  fill_long[2]  inside a filled run longer than FILL_MASK_S (0.5 s)  -> never a training target (IMU shows real motion hidden
  there)
  anchor_ok[4]  [pos_ok_L, rot_ok_L, pos_ok_R, rot_ok_R]: pos_ok = not in a fill > 0.5 s AND IK <= 10 mm AND grasp gate ok;
                rot_ok = IK <= 10 mm AND grasp gate ok (IMU-filled rotation stays valid in pose gaps -- 10-23 mask spec)
  action_mask[20]  validity of the stored absolute target (= frame i+1): xyz dims = pos_ok, rot6d dims = rot_ok, gripper = jaw
  label valid The training glue (laundry_training.loss_masks) builds the chunk mask from frames t..t+31 (ACT) / t-1..t+31 (DiT)
  and the anchor's pose_ok. The state stays filled (no mask). NOTE (11-01 audit): the IK terms above apply only to --label ikfk.
  For --label rob / fused (labels A / F) the IK error and the grasp displacement are set to 0, so --ik-fail-mm has no effect:
  pos_ok / rot_ok are all 1 in stitched mode and come from fill length only in gated mode. A demo without an ik_v1 file is not
  built.
Episode drop only if: measured < 50 % for BOTH arms after the trim, or a filled run > 5 s overlaps a grasp window (+-0.5 s of a
close). Split: frozen by demo -- demos of the v1 split keep their split; new eligible demos get a stratified, seeded 85/8/7
assignment. 11-04: every split named in --old-split (e.g. silver / reject) gets its own root next to train / val / test.
Environment: FILL_MASK_S, JAW_FILE_LEFT / JAW_FILE_RIGHT, EP_WEIGHT (see unsplit_build).
usage: tag_rig_unsplit.py --out data/lerobot/tagrig_laundry_v2u --takes TAKE... [--old-split .../meta/split.json] [--render 4]
[--cam-shift-ms 0]
    python -m laundry_dataset.tag_rig_unsplit --out ... --takes ..."""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import shutil
import sys

import numpy as np

from laundry_common import paths as egopaths
from laundry_shared.contracts.action_contract import ACTION_DT_S, T_to_pose9
from laundry_shared.umi_to_lerobot import ABS_NAMES, proprio_rows_lr
from laundry_shared.xvla_core import abs_action_rows

from . import calibration
from .arm_assignment import prepTake
from .export_common import ACTION_DIM, CAMS, TASK, baseFeatures, createDataset, loadImage, splitAssign, tailFeatures
from .tag_rig_to_lerobot import contractVersion, sourceMeta
from .takes import demoKey, demoWindows, loadIdleTrim
from .unsplit_build import (FILL_MASK_S, GRASP_OK_MM, GRASP_W, GRASP_WARN_MM, IK_SEVERE_MM, UnsplitOptions, build, runs)
from .unsplit_render import renderMaskVideo

# user 10-23: never use (aborted / empty)
EXCLUDE_TAKES = {"20261023_113219", "20261023_115848", "20261023_112832", "20261023_113038"}
TRIM_CSV = egopaths.IDLE_TRIM_CSV
DROP_MEASURED, DROP_FILL_IN_GRASP_S = 0.5, 5.0
SEVERE_RUN_S, SEVERE_FLAG_S = 0.5, 1.0   # IK > 50 mm runs counted from 0.5 s; a run >= 1 s flags the demo (QA only)
IK_BANDS = (("le10", -1, 10), ("10_20", 10, 20), ("20_30", 20, 30), ("30_50", 30, 50), ("gt50", 50, 1e9))
POS_DIMS, ROT_DIMS = [0, 9], [3, 12]     # first xyz / rot6d dim of each arm in the action mask
RENDER_TAKES = {"20261022_124754", "20261022_134315", "20261022_161756", "20261023_113550"}   # QA videos only for these takes
BASE_SPLITS = ["train", "val", "test"]
NS_PER_S = 1e9
NS_PER_MS = 1e6
MM_PER_M = 1000
PCT = 100
REASON_PRINT_CHARS = 80              # drop reason shown in the progress line
N_ANCHOR = 4                         # anchor_ok / anchor_w columns [pos_L, rot_L, pos_R, rot_R]


class EpisodeSplitError(RuntimeError):
    """A written split does not hold exactly one continuous LeRobot episode per converted demo (10-23 hard rule)."""


def buildParser() -> argparse.ArgumentParser:
    tAp = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tAp.add_argument("--out", required=True)
    tAp.add_argument("--takes", nargs="+", required=True)
    tAp.add_argument("--derived", default=str(egopaths.DERIVED))
    tAp.add_argument("--old-split",
                     default=str(egopaths.TRAIN_ROOT / "data/lerobot/tagrig_laundry_v1_fused_train/meta/split.json"))
    tAp.add_argument("--seed", type=int, default=0)
    tAp.add_argument("--render", type=int, default=4)
    tAp.add_argument("--render-dir", default=str(egopaths.REPORTS / "tagrig_qa/unsplit_videos"))
    tAp.add_argument("--cam-shift-ms", type=float, default=0.0,
                     help="shift Orbbec (pose + colour) time; coordinator 10-23: ~-40 ms vs IMU, OFF until confirmed")
    tAp.add_argument("--pose-src", default="auto", choices=["auto", "_fused", "_fgo", ""],
                     help="auto = _fused when present else tag-only; _fgo = fgo_v0 (10-23)")
    tAp.add_argument("--pos-conf-dir", default="",
                     help="per-frame SLAM<->IMU consistency files (~/ego_collector/datasets/m3_runs/_imu_consistency/) "
                          "-> variant imuconf")
    tAp.add_argument("--conf-high", type=float, default=None)
    tAp.add_argument("--conf-mid", type=float, default=None)
    tAp.add_argument("--w-mid", type=float, default=0.3)
    tAp.add_argument("--demos", nargs="*", default=[], help="subset of demo keys take/demoNN (QA passes on a few demos)")
    tAp.add_argument("--ik-fail-mm", type=float, default=30.0,
                     help="10-23 user: hard IK cutoff for the pos/rot masks (was 10 mm)")
    tAp.add_argument("--mask-mode", default="gated", choices=["gated", "stitched"],
                     help="stitched: pos weight 1.0 on the continuous stitched trajectory, masks = IK > 10 mm (pos+rot) + "
                          "invalid jaw (grip); no drops")
    tAp.add_argument("--grasp-fill-drop", default="on", choices=["on", "off"],
                     help="off = v2u_all (10-23 decision): keep demos whose grasp is inside a > 5 s fill (positions stay masked)")
    tAp.add_argument("--label", default="ikfk", choices=["ikfk", "rob", "fused"],
                     help="10-26: rob = action_cartesian_robotized (A, policy target), ikfk = old IK->FK (B); "
                          "10-29: fused = human TCP without the table lift (F)")
    tAp.add_argument("--grip-label", default="binary", choices=["binary", "cont"],
                     help="11-02: cont = gripper target is the continuous servo width (gripper_cont)")
    tAp.add_argument("--speed", type=float, default=1.0,
                     help="10-29: playback factor, 0.3 = demos 3.33x slower (pose, jaw and video on one timeline)")
    tAp.add_argument("--qa-only", default="")
    tAp.add_argument("--overwrite", action="store_true")
    return tAp


def _qaFields(row, b, tr, options):
    """QA columns of one built demo into its row (order = the episode_table.csv columns)"""
    tN = b["n"]
    row.update(trim_start_s=tr["keep_start_s"], trim_end_s=tr["keep_end_s"], cut_head_s=tr["cut_head_s"],
               cut_tail_s=tr["cut_tail_s"],
               frames=tN, length_s=round(tN * ACTION_DT_S, 1))
    for tS in "LR":
        tGi = b[f"ginfo_{tS}"]
        row.update({f"measured_{tS}_pct": round(float(b[f"meas_{tS}"].mean()) * PCT, 1),
                    f"fill_long_{tS}_pct": round(float(b[f"fill_{tS}"].mean()) * PCT, 1),
                    f"ik_fail_{tS}_pct": round(float((b[f"ikmm_{tS}"] > options.mIkFailMm).mean()) * PCT, 1),
                    f"pos_masked_{tS}_pct": round(float(1 - b[f"pos_ok_{tS}"].mean()) * PCT, 1),
                    f"rot_masked_{tS}_pct": round(float(1 - b[f"rot_ok_{tS}"].mean()) * PCT, 1),
                    f"no_pose_{tS}_pct": round(float(1 - b[f"meas_{tS}"].mean()) * PCT, 1),
                    f"grip_masked_{tS}_pct": round(float(1 - b[f"jv_{tS}"].mean()) * PCT, 1), f"grasps_{tS}": len(tGi),
                    f"grasp_warn_{tS}": sum(GRASP_OK_MM <= tX["max_mm"] <= GRASP_WARN_MM for tX in tGi),
                    f"grasp_fail_{tS}": sum(tX["max_mm"] > GRASP_WARN_MM for tX in tGi),
                    f"grasp_max_mm_{tS}": " ".join(str(tX["max_mm"]) for tX in tGi)})
    for tV in options.mVariants:
        row.update({f"pos_eff_{tV}_{tS}_frames": round(float(b[f"pw_{tV}_{tS}"].sum()), 1) for tS in "LR"})
    for tS in "LR":
        tE = b[f"ikmm_{tS}"]
        for tLab, tLo, tHi in IK_BANDS:
            row[f"ik_{tLab}_{tS}_pct"] = round(float(((tE > tLo) & (tE <= tHi)).mean()) * PCT, 1)
        tRuns = [(tY - tX) * ACTION_DT_S for tX, tY in runs(tE > IK_SEVERE_MM)]
        tLong = [tR for tR in tRuns if tR >= SEVERE_RUN_S]
        row.update({f"severe_runs_{tS}": len(tLong), f"severe_runs_s_{tS}": round(sum(tLong), 1),
                    f"severe_flag_{tS}": bool(any(tR >= SEVERE_FLAG_S for tR in tRuns)),
                    f"severe_runs_1s_{tS}": int(sum(tR >= SEVERE_FLAG_S for tR in tRuns)),
                    f"severe_longest_s_{tS}": round(max(tRuns or [0.0]), 1)})
    row.update(conf_L_pct=round(float(b["conf_L"].mean()) * PCT, 1), conf_R_pct=round(float(b["conf_R"].mean()) * PCT, 1))
    row.update(min_link_gap_mm=round(float(b["gap_m"].min()) * MM_PER_M, 1),
               pos_supervised_frames=int(np.round(b["action_mask"][:, POS_DIMS].mean(1).sum())),
               rot_supervised_frames=int(np.round(b["action_mask"][:, ROT_DIMS].mean(1).sum())),
               img_res_max_ms=round(float(max(b[f"res_{tC}"].max() for tC in CAMS)), 1))


def _demoRow(t, d, tr, src, asgs, derived, a, options, built) -> dict:
    """build + QA of one demo -> its episode-table row (status eligible / dropped); eligible demos go into `built`"""
    tKey = demoKey(t.name, d.mIdx)
    tSrc = src.get(t.name)
    tRow = dict(episode_key=tKey, take=t.name, demo=d.mIdx,
                pose_src=(tSrc or "tag_only").lstrip("_") if tSrc is not None else "missing",
                status="", reason="")
    try:
        if t.name not in asgs:
            raise ValueError("take prep failed")
        if tr is None:
            raise ValueError("no idle-trim row (demo outside the recording)")
        tWin = (d.mT0Ns + int(float(tr["keep_start_s"]) * NS_PER_S), d.mT0Ns + int(float(tr["keep_end_s"]) * NS_PER_S))
        tB = build(t, derived / (t.name + src[t.name]), asgs[t.name], d, tWin, int(a.cam_shift_ms * NS_PER_MS), options)
        _qaFields(tRow, tB, tr, options)
        tGated = options.mMaskMode != "stitched"
        if tGated and tRow["measured_L_pct"] < DROP_MEASURED * PCT and tRow["measured_R_pct"] < DROP_MEASURED * PCT:
            raise ValueError(f"measured < 50 % for both arms ({tRow['measured_L_pct']} / {tRow['measured_R_pct']} %)")
        tLongInGrasp = max([tX["fill_overlap_s"] for tS in "LR" for tX in tB[f"ginfo_{tS}"]] or [0.0])
        tRow["fill_in_grasp_s"] = round(tLongInGrasp, 1)
        if tGated and a.grasp_fill_drop == "on" and tLongInGrasp > DROP_FILL_IN_GRASP_S:
            raise ValueError(f"filled run of {tLongInGrasp:.1f} s inside a grasp window")
        tRow["status"] = "eligible"
        built[tKey] = (t, tB)
    except Exception as tEx:
        tRow.update(status="dropped", reason=f"{type(tEx).__name__}: {tEx}")
    return tRow


def _features(variants) -> dict:
    tFeats = baseFeatures()
    tFeats.update({"action_mask": dict(dtype="float32", shape=(ACTION_DIM,), names=[f"mask_{tX}" for tX in ABS_NAMES]),
                   "anchor_ok": dict(dtype="float32", shape=(N_ANCHOR,), names=["pos_ok_L", "rot_ok_L", "pos_ok_R", "rot_ok_R"]),
                   "measured": dict(dtype="float32", shape=(2,), names=["measured_L", "measured_R"]),
                   "fill_long": dict(dtype="float32", shape=(2,), names=["fill_long_L", "fill_long_R"]),
                   "gap_len_s": dict(dtype="float32", shape=(2,), names=["gap_len_L", "gap_len_R"]),
                   "pos_conf": dict(dtype="float32", shape=(2,), names=["conf_L", "conf_R"]),
                   "ik_warn": dict(dtype="float32", shape=(2,), names=["ik_warn_L", "ik_warn_R"]),
                   "ik_severe": dict(dtype="float32", shape=(2,), names=["ik_severe_L", "ik_severe_R"]),
                   "ik_err_mm": dict(dtype="float32", shape=(2,), names=["ik_err_L", "ik_err_R"]),
                   "grasp_win": dict(dtype="float32", shape=(2,), names=["grasp_win_L", "grasp_win_R"]),
                   **{f"action_weight_{tV}": dict(dtype="float32", shape=(ACTION_DIM,), names=[f"w_{tX}" for tX in ABS_NAMES])
                      for tV in variants},
                   **{f"anchor_w_{tV}": dict(dtype="float32", shape=(N_ANCHOR,), names=["pos_L", "rot_L", "pos_R", "rot_R"])
                      for tV in variants},
                   **{f"pos_w_{tV}": dict(dtype="float32", shape=(2,), names=["pos_w_L", "pos_w_R"]) for tV in variants},
                   "weight": dict(dtype="float32", shape=(1,), names=["weight"])})
    tFeats.update(tailFeatures())
    return tFeats


def _pair(b, key, i, dtype=np.float32):
    return np.array([b[f"{key}_L"][i], b[f"{key}_R"][i]], dtype)


def _writeEpisode(ds, b, gripLabel, variants):
    """every frame of one built demo -> one LeRobot episode (not saved yet)"""
    tTL, tTR, tGL, tGR = b["T_left"], b["T_right"], b["g_left"], b["g_right"]
    # 11-02 user: train the jaw CONTINUOUS (the measured servo width, = gripper_cont), binarise only at inference
    if gripLabel == "cont":
        tGL, tGR = np.asarray(b["gc_left"], np.float32), np.asarray(b["gc_right"], np.float32)
    tState = proprio_rows_lr(tTL, tTR, tGL, tGR)
    tAct = abs_action_rows(tTL, tTR, tGL, tGR)
    tPoseAbs = np.concatenate([T_to_pose9(tTL), T_to_pose9(tTR), tGL[:, None], tGR[:, None]], -1).astype(np.float32)
    for tI in range(b["n"]):
        tFrame = {f"observation.images.{tC}": loadImage(b, tC, int(b[f"img_{tC}"][tI])) for tC in CAMS}
        tFrame.update({"observation.state": tState[tI], "action": tAct[tI], "pose_abs": tPoseAbs[tI],
                       "action_mask": b["action_mask"][tI],
                       "anchor_ok": b["anchor_ok"][tI], "measured": _pair(b, "meas", tI), "fill_long": _pair(b, "fill", tI),
                       "gap_len_s": _pair(b, "gl", tI), "pos_conf": _pair(b, "conf", tI), "ik_warn": _pair(b, "ikwarn", tI),
                       "ik_severe": _pair(b, "iksev", tI), "ik_err_mm": _pair(b, "ikmm", tI), "grasp_win": _pair(b, "grasp", tI),
                       **{f"action_weight_{tV}": b[f"aw_{tV}"][tI] for tV in variants},
                       **{f"anchor_w_{tV}": b[f"anw_{tV}"][tI] for tV in variants},
                       **{f"pos_w_{tV}": np.array([b[f"pw_{tV}_L"][tI], b[f"pw_{tV}_R"][tI]], np.float32) for tV in variants},
                       "weight": np.array([b["weight"][tI]], np.float32),
                       "gripper_cont": np.array([b["gc_left"][tI], b["gc_right"][tI]], np.float32),
                       "src_index": np.array([b[f"img_{tC}"][tI] for tC in CAMS], np.int64),
                       "img_residual_ms": np.array([b[f"res_{tC}"][tI] for tC in CAMS], np.float32), "task": TASK})
        ds.add_frame(tFrame)


def _checkUnsplit(roots, dss, rows, base):
    """10-23 hard rule: NO split episodes. #episodes per split == #converted demos in it, frame_index continuous 0..n-1, length
    == built"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    for tSplit, tRoot in roots.items():
        if tSplit not in dss:
            continue
        tChk = LeRobotDataset(f"local/{base.name}_{tSplit}", root=tRoot)
        tHf = tChk.hf_dataset.with_format("numpy")
        tEp = np.asarray(tHf["episode_index"])
        tFi = np.asarray(tHf["frame_index"])
        tConv = [tR for tR in rows if tR["status"] == "converted" and tR.get("split") == tSplit]
        if tChk.meta.total_episodes != len(tConv):
            raise EpisodeSplitError(f"{tSplit}: {tChk.meta.total_episodes} episodes != {len(tConv)} demos")
        for tR in tConv:
            tFrames = tFi[tEp == tR["lerobot_episode"]]
            if not (len(tFrames) == tR["frames"] and np.array_equal(tFrames, np.arange(len(tFrames)))):
                raise EpisodeSplitError(f"{tR['episode_key']}: episode split / frame_index not continuous")
        print(f"[assert] {tSplit}: {len(tConv)} demos == {tChk.meta.total_episodes} episodes, every frame_index continuous",
              flush=True)


def _writeCsv(path, rows):
    tCols = list(dict.fromkeys(tK for tR in rows for tK in tR))
    with open(path, "w", newline="") as tFh:
        tW = csv.DictWriter(tFh, fieldnames=tCols)
        tW.writeheader()
        tW.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    tA = buildParser().parse_args(argv)
    try:
        calibration.checkJawCalib()
    except calibration.JawCalibrationError as tE:
        sys.exit(str(tE))
    if tA.pos_conf_dir and (tA.conf_high is None or tA.conf_mid is None):
        sys.exit("--pos-conf-dir needs --conf-high and --conf-mid (from the calibration)")
    tOptions = UnsplitOptions(mMaskMode=tA.mask_mode, mIkFailMm=tA.ik_fail_mm, mLabel=tA.label, mSpeed=tA.speed,
                              **(dict(mConfDir=tA.pos_conf_dir, mConfHigh=tA.conf_high, mConfMid=tA.conf_mid,
                                      mWMid=tA.w_mid) if tA.pos_conf_dir else {}))
    tVariants = list(tOptions.mVariants)
    tTrim = loadIdleTrim(TRIM_CSV)
    tTakes = [pathlib.Path(tT).expanduser().resolve() for tT in tA.takes]
    tBad = [tT.name for tT in tTakes if tT.name in EXCLUDE_TAKES]
    if tBad:
        sys.exit(f"excluded takes passed: {tBad}")
    tDerived = pathlib.Path(tA.derived)
    tRows, tBuilt, tAsgs, tSrc = [], {}, {}, {}
    for tT in tTakes:
        tFused = all((tT / f"umi_traj_{tS}_fused.npz").exists()
                     and "fused_valid" in np.load(tT / f"umi_traj_{tS}_fused.npz").files
                     for tS in ("left", "right"))
        if tA.pose_src == "auto":
            tPoseSrc = "_fused" if tFused else ""
        else:
            tPoseSrc = tA.pose_src
            if tA.pose_src and not all((tT / f"umi_traj_{tS}{tA.pose_src}.npz").exists() for tS in ("left", "right")):
                print(f"[prep] {tT.name}: no {tA.pose_src} files")
                tSrc[tT.name] = None
                continue
        tSrc[tT.name] = tPoseSrc
        try:
            tAsgs[tT.name] = prepTake(tT, tDerived, tPoseSrc, "ik_v1")
            print(f"[prep] {tT.name} pose_src={'fused' if tFused else 'tag_only'} rig_of_arm={tAsgs[tT.name]['rig_of_arm']}",
                  flush=True)
        except BaseException as tEx:
            print(f"[prep] {tT.name} FAILED {type(tEx).__name__}: {tEx}", flush=True)
        for tD in demoWindows(tT):
            tKey = demoKey(tT.name, tD.mIdx)
            if tA.demos and tKey not in tA.demos:
                continue
            tRow = _demoRow(tT, tD, tTrim.get((tT.name, tD.mIdx)), tSrc, tAsgs, tDerived, tA, tOptions, tBuilt)
            tRows.append(tRow)
            print(f"{tKey} {tRow['status']} {tRow['reason'][:REASON_PRINT_CHARS]} len {tRow.get('length_s')} s "
                  f"meas L/R {tRow.get('measured_L_pct')}/{tRow.get('measured_R_pct')} "
                  f"pos-masked L/R {tRow.get('pos_masked_L_pct')}/{tRow.get('pos_masked_R_pct')}", flush=True)
    if tA.qa_only:
        _writeCsv(tA.qa_only, tRows)
        print("qa-only:", sum(tR["status"] == "eligible" for tR in tRows), "eligible of", len(tRows))
        return 0
    # ---- frozen split: old keys keep their split, new keys stratified
    tOld = json.loads(pathlib.Path(tA.old_split).read_text())["episodes"] if pathlib.Path(tA.old_split).exists() else {}
    tOldMap = {tK: tS for tS, tKs in tOld.items() for tK in tKs}
    tElig = [tR["episode_key"] for tR in tRows if tR["status"] == "eligible"]
    tSplit = {tK: tOldMap[tK] for tK in tElig if tK in tOldMap}
    tSplit.update(splitAssign([tK for tK in tElig if tK not in tOldMap], tA.seed))
    # noqa: F401  (import check before anything is deleted, as before)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    tBase = pathlib.Path(tA.out).expanduser()
    tSplits = list(dict.fromkeys(BASE_SPLITS + list(tOld.keys())))   # 11-04: extra splits (silver / reject) from --old-split
    tRoots = {tS: tBase.parent / f"{tBase.name}_{tS}" for tS in tSplits}
    for tR in tRoots.values():
        if tR.exists():
            if not tA.overwrite:
                sys.exit(f"{tR} exists (pass --overwrite)")
            shutil.rmtree(tR)
    tFeats = _features(tVariants)
    tDss = {}
    tRenderDir = pathlib.Path(tA.render_dir)
    tRenderDir.mkdir(parents=True, exist_ok=True)
    tRendered = 0
    for tRow in tRows:
        tKey = tRow["episode_key"]
        if tRow["status"] != "eligible":
            continue
        tSp = tSplit[tKey]
        tRow["split"] = tSp
        tT, tB = tBuilt[tKey]
        if tSp not in tDss:
            tDss[tSp] = createDataset(f"local/{tBase.name}_{tSp}", tRoots[tSp], tFeats)
        tDs = tDss[tSp]
        _writeEpisode(tDs, tB, tA.grip_label, tVariants)
        tRow["lerobot_episode"] = tDs.meta.total_episodes
        tDs.save_episode()
        tRow["status"] = "converted"
        if tRendered < tA.render and tRow["take"] in RENDER_TAKES and tRow["take"] not in {tR.get("rendered") for tR in tRows}:
            renderMaskVideo(tB, tRenderDir / f"{tKey.replace('/', '_')}.mp4", tKey)
            tRow["rendered"] = tRow["take"]
            tRendered += 1
        print(f"wrote {tKey} [{tSp}] {tB['n']} frames", flush=True)
    for tDs in tDss.values():
        tDs.finalize()
    _checkUnsplit(tRoots, tDss, tRows, tBase)
    tMethod = "FROZEN by demo: v1 split kept for its demos; new eligible demos stratified (seed %d, 85/8/7)" % tA.seed
    tEpisodes = {tS: sorted(tK for tK, tV in tSplit.items() if tV == tS) for tS in tSplits}
    tSplitDoc = dict(method=tMethod, seed=tA.seed, episodes=tEpisodes,
                     old_split=tA.old_split)
    if tA.grasp_fill_drop == "on":
        tDropTail = f", or a filled run > {DROP_FILL_IN_GRASP_S} s containing a jaw close"
    else:
        tDropTail = " (grasp-fill drop OFF: v2u_all)"
    tDrop = f"measured < {DROP_MEASURED * PCT:.0f} % both arms" + tDropTail
    tMeta = dict(sourceMeta(TASK), dataset="tagrig_laundry_v2u (unsplit + per-arm masks)", converter="tag_rig_unsplit.py",
                 speed=tOptions.mSpeed,
                 label=tOptions.mLabel, grip_label=tA.grip_label, pose_src=tSrc, robotize="ik_v1", trim_csv=str(TRIM_CSV),
                 fill_mask_s=FILL_MASK_S,
                 ik_fail_mm=tOptions.mIkFailMm,
                 ik_qa_flags="ik_warn: cutoff < err <= 50 mm, ik_severe: err > 50 mm (QA only, not weights); "
                             "severe_flag = a > 50 mm run >= 1 s",
                 grasp_gate=dict(ok_mm=GRASP_OK_MM, warn_mm=GRASP_WARN_MM, warn_weight=GRASP_W), drop_rule=tDrop,
                 cam_shift_ms=tA.cam_shift_ms,
                 exclude_takes=sorted(EXCLUDE_TAKES), contract_version=contractVersion())
    for tS, tR in tRoots.items():
        tR.mkdir(parents=True, exist_ok=True)
        (tR / "meta").mkdir(exist_ok=True)
        _writeCsv(tR / "episode_table.csv", tRows)
        (tR / "meta" / "split.json").write_text(json.dumps(dict(tSplitDoc, this_split=tS), indent=1))
        (tR / "source_meta.json").write_text(json.dumps(tMeta, indent=1, ensure_ascii=False))
        shutil.copy(tR / "episode_table.csv", tR / "conversion_manifest.csv")
    print("done", sum(tR["status"] == "converted" for tR in tRows), "converted,", sum(tR["status"] == "dropped" for tR in tRows),
          "dropped;",
          {tS: sum(1 for tR in tRows if tR.get("split") == tS and tR["status"] == "converted") for tS in tRoots})
    return 0


if __name__ == "__main__":
    sys.exit(main())
