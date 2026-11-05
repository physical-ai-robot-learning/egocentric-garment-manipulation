"""Per-demo grid of the unsplit tag-rig dataset: pose label, per-arm / per-step loss masks, position-weight variants, jaw labels
and camera frame indices of one idle-trimmed demo (dataset stage; was build() of tag_rig_unsplit.py, 10-23 .. 11-04).

Inputs: the take (camera clocks, wrist jaw labels JAW_FILE), the derived take (umi_traj_<arm>.npz, ik_v1/demoNN.npz of
robotize_ik) and the trimmed window. Output: a dict of per-frame columns (see tag_rig_unsplit for what is stored):
  measured     the pose sample is measured (fused_valid, or ntags > 0 for tag-only) -- else filled
  fill_long    inside a filled run longer than FILL_MASK_S -> never a position target (IMU shows real motion hidden there)
  pos_ok       not in a long fill AND IK <= --ik-fail-mm AND grasp gate ok (gated); IK only (stitched)
  rot_ok       IK <= --ik-fail-mm AND grasp gate ok (IMU-filled rotation stays valid in pose gaps -- 10-23 mask spec)
  action_mask  validity of the stored absolute target (= frame i+1): xyz dims = pos_ok, rot6d dims = rot_ok, gripper = jaw valid
  NOTE (11-01 audit): the IK terms apply only to --label ikfk. For --label rob / fused (labels A / F) the IK error and the grasp
  displacement are set to 0, so --ik-fail-mm has no effect: pos_ok / rot_ok are all 1 in stitched mode and come from fill length
  only in gated mode.
Environment switches (names / defaults fixed): FILL_MASK_S (0.5; 11-04 user: gated, position masked in CoTracker/tag gaps >= 0.3
s), JAW_FILE_LEFT / JAW_FILE_RIGHT (umi_jaw.npz; 10-31 jaw relabel), EP_WEIGHT (1; read per build, 10-30 per-sample weight of a
build).
"""
from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass

import numpy as np

from laundry_shared.contracts.action_contract import ACTION_DT_S, make_T
from laundry_shared.xvla_core import nearest_frame, resample_pose

from .export_common import MAX_JAW_GAP_S, rampedJaw

FILL_MASK_S = float(os.environ.get("FILL_MASK_S", 0.5))   # 11-04 user: gated, position masked in CoTracker/tag gaps >= 0.3 s
IK_FAIL_MM = 10.0                    # library default; the CLI default is 30 (10-23 user)
GRASP_OK_MM, GRASP_WARN_MM, GRASP_W = 20.0, 50.0, 0.5    # grasp-window gate: < 20 ok, 20-50 weight 0.5, > 50 masked
IK_SEVERE_MM = 50.0                  # ik_warn: cutoff < err <= 50 mm, ik_severe: err > 50 mm (QA only)
JAW_WIN_S = 0.5                      # grasp window = +-0.5 s around jaw TRANSITIONS (open and close edges)
# tag-gap reference baselines; "imuconf" is added with a confidence dir (MAIN track)
VARIANTS = ("strict", "relaxed_2s", "weighted")
CONF_VARIANT = "imuconf"
# "stitched" (10-23 user GO): position weight 1.0 everywhere on the continuous stitched trajectory
MASK_MODES = ("gated", "stitched")
# 10-26 contract: "rob" = action_cartesian_robotized (robotize_ik <arm>_p/R_rob, before IK; IK residual is QA only, never a mask);
# "ikfk" = old label FK(q) of the offline IK path (<arm>_p/R), kept for the A-vs-B comparison; 10-29 "fused" = the human TCP, NO
# table-clearance lift (rob lifted 44 % of frames by ~27 mm)
LABEL_SUFFIX = {"rob": "_rob", "fused": "_fused"}
# 10-31 jaw relabel: per-arm jaw label file (default umi_jaw.npz). JAW_FILE_LEFT=umi_jaw_motorlike.npz -> the crop->servo relabel
# for the left hand only (user review: left OK, right wrong on black cloth); its `valid` (0 = UNKNOWN) masks the BCE jaw loss like
# any jaw gap.
JAW_FILE = {"left": os.environ.get("JAW_FILE_LEFT", "umi_jaw.npz"), "right": os.environ.get("JAW_FILE_RIGHT", "umi_jaw.npz")}
MIN_WINDOW_NS = 5e9                  # trimmed window inside the recording shorter than 5 s -> error
POSE_MAX_GAP_S = 1.0                 # label resampling bracket
NEAREST_MAX_S = 0.07                 # a trajectory sample further than this from a grid time counts as missing
NO_GAP_S = 9.9                       # gap length of a missing sample
CONF_GAP_S = 0.5                     # pos_conf: gap <= 0.5 s
# position-weight variants from the gap length gl (s): strict = fgo valid; relaxed = gl <= 2 s; weighted = 1 / 0.7 / 0.25 / 0
RELAXED_GAP_S = 2.0
WEIGHTED_STEPS = ((0.5, 1.0), (1.0, 0.7), (2.0, 0.25))
CONTACT_GAP_S = 1.0                  # contact rule: no long-gap positions around jaw events
NS_PER_S = 1e9
MS_PER_NS = 1e6
MM_PER_M = 1000
ARMS = (("left", "L"), ("right", "R"))
W_MID = 0.3                          # position weight of a mid-confidence frame (--w-mid default)
XYZ_DIMS, ROT6D_DIMS = 3, 6          # per-arm position / rotation dims of the 20-D action
CONF_KEYS = ("conf", "score", "confidence")


@dataclass(frozen=True)
class UnsplitOptions:
    """the builder settings that were module globals of tag_rig_unsplit (MASK_MODE, IK_FAIL_MM, LABEL, SPEED, CONF,
    FILL_MASK_S)"""
    mMaskMode: str = "gated"
    mIkFailMm: float = IK_FAIL_MM
    mLabel: str = "ikfk"
    # 10-29 (user: demos too fast for the reBot): playback factor. Output rows stay 20 Hz (ACTION_DT_S) but advance ACTION_DT_S *
    # speed of SOURCE time, so pose, action, jaw edges and EVERY camera frame (nearest source frame at the warped time) share one
    # stretched timeline; 0.3 = 3.33x slower. Source-time windows (fill, grasp, jaw ramp) keep their real-time meaning.
    mSpeed: float = 1.0
    mFillMaskS: float = FILL_MASK_S
    mConfDir: str | None = None       # 10-23 user rule: position weight from the marker-free SLAM<->IMU consistency score
    mConfHigh: float | None = None
    mConfMid: float | None = None
    mWMid: float = W_MID

    @property
    def mVariants(self) -> tuple:
        return VARIANTS + ((CONF_VARIANT,) if self.mConfDir else ())


def runs(mask):
    """[(start, end)) of every True run"""
    tX = np.r_[0, mask.astype(int), 0]
    return list(zip(np.flatnonzero(np.diff(tX) == 1), np.flatnonzero(np.diff(tX) == -1)))


def measuredFlag(z):
    """per trajectory sample: measured (fused_valid | slam_ok or tags | tags), else filled (was measured_flag)"""
    if "fused_valid" in z.files:
        return z["fused_valid"].astype(bool)
    if "slam_ok" in z.files:
        return z["slam_ok"].astype(bool) | (z["ntags"] > 0)
    return z["ntags"] > 0


def loadPosConf(confDir, take, rig, arm):
    """per-frame 0..1 SLAM<->IMU consistency for one UMI -> (t_ns, conf). Accepts <dir>/<take>/<rig|arm>.npz,
    <dir>/<take>_<rig|arm>.npz or <dir>/<take>/umi_conf_<rig|arm>.npz with keys t_ns + one of conf / score / confidence (was
    load_pos_conf)"""
    tD = pathlib.Path(confDir)
    for tName in (rig, arm, f"umi_conf_{rig}", f"umi_conf_{arm}"):
        for tFile in (tD / take / f"{tName}.npz", tD / f"{take}_{tName}.npz"):
            if tFile.exists():
                tZ = np.load(tFile)
                tKey = next(tK for tK in CONF_KEYS if tK in tZ.files)
                return tZ["t_ns"].astype(np.int64), tZ[tKey].astype(float)
    raise FileNotFoundError(f"no SLAM<->IMU confidence for {take} {rig}/{arm} under {tD}")


def confWeight(conf, options: UnsplitOptions):
    """High (>= high) 1.0 | Mid (>= mid) w_mid | Low masked; thresholds from the calibration (--conf-high / --conf-mid)
    (10-23: position-weight variants A/B/C, same frames, same split; selected at train time)"""
    return np.where(conf >= options.mConfHigh, 1.0, np.where(conf >= options.mConfMid, options.mWMid, 0.0))


def posWeights(gl, fgoValid, jawWin) -> dict:
    """per-frame position loss weight per variant from the fgo gap length (s; 0 on tag frames) -> dict variant -> (n,)"""
    # tag | gap <= 1.0 s with SLAM | <= 0.5 s without (fgo valid_rule)
    tStrict = fgoValid.astype(float)
    tRelaxed = (gl <= RELAXED_GAP_S).astype(float)
    (tG1, tW1), (tG2, tW2), (tG3, tW3) = WEIGHTED_STEPS
    tW = np.where(gl <= tG1, tW1, np.where(gl <= tG2, tW2, np.where(gl <= tG3, tW3, 0.0)))
    tW = np.where(jawWin & (gl > CONTACT_GAP_S), 0.0, tW)                   # contact rule: no long-gap positions near jaw events
    return dict(strict=tStrict, relaxed_2s=tRelaxed, weighted=tW)


def _armColumns(out, take, dtake, asg, a, s, IK, lo, tis, ki, tg, n, dtSrc, camShiftNs, options):
    tZ = np.load(dtake / f"umi_traj_{a}.npz", allow_pickle=True)
    tTz = (tZ["t_ns"].astype(np.int64) + camShiftNs - lo) / NS_PER_S
    tSfx = LABEL_SUFFIX.get(options.mLabel, "")
    tLabel = make_T(IK[f"{a}_R{tSfx}"], IK[f"{a}_p{tSfx}"])
    tT, _, _, _ = resample_pose(tis, tLabel, np.zeros(len(tis)), tg, max_gap=POSE_MAX_GAP_S)
    tPh = np.stack([np.interp(tg, tis, IK[f"{a}_p_h"][:, tAxis]) for tAxis in range(3)], 1)
    tKz, tRz = nearest_frame(tTz, tg)
    tMeas = measuredFlag(tZ)[tKz] & (tRz <= NEAREST_MAX_S)
    tGl = (tZ["gap_len_s"][tKz] if "gap_len_s" in tZ.files else np.where(tMeas, 0.0, NO_GAP_S)).astype(float)
    tGl = np.where(tRz <= NEAREST_MAX_S, tGl, NO_GAP_S)
    tFill = np.zeros(n, bool)
    for tX, tY in runs(~tMeas):
        if (tY - tX) * dtSrc > options.mFillMaskS:
            tFill[tX:tY] = True
    tIkMm = IK[f"{a}_ik_mm"][ki]
    tDisp = np.linalg.norm(tT[:, :3, 3] - tPh, axis=1) * MM_PER_M
    if options.mLabel in ("rob", "fused"):
        # A: no IK mask / grasp-displacement gate (IK stats live in ik_v1 + ik_qa_stats.py)
        tIkMm = np.zeros(n)
        tDisp = np.zeros(n)
    tJ = np.load(take / asg["cam_of_arm"][a] / JAW_FILE[a])
    tStateBin = tJ["state"][nearest_frame((tJ["t_ns"].astype(np.int64) - lo) / NS_PER_S, tg)[0]].astype(int)
    tGrasp = np.zeros(n, bool)
    tHalf = int(JAW_WIN_S / dtSrc)
    for tC in np.flatnonzero(np.diff(tStateBin) != 0) + 1:
        tGrasp[max(tC - tHalf, 0): tC + tHalf + 1] = True                     # both edges (open + close)
    tGfail = np.zeros(n, bool)
    tGw = np.ones(n, np.float32)
    tInfo = []
    for tX, tY in runs(tGrasp):
        tMax = float(tDisp[tX:tY].max())
        tC = (tX + tY) // 2                                                 # the jaw-close instant (window = +-0.5 s around it)
        # filled run CONTAINING the close
        tLongest = max([(tE - tB) * dtSrc for tB, tE in runs(~tMeas) if tB <= tC < tE] or [0.0])
        tInfo.append(dict(max_mm=round(tMax), fill_overlap_s=round(tLongest, 2)))
        if tMax > GRASP_WARN_MM:
            tGfail[tX:tY] = True
        elif tMax >= GRASP_OK_MM:
            tGw[tX:tY] = GRASP_W
    tTj = (tJ["t_ns"].astype(np.int64) - lo) / NS_PER_S
    tKj, tRj = nearest_frame(tTj, tg)
    tG = rampedJaw(tJ, tTj, tg, nearest_frame)
    tJawValid = (tJ["valid"][tKj] > 0) & (tRj <= MAX_JAW_GAP_S)
    tIkOk = (tIkMm <= options.mIkFailMm) & (~tGfail)
    tPw = posWeights(tGl, tMeas, tGrasp)
    # MAIN track: marker-free consistency confidence -> position weight
    if options.mConfDir:
        tTcf, tCf = loadPosConf(options.mConfDir, take.name, str(tZ["source_rig_label"]), a)
        tKc, tRc = nearest_frame((tTcf + camShiftNs - lo) / NS_PER_S, tg)
        tPw[CONF_VARIANT] = np.where(tRc <= NEAREST_MAX_S, confWeight(tCf[tKc], options), 0.0)
    tPwv = {tV: tW * tIkOk for tV, tW in tPw.items()}
    tStitched = options.mMaskMode == "stitched"
    out.update({f"T_{a}": tT, f"Ph_{a}": tPh, f"meas_{s}": tMeas, f"fill_{s}": tFill, f"ikmm_{s}": tIkMm, f"grasp_{s}": tGrasp,
                f"gfail_{s}": tGfail,
                f"gw_{s}": tGw, f"ikwarn_{s}": (tIkMm > options.mIkFailMm) & (tIkMm <= IK_SEVERE_MM),
                f"iksev_{s}": tIkMm > IK_SEVERE_MM,
                f"g_{a}": tG, f"gc_{a}": np.clip(np.interp(tg, tTj, tJ["g_cont"].astype(float)), 0, 1), f"jv_{s}": tJawValid,
                f"ginfo_{s}": tInfo,
                f"disp_{s}": tDisp,
                # IMU-filled rotation stays a target
                f"rot_ok_{s}": (tIkMm <= options.mIkFailMm) & ((~tGfail) if not tStitched else True),
                f"pos_ok_{s}": ((~tFill) & (tIkMm <= options.mIkFailMm) & (~tGfail)) if not tStitched
                else (tIkMm <= options.mIkFailMm),
                f"gl_{s}": tGl, f"conf_{s}": (tGl <= CONF_GAP_S) & (tRz <= NEAREST_MAX_S),
                **{f"pw_{tV}_{s}": tPwv[tV] for tV in options.mVariants}})


def _rowsOfNext(col, nx, width):
    return np.repeat(col[nx, None], width, 1)


def build(take, dtake, asg, demo, win, camShiftNs: int = 0, options: UnsplitOptions = UnsplitOptions()) -> dict:
    """one demo over its trimmed window `win` (ns) -> per-frame columns; camShiftNs shifts the Orbbec (pose + colour) time"""
    tIK = np.load(dtake / "ik_v1" / f"demo{demo.mIdx:02d}.npz")
    tTc = np.load(take / "t_ns.npy").astype(np.int64) + camShiftNs
    tTi = tIK["t_ns"].astype(np.int64) + camShiftNs
    tLo = max(win[0], int(tTi[0]), int(tTc[0]))
    tHi = min(win[1], int(tTi[-1]), int(tTc[-1]))
    if tHi - tLo < MIN_WINDOW_NS:
        raise ValueError(f"trimmed window inside the recording only {(tHi - tLo) / NS_PER_S:.1f} s")
    tDtSrc = ACTION_DT_S * options.mSpeed
    tGrid = tLo + np.round(np.arange(int((tHi - tLo) / NS_PER_S / tDtSrc) + 1) * tDtSrc * NS_PER_S).astype(np.int64)
    tN = len(tGrid)
    tTg = (tGrid - tLo) / NS_PER_S
    tTis = (tTi - tLo) / NS_PER_S
    tKi, _ = nearest_frame(tTis, tTg)
    tOut = dict(t_ns=tGrid, n=tN, gap_m=tIK["gap"][tKi])
    for tA, tS in ARMS:
        _armColumns(tOut, take, dtake, asg, tA, tS, tIK, tLo, tTis, tKi, tTg, tN, tDtSrc, camShiftNs, options)
    for tC, tTt in (("global", tTc), ("left_wrist", np.load(take / asg["cam_of_arm"]["left"] / "t_ns.npy").astype(np.int64)),
                    ("right_wrist", np.load(take / asg["cam_of_arm"]["right"] / "t_ns.npy").astype(np.int64))):
        tIdx, tRes = nearest_frame(tTt.astype(np.float64), tGrid.astype(np.float64))
        tOut[f"img_{tC}"] = tIdx
        tOut[f"res_{tC}"] = tRes / MS_PER_NS
    tOut["cam_dir"] = {"global": take, "left_wrist": take / asg["cam_of_arm"]["left"],
                       "right_wrist": take / asg["cam_of_arm"]["right"]}
    tNx = np.minimum(np.arange(tN) + 1, tN - 1)
    # 10-23 spec: gaps mask only POSITION (xyz); rotation (IMU) and gripper (wrist-camera jaw) stay supervised
    tOut["action_mask"] = np.concatenate([_rowsOfNext(tOut["pos_ok_L"], tNx, XYZ_DIMS),
                                          _rowsOfNext(tOut["rot_ok_L"], tNx, ROT6D_DIMS),
                                          _rowsOfNext(tOut["pos_ok_R"], tNx, XYZ_DIMS),
                                          _rowsOfNext(tOut["rot_ok_R"], tNx, ROT6D_DIMS),
                                          tOut["jv_L"][tNx, None], tOut["jv_R"][tNx, None]], 1).astype(np.float32)
    tOut["anchor_ok"] = np.stack([tOut["pos_ok_L"], tOut["rot_ok_L"], tOut["pos_ok_R"], tOut["rot_ok_R"]], 1).astype(np.float32)
    for tV in options.mVariants:
        tOut[f"aw_{tV}"] = np.concatenate([_rowsOfNext(tOut[f"pw_{tV}_L"], tNx, XYZ_DIMS),
                                           _rowsOfNext(tOut["rot_ok_L"], tNx, ROT6D_DIMS),
                                           _rowsOfNext(tOut[f"pw_{tV}_R"], tNx, XYZ_DIMS),
                                           _rowsOfNext(tOut["rot_ok_R"], tNx, ROT6D_DIMS),
                                           tOut["jv_L"][tNx, None], tOut["jv_R"][tNx, None]], 1).astype(np.float32)
        tAnchorW = [tOut[f"pw_{tV}_L"], tOut["rot_ok_L"], tOut[f"pw_{tV}_R"], tOut["rot_ok_R"]]
        tOut[f"anw_{tV}"] = np.stack(tAnchorW, 1).astype(np.float32)
    tOut["weight"] = np.minimum(tOut["gw_L"], tOut["gw_R"]) if options.mMaskMode != "stitched" else np.ones(tN, np.float32)
    # 10-30: EP_WEIGHT env = per-sample loss weight of every frame of this build (new motor-jaw flattening mixed into GOLD-281 at
    # 2x)
    tOut["weight"] = (tOut["weight"] * float(os.environ.get("EP_WEIGHT", "1"))).astype(np.float32)
    return tOut
