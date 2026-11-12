"""One run on a started Rig (dry or --go): the receding-horizon loop -- observe, infer, re-anchor, limiter -> IK -> safety,
commit -- and the ACT temporal ensemble (deploy stage). The stages are described in the docstring of session.cli (= its --help).

Inputs: parsed `run` arguments, a started Rig, a run control (poll / publish / optional takeStep). Outputs: commands to the
bridge (--go only), reports/act_deploy/<run>/ (summary.json, log.npz, lookahead.json), the summary dict.
"""

from __future__ import annotations
import collections
import json
import time

import numpy as np

from ..core import height_calib as heightCalib
from ..core.contact import ContactFloor
from ..core.chunk import buildState, decodeChunk, retime, rtcWeights
from ..core.contract import (
    ACT_CKPT,
    ACTION_DIM,
    CHUNK_ROWS,
    DIT_CKPT,
    GRIP_COLUMNS,
    GRIP_INDEX,
    NS_PER_S,
    POSE_SLICE,
    ROW_DT_S,
    SIDES,
)
from ..core.errors import DeployError, HardwareError, RunRefused
from ..core.kinematics import fk, linkGap, rotAngleDeg, rotInterp
from ..core.label_runner import LabelRunner
from ..core.pgrasp import VERSION as PGRASP_VERSION
from ..core.xvla_runner import XvlaRunner
from laundry_shared.contracts.action_contract import T_to_pose9, inv_T
from .bridge import checkGripLimits
from .paths import HIST_TOL_S, LATENCY_FILE, RUN_OUT_DIR, START_FILE, say
from .preview import Preview
from .report import summarize
from .streamer import Streamer

POLICY_DIT, POLICY_LABEL = "dit", "label"
KIND_DIT, KIND_XVLA = "multi_task_dit", "xvla"
DEFAULT_COMMIT_ACT = 1
DEFAULT_COMMIT_DIT = 4  # DiT and label replay
LABEL_SECS_FACTOR, LABEL_SECS_PAD_S = 2.0, 10.0  # cursor mode ends at the label end; secs = only a safety cap
Z_RAMP_ROWS = 16.0  # the first committed chunk ramps a z correction in over this many rows
MIN_SPEED = 0.05  # floor of the speed in the look-ahead row period
LATENCY_HISTORY = 5  # recent plan latencies kept for the look-ahead re-plan point
RTC_DEFAULT_DELAY = 2  # RTC rows executed during inference before any latency was measured
RTC_BETA_DEFAULT = 5.0
RTC_JAW_KEEP_TICKS = 64  # committed jaw rows kept behind the anchor tick
WARMUP_PREDICTIONS = 3  # MPS warm-up (first call ~1.4 s DiT / 0.2 s ACT); nothing is used
START_POSE_TOL_DEG = 5  # "at the start pose" = every joint within this of start_pose.json
UNPARK_WAIT_S = 3.0
POLL_S = 0.05
PGRASP_START_TIMEOUT_S = 10.0  # the p_grasp estimator must give its first value within this (classifier load + first tick)
PGRASP_POLL_S = 0.05
GRIP_AT_ROW = "row"  # 11-06: --grip-at row, the jaw command follows each committed chunk row
RECOVER_WAIT_S = 1.0
STALE_IMAGE_SKIP_S = 0.05
RESYNC_HOLD_S = 0.3
GLOBAL_HIST_EXTRA_TOL_S = 0.04  # the 15-30 Hz Orbbec gets a wider history window than the robot / wrists
WARN_SLOW_FACTOR = 0.5  # speed factor of the plan after a link-gap warning
GRIP_MID = 0.5
SIGMOID_GAIN_DEFAULT, RELU_GAIN_DEFAULT = 10.0, 2.0
LAT_EMA_KEEP, LAT_EMA_GAIN = 0.8, 0.2
SOFT_STOP_SETTLE_S = 0.8
SMOOTHNESS_ROWS = (1, 8, 16, 32)
SOFT_STOP_PREFIXES = ("정지", "max time", "label end", "Ctrl+C")
NO_SAFETY_DEMOTED = ("STOP:arm_collision", "STOP:link_table", "STOP:joint_limit", "STOP:workspace", "STOP:tcp_table")
BLOCKING = ("STOP", "HOLD")
MIN_BRIDGE_TIMEOUT_S = 0.3
MM_PER_M = 1e3
MS_PER_S = 1e3
LIMIT_TOUCH_RAD = np.radians(0.2)  # a joint this close to its solver limit counts as pinned (ik_hold)
MIN_ELAPSED_S = 1e-3  # plan-rate denominator floor of the first cycle
NS_PER_MS = 1e6


def scaleTransform(D: np.ndarray, s: float) -> np.ndarray:
    """Relative transform D scaled by s: translation x s, rotation angle x s (same axis). Was scale_T."""
    from scipy.spatial.transform import Rotation

    tOut = np.eye(4)
    tOut[:3, :3] = Rotation.from_rotvec(Rotation.from_matrix(D[:3, :3]).as_rotvec() * s).as_matrix()
    tOut[:3, 3] = D[:3, 3] * s
    return tOut


def checkRunArgs(args) -> None:
    """Fills the default --commit and refuses invalid / unsafe combinations (RunRefused)."""
    tDit = args.policy == POLICY_DIT
    if args.commit is None:
        args.commit = DEFAULT_COMMIT_DIT if (tDit or args.policy == POLICY_LABEL) else DEFAULT_COMMIT_ACT
    if args.policy == POLICY_LABEL and not getattr(args, "label", ""):
        raise RunRefused("--policy label 에는 --label <ik_v1/demoNN.npz> 필요")
    if not 1 <= args.commit <= CHUNK_ROWS:  # 10-23 user: up to the full 32-step chunk
        raise RunRefused(f"--commit {args.commit} 불가 (1..32 = the whole chunk)")
    # 11-01 user "지금 너무 느린데": the S03/S05 sets are time-stretched, so speed > 1 is allowed; retime() then reaches the chunk end
    # after 32 / speed rows and the rest would hold the last pose -> commit at most that many rows
    if tDit and args.speed > 1.0 and args.commit > int(CHUNK_ROWS / args.speed):
        say(f"속도 x{args.speed}: commit {args.commit} -> {int(CHUNK_ROWS / args.speed)} (청크 끝을 넘지 않게)")
        args.commit = int(CHUNK_ROWS / args.speed)
    # 10-23 user: commit 8 allowed with --go after the first GO at commit 1 drifted (ACT is trained to execute whole chunks)
    if args.go and (args.assume_ready or args.assume_start):
        raise RunRefused("--assume-ready / --assume-start 는 dry-run 전용")
    if args.go and args.replay_global:
        raise RunRefused("--replay-global (녹화 프레임) 은 테스트 전용 -- --go 금지")


def ensemble(ens: list, tTarget: float, side: str, m: float) -> np.ndarray:
    """ACT temporal ensemble on ABSOLUTE base-frame targets: every buffered chunk's step nearest tTarget, weights exp(-m * i),
    i = 0 oldest."""
    from scipy.spatial.transform import Rotation

    tP, tR, tW = [], [], []
    for tTp, tTc in ens:
        tK = int(round((tTarget - tTp) / ROW_DT_S)) - 1
        if 0 <= tK < CHUNK_ROWS:
            tP.append(tTc[side][tK][:3, 3])
            tR.append(tTc[side][tK][:3, :3])
            tW.append(np.exp(-m * len(tW)))
    tW = np.array(tW) / np.sum(tW)
    tT = np.eye(4)
    tT[:3, 3] = np.sum(np.array(tP) * tW[:, None], 0)
    tT[:3, :3] = Rotation.from_matrix(np.array(tR)).mean(weights=tW).as_matrix()
    return tT


def jawCommand(args, gripPred: dict, active: list, gCmd: dict) -> dict:
    """Predicted P(open) of the committed segment -> jaw command per arm; a dry arm keeps its last command.
    10-30 user "그냥 continuous로 해보자 gripper": --grip-continuous sends P(open) (0..1) instead of the 0/1 threshold (the left jaw
    kept stopping half-way while the binary command flipped every cycle); also a SIGMOID jaw 1 / (1 + exp(-gain (p - 0.5))) and a
    ReLU jaw clip(gain (p - 0.5), 0, 1): closed below 0.5, ramps open above."""
    if getattr(args, "grip_mode", "") == "sigmoid":
        tGain = float(getattr(args, "grip_gain", SIGMOID_GAIN_DEFAULT))
        return {
            tSide: float(1.0 / (1.0 + np.exp(-tGain * (float(gripPred[tSide]) - GRIP_MID)))) if tSide in active else gCmd[tSide]
            for tSide in SIDES
        }
    if getattr(args, "grip_mode", "") == "relu":
        tGain = float(getattr(args, "grip_gain", RELU_GAIN_DEFAULT))
        return {
            tSide: float(np.clip(tGain * (float(gripPred[tSide]) - GRIP_MID), 0.0, 1.0)) if tSide in active else gCmd[tSide]
            for tSide in SIDES
        }
    if getattr(args, "grip_continuous", False):
        return {tSide: float(np.clip(gripPred[tSide], 0.0, 1.0)) if tSide in active else gCmd[tSide] for tSide in SIDES}
    return {tSide: (1.0 if gripPred[tSide] > args.grip_thr else 0.0) if tSide in active else gCmd[tSide] for tSide in SIDES}


def relaxFlags(flags: list, args, active: list) -> list:
    """--no-safety: collision / table / joint-limit / workspace STOPs become WARN (logged only, 10-27 label replay). One-arm runs:
    the dry arm's own STOP / HOLD flags become DRY:... (they never block the active arm); arm-arm collision always blocks."""
    if getattr(args, "no_safety", False):
        flags = [("WARN:" + tF[5:]) if tF.startswith(NO_SAFETY_DEMOTED) else tF for tF in flags]
    if len(active) < 2:
        tDrySides = [tX[0] for tX in SIDES if tX not in active]
        flags = [
            ("DRY:" + tF.split(":", 1)[1])
            if any(tF.endswith(f"_{tC}") or f"_{tC}_" in tF for tC in tDrySides)
            and tF.startswith(BLOCKING)
            and not tF.startswith("STOP:arm_collision")
            else tF
            for tF in flags
        ]
    return flags


def _limitAndSolve(side: str, args, Tn, cand, spd: float, rowDt: float, j: int, arm, gNew: float, ws, qPrev, flags: list):
    """One committed step of one active arm: speed scale -> Cartesian clamps -> finger-corner floor -> workspace -> IK (closest
    reachable pose) -> joint-speed limiter -> joint limits. Appends flags in that order (the caller adds the TCP-above-table check).
    -> (limited target, IK joints, accepted joints, IK error mm, IK error deg)"""
    tT = Tn.copy()
    tTarget = cand
    tT[:3, 3] = Tn[:3, 3] + spd * (tTarget[:3, 3] - Tn[:3, 3])
    tT[:3, :3] = rotInterp(Tn[:3, :3], tTarget[:3, :3], spd)
    tDp = tT[:3, 3] - Tn[:3, 3]
    tMaxStep = args.max_tcp_mps * rowDt * j
    if np.linalg.norm(tDp) > tMaxStep:
        tT[:3, 3] = Tn[:3, 3] + tDp * tMaxStep / np.linalg.norm(tDp)
        flags.append(f"LIM:tcp_speed_{side[0]}")
    tAng = rotAngleDeg(Tn[:3, :3], tT[:3, :3])
    tMaxRot = args.max_rot_dps * rowDt * j
    if tAng > tMaxRot:
        tT[:3, :3] = rotInterp(Tn[:3, :3], tT[:3, :3], tMaxRot / tAng)
        flags.append(f"LIM:rot_speed_{side[0]}")
    tLift = (
        0.0 if getattr(args, "no_floor", False) else arm.cornerLift(tT, gNew)
    )  # 10-29 user "바닥 제한을 없애줘": UI GO sets no_floor
    if tLift > 0:
        tT[2, 3] += tLift
        flags.append(f"LIM:corner_lift_{side[0]}_{tLift * MM_PER_M:.0f}mm")
    if np.any(tT[:3, 3] < ws[0]) or np.any(tT[:3, 3] > ws[1]):
        flags.append(f"STOP:workspace_{side[0]}")
    tQ, tPe, tRe = arm.mFree.ik(qPrev, tT)
    tIkMm, tIkDeg = tPe * MM_PER_M, tRe
    if not np.isfinite(tQ).all():
        flags.append(f"STOP:ik_nan_{side[0]}")
    # 11-07 user "ik 관절값이 안되는 곳을 가는거 방어": a target that needs a joint past its limit (IK pinned at a solver limit AND far
    # from the target) holds THIS arm at its last accepted joints instead of going to a twisted closest pose; the other arm runs on
    tAtLim = np.any((tQ <= arm.mLo + arm.mLimMargin + LIMIT_TOUCH_RAD) | (tQ >= arm.mHi - arm.mLimMargin - LIMIT_TOUCH_RAD))
    if tAtLim and (tPe > args.ik_hold_mm / MM_PER_M or tRe > args.ik_hold_deg):
        flags.append(f"LIM:ik_hold_{side[0]}_{tIkMm:.0f}mm_{tRe:.0f}deg")
        return tT, tQ, qPrev.copy(), tIkMm, tIkDeg
    # 10-23 user: like the robotized labels (FK of the feasible IK path), go to the closest reachable pose; STOP only if far off
    if tPe > args.max_ik_stop_mm / MM_PER_M:
        flags.append(f"STOP:ik_{side[0]}_{tPe * MM_PER_M:.0f}mm_{tRe:.0f}deg")
    elif tPe > args.max_ik_mm / MM_PER_M or tRe > args.max_ik_deg:
        flags.append(f"LIM:ik_closest_{side[0]}_{tPe * MM_PER_M:.0f}mm_{tRe:.0f}deg")
    tDq = tQ - qPrev
    tLimRad = np.radians(args.max_deg_s) * rowDt
    tMaxDq = float(np.abs(tDq).max())
    if tMaxDq > tLimRad:
        tDq = tDq * tLimRad / tMaxDq
        flags.append(f"LIM:joint_speed_{side[0]}")
    tQa = qPrev + tDq
    if np.any(tQa < arm.mLo) or np.any(tQa > arm.mHi):
        flags.append(f"STOP:joint_limit_{side[0]}")
    return tT, tQ, tQa, tIkMm, tIkDeg


def runSession(args, rig, ctl):
    """ONE run (dry or --go) on a started Rig.  ctl.poll() -> stop reason or None (keys / UI button / heartbeat watchdog);
    ctl.publish(snapshot) every cycle (UI).  --go: a live HttpBridge (single writer) exists only inside this function; any stop
    -> streamer stopped + bridge estop (holds the measured pose), operator stops brake softly (11-01)."""
    checkRunArgs(args)
    tDit = args.policy == POLICY_DIT
    tLabel = args.policy == POLICY_LABEL
    # 10-28 user: step mode (GO only) -- the policy keeps planning and the UI shows each plan, but a plan is sent ONLY when the UI
    # [한 스텝 실행] button set a step token (ctl.takeStep); one token = one committed segment (args.commit steps), then hold again.
    # A plan with STOP/HOLD flags is never sent; in step mode it does not end the run (the token is consumed and the step
    # refused).
    tStep = bool(getattr(args, "step_mode", False)) and args.go
    # 10-28 user: one-arm execution -- the other arm's plan is computed and shown (dry) but its joints/jaw stay at the last
    # command;
    # its own safety flags become DRY:... (logged, never block the active arm); arm-arm collision is checked with the held pose.
    tActive = [tX for tX in str(getattr(args, "active_arms", "both")).split(",") if tX in SIDES] or list(SIDES)
    if len(tActive) < 2:
        say(f"한 팔 실행: {tActive} 만 움직임, 나머지는 계획만 표시 (DRY)")
    tTakeStep = getattr(ctl, "takeStep", lambda: False)
    from laundry_shared.visual_teleop.robot.bridge import HttpBridge
    from laundry_shared.visual_teleop.tools.robot_moves import unpark

    tArms, tRobotArgs, tCol, tPoller = rig.mArms, rig.mRobotArgs, rig.mCollision, rig.mPoller
    tGlobalCam, tWristCams, tQReady, tCamsCfg = rig.mGlobalCam, rig.mWristCams, rig.mQReady, rig.mCamsCfg
    for tSide in SIDES:  # 10-23 user: let the fingers reach / scrape the table (umi replay 10-21 used --tip-clear -0.01)
        tArms[tSide].mArgs.tip_clear = args.tip_clear
        tArms[tSide].mArgs.tip_clear_left_extra = 0.0
    if args.go:
        if not rig.mWaveOk:
            raise HardwareError("--go 전에 wave 테스트로 손목 카메라 좌우를 확인하세요")
        if rig.mReplay or rig.mFakeQ is not None:
            raise RunRefused("--go 는 라이브 Orbbec + 실제 로봇 상태에서만")
    tMock = rig.mUrl.rstrip("/") != rig.mTeleopCfg.robot["base_url"].rstrip("/")  # e.g. the MIT_MOCK bridge on :8031
    tRunName = ("mock_" if tMock else "") + f"{args.policy}_" + ("go_" if args.go else "dry_") + time.strftime("%Y%m%d_%H%M%S")
    tRunDir = RUN_OUT_DIR / tRunName
    tRunDir.mkdir(parents=True, exist_ok=True)
    tCkpt = args.label if tLabel else (args.ckpt or str(DIT_CKPT if tDit else ACT_CKPT))
    say(
        f"{args.policy.upper()} {tCkpt} | {'실행 (--go)' if args.go else 'DRY-RUN (명령 안 보냄)'}  commit {args.commit}, speed {args.speed} "
        f"(처음 {args.slow_start_s}s 는 {args.slow_start_speed}), 최대 {args.secs}s -> {tRunDir}"
    )
    if tLabel:  # 10-27 label replay (A' vs Cg): the label as an oracle policy, same loop
        tRunner = LabelRunner(args.label, args.label_rate)
        args.secs = tRunner.mDurS / args.label_rate * LABEL_SECS_FACTOR + LABEL_SECS_PAD_S
        say(f"라벨 replay: {args.label} ({tRunner.mDurS:.1f}s 라벨, 속도 x{args.label_rate} -> {args.secs:.0f}s)")
    else:
        tRunner = rig.runner(tCkpt, args.device, args.dit_samples)
    if hasattr(tRunner, "mGripFakeOpen"):  # 10-29 diagnostic: jaw state fed as OPEN (copycat checkpoints never open on their own)
        tRunner.mGripFakeOpen = bool(getattr(args, "grip_fake_open", False))
        if tRunner.mGripFakeOpen:
            say("진단: 상태 입력의 집게를 항상 '열림'으로 넣음 (집게 명령은 청크 중간 행)")
        if getattr(tRunner, "mGripMask", False) and not getattr(tRunner, "mPGrasp", None):
            say("집게 상태 가림 체크포인트: 상태 입력의 집게 = 0")
    # 11-06 FLAT-PGRASP checkpoints: the jaw slots of the state (18, 19) = robot-side p_grasp at the observation time (and at t - 50 ms
    # for the DiT's older obs step), same frozen classifier / rule as the training side file (core.pgrasp)
    tPGrasp = None
    if getattr(tRunner, "mPGrasp", None):
        if tRunner.mPGrasp != PGRASP_VERSION:
            raise RunRefused(f"체크포인트 p_grasp 버전 {tRunner.mPGrasp} != 배포 {PGRASP_VERSION}")
        tPGrasp = rig.pGrasp(args.device)
        tT0 = time.monotonic()
        while tPGrasp.at() is None:
            if time.monotonic() - tT0 > PGRASP_START_TIMEOUT_S:
                raise HardwareError(f"p_grasp 추정기가 값을 못 냄: {tPGrasp.mError}")
            time.sleep(PGRASP_POLL_S)
        tP0 = tPGrasp.at()
        tSrc = "집게 개폐 (닫힘 = 1, 열림 = 0)" if tPGrasp.mSource == "jaw" else "로봇 손목 카메라 집게 판정"
        say(f"p_grasp 입력 ({PGRASP_VERSION}, {tPGrasp.mSource}): state 18,19 = {tSrc} (지금 L {tP0[0]:.2f} R {tP0[1]:.2f})")
    tKind = getattr(tRunner, "mKind", "")
    tXvla = tKind == KIND_XVLA
    tRetimed = tXvla or tDit  # 10-29: X-VLA, then DiT (user "dit에도 적용해줘")
    if tXvla:  # 10-29: X-VLA speed = time scale along the predicted path (amplitude scale stays 1)
        tRunner.mTimeScale = float(args.speed)
        say(f"X-VLA: 15 Hz 청크를 20 Hz 실행 행으로 재샘플, 속도 x{args.speed} = 시간 배율 (경로는 그대로)")
    elif tDit:
        say(f"DiT: 속도 x{args.speed} = 시간 배율 (청크 경로를 그대로 천천히)")
    tAmp = float(getattr(args, "amp", 1.0))
    if tAmp != 1.0:
        say(f"움직임 크기 x{tAmp}: 커밋한 각 스텝의 이동(위치 + 회전)을 현재 자세 기준으로 줄임")
    # 10-29 indy7 c4fba14 look-ahead (continuous GO of X-VLA / DiT): the next forward starts while the current chunk still runs,
    # the
    # chunk is anchored on the COMMANDED pose at the observation tick and merged into the streamer's indexed timeline
    # (Streamer.merge)
    # 10-30 user (backstep at every chunk switch, 37/40 cycles -40 mm): --no-lookahead = run the whole commit, THEN observe +
    # infer
    # 10-30 user "z offset": height correction of the WHOLE trajectory -- added once to the first committed chunk (ramped over 16
    # rows);
    # later chunks chain on the commanded pose (q_acc), so it neither accumulates nor fades.  Not a limit: any value, 0 = off.
    tZOffM = float(getattr(args, "z_offset_mm", 0.0)) / MM_PER_M
    # 11-06: optional floor (mm above the robot table plane, + the height-calibration plane when on) for every committed TCP target;
    # None = off. The policy's chunks reached 10 cm below the table (training labels: 1st percentile -3 mm)
    tZFloorMm = getattr(args, "z_floor_mm", None)
    tZFloorM = None if tZFloorMm is None or tZFloorMm == "" else float(tZFloorMm) / MM_PER_M
    if tZFloorM is not None:
        say(f"바닥 제한: 집게 TCP 목표 높이 >= 테이블 + {tZFloorM * MM_PER_M:.0f} mm (로봇 좌표)")
    tZOffDone = tZOffM == 0.0
    # 11-01 user "로봇에 맞추어서": --height-calib adds the per-arm table-height error measured by tools.height_calib (robot jaw tip on
    # the table tags vs the training B frame), on top of any manual z offset.  The measured error is a PLANE (left 5 touches: tilt
    # ~4.9 deg, +3 mm near the base .. -24 mm far out), so it follows the TCP: every committed row gets dz(row xy) minus what the
    # commanded pose already carries (tZBase); the first chunk ramps in over 16 rows like the manual offset
    tCalib = None
    tZOff = {tSide: tZOffM for tSide in SIDES}
    tZBase = {tSide: 0.0 for tSide in SIDES}
    if getattr(args, "height_calib", False):
        tCalib = heightCalib.loadCalibration()
        tZOffDone = False
        if tCalib is None:
            raise RunRefused("deploy/height_calib.json 없음 -- 먼저 `python deploy/height_calib.py measure`")
        say(
            "높이 캘리브레이션 "
            + tCalib["created"]
            + ": "
            + ", ".join(f"{tS} {tF['dz_mean_mm']:+.1f} mm" for tS, tF in tCalib["fit"].items())
        )
    tLook = tRetimed and args.go and not tStep and not getattr(args, "no_lookahead", False)
    tLatHist = collections.deque(maxlen=LATENCY_HISTORY)
    tDropped = 0
    tBlended = 0
    # 10-29 user "Real time chunking도 같이 구현해줘": look-ahead only (rows keyed by tick).  The previous plan = the COMMANDED timeline
    # (after limiter / IK) re-anchored at this observation's tick; jaws from the committed rows.  guided = RTC guidance (DiT, one
    # VJP per
    # DDIM step), blend = gradient-free clean-chunk inpainting (X-VLA always: a VJP through the 0.9B model would triple its ~950
    # ms).
    tRtcMode = getattr(args, "rtc", "off") if tLook and tKind in (KIND_DIT, KIND_XVLA) else "off"
    if tRtcMode != "off" and tKind == KIND_XVLA:
        tRtcMode = "blend"
    tRtcJaw, tRtcLog = {}, []
    if tRtcMode != "off":
        say(f"Real-Time Chunking: {tRtcMode} (beta {getattr(args, 'rtc_beta', RTC_BETA_DEFAULT)})")
    tRowDt = ROW_DT_S  # robot seconds per executed chunk row (limiter caps, pacing)
    if tLook:
        # the model's own rows are executed (no resampling): one tick = row period / speed (X-VLA 1/15 s, DiT 1/20 s at speed 1)
        # -> the
        # 32-row chunk spans 2.13 s / 1.6 s at speed 1, half of it covers one forward (X-VLA ~0.9 s on MPS, DiT 0.2-0.45 s)
        tRowDt = (XvlaRunner.ROW_DT_S if tXvla else ROW_DT_S) / max(float(args.speed), MIN_SPEED)
        if tXvla:
            tRunner.mTimeScale = None  # XvlaRunner.predict: raw 15 Hz rows
        say(
            f"look-ahead (indy7 / LeRobot async: 남은 행이 청크 절반 이하면 다음 추론, 겹치는 행 섞음): 행 간격 {tRowDt * MS_PER_S:.0f} ms (속도 x{args.speed})"
        )
    tWs = tRunner.workspace(args.ws_margin)
    if tDit and getattr(args, "dit_steps", None):
        tRunner.setDitSteps(int(args.dit_steps))
    tQ0, tGObs0, tEst = tPoller.read()
    if tEst:
        if args.go:
            raise HardwareError("로봇 상태 E-STOP/미연결/읽기 지연 -- --go 불가 (브리지 /status 의 read_age_ms 확인)")
        raise HardwareError("로봇 상태 E-STOP/미연결/읽기 지연 (읽기 지연 = 로봇 꺼짐/USB). 로봇 없이 확인만: --assume-ready")
    tReadyOff = {tSide: float(np.degrees(np.abs(tQ0[tSide] - tQReady[tSide])).max()) for tSide in SIDES}
    tStartPose = json.loads(START_FILE.read_text()) if START_FILE.exists() else None
    # 11-06 UMI latency matching (LATENCY_FILE from tools.latency_probe / tools.cam_latency): arm_ms / jaw_ms = send-ahead of the
    # streamer, cam_ms.global = the head image's capture delay -> state, history and chunk anchor are taken at the capture time
    tLat = {}
    if getattr(args, "latency_match", False):
        if LATENCY_FILE.exists():
            tLat = json.loads(LATENCY_FILE.read_text())
            say(f"지연 보정: 팔 {tLat.get('arm_ms')} ms, 집게 {tLat.get('jaw_ms')} ms, 카메라 {tLat.get('cam_ms')} ms 앞당김")
        else:
            say("지연 보정 켜짐, 하지만 측정값 없음 (tools.latency_probe 먼저) -- 보정 없이 진행")
    tCamLagNs = int(float((tLat.get("cam_ms") or {}).get("global", 0.0)) * NS_PER_MS)
    tStartOff = (
        {tSide: float(np.degrees(np.abs(tQ0[tSide] - np.asarray(tStartPose["q"][tSide]))).max()) for tSide in SIDES}
        if tStartPose
        else None
    )
    tAtStart = bool(tStartPose and max(tStartOff.values()) <= START_POSE_TOL_DEG)
    if max(tReadyOff.values()) > args.ready_tol_deg and not (args.allow_start_pose and tAtStart):
        if args.go:
            raise RunRefused(
                f"두 팔이 READY 가 아님 (최대 관절 차이 {tReadyOff}){' / 시작 자세 차이 ' + str(tStartOff) if tStartPose else ''} "
                f"-- 먼저 READY (또는 `start --go` 후 --allow-start-pose)"
            )
        say(f"주의: READY 가 아님 (관절 차이 {tReadyOff} deg) -- dry-run 은 계속")
    if tAtStart:
        say(f"시작 자세 (start_pose.json, 학습 에피소드 {tStartPose['episode']}) 에서 시작")
    tPreview = Preview(tArms, tCol, tWs, args, tRobotArgs.table_z)
    tPreview.start()
    tWarmState = buildState(
        *(fk(tArms, tSide, tQ0[tSide]) for tSide in SIDES), *(fk(tArms, tSide, tQ0[tSide]) for tSide in SIDES), 0.0, 0.0
    )
    tWarmImgs = {
        "global": tGlobalCam.mRing.latest()[1][0],
        "left_wrist": tWristCams["left"].mRing.latest()[1],
        "right_wrist": tWristCams["right"].mRing.latest()[1],
    }
    for _ in range(WARMUP_PREDICTIONS):
        tRunner.predict(tRunner.item(tWarmImgs, tWarmState, None))
    # ---- start state
    tT0 = {tSide: fk(tArms, tSide, tQ0[tSide]) for tSide in SIDES}
    tAnchor = {tSide: tT0[tSide].copy() for tSide in SIDES}
    tGCmd = {
        tSide: float(args.init_grip if args.init_grip >= 0 else (round(tGObs0[tSide]) if tGObs0[tSide] is not None else 0.0))
        for tSide in SIDES
    }
    tQAcc = {tSide: tQ0[tSide].copy() for tSide in SIDES}
    tStreamer = None
    tContact = None
    tLiveBridge = None

    def floorBase(side, xy):  # 11-06: the static floor at a TCP xy (robot table + z_floor + height-calibration plane)
        return tRobotArgs.table_z + (tZFloorM or 0.0) + (heightCalib.dzAt(tCalib, side, xy) if tCalib is not None else 0.0)

    tLog = collections.defaultdict(list)
    tEns = []
    tStopReason = None
    tViolations = 0
    tCycle = 0
    tLatEma = None
    tPrevTc = None
    tWarnSlow = False
    tResyncs = 0
    tRecoveries = 0
    try:
        if args.go:
            checkGripLimits(rig.mUrl)
            tLiveBridge = HttpBridge(
                rig.mUrl, live=True, timeout_s=max(rig.mTeleopCfg.robot["request_timeout_s"], MIN_BRIDGE_TIMEOUT_S)
            )
            say("두 팔 토크 켬 (unpark), 3초 뒤 시작. 멈춤 = 스페이스")
            for tSide in SIDES:
                unpark(tLiveBridge, tSide)
            tTu = time.monotonic()
            while time.monotonic() - tTu < UNPARK_WAIT_S:
                tReason = ctl.poll()
                if tReason:
                    raise RunRefused(f"시작 전 정지: {tReason}")
                time.sleep(POLL_S)
            tQ0, _, tEst = tPoller.read()
            if tEst:
                raise HardwareError("unpark 후 E-STOP/읽기 지연")
            # 11-06: the start pose has right j4 ON its lower limit (-79.5 deg); the measured q read 0.005 deg past it and every IK
            # seeded from it warned "Value ... at index 3 is out of limits" -> seed the timeline inside the limits
            tQ0 = {tSide: np.clip(tQ0[tSide], tArms[tSide].mLo + tArms[tSide].mLimMargin, tArms[tSide].mHi - tArms[tSide].mLimMargin)
                   for tSide in SIDES}  # 11-07: inside the (3 deg) solver limits
            tQAcc = {tSide: tQ0[tSide].copy() for tSide in SIDES}
            tStreamer = Streamer(
                tLiveBridge, tQ0, tGCmd, args.stream_hz, dt=tRowDt, gripHoldS=float(getattr(args, "grip_hold_s", 0.0)),
                execLeadS={tSide: tV / MS_PER_S for tSide, tV in (tLat.get("arm_ms") or {}).items()},
                jawLeadS={tSide: tV / MS_PER_S for tSide, tV in (tLat.get("jaw_ms") or {}).items()},
            )
            tPoller.mTauLog, tPoller.mCmdFn = [], (lambda: tStreamer.mLast)  # 11-07: q / torque / commanded q -> qtau.npz
            if getattr(args, "contact_floor", False):
                tContact = ContactFloor(tArms, tPoller, tStreamer, float(getattr(args, "contact_gap_mm", 8.0)),
                                        float(getattr(args, "contact_tau_nm", 2.0)), say)
                tStreamer.mFloorGuard = tContact.below
                tContact.start()
                say(f"바닥 접촉 감지 (토크, 높이 모델 없음): 명령보다 {tContact.mGap * 1e3:.0f} mm 위에서 멈추고 토크 +{tContact.mTau} N·m "
                    "-> 그 자리 높이 아래로 안 감")
            tStreamer.start()
        tStart = time.monotonic()
        if tLabel:
            tRunner.mCursor = 0.0
            tRunner.mT0 = tStart
            (tRunDir / "t_start_monotonic_ns.txt").write_text(str(int(tStart * NS_PER_S)))
        say("시작" + ("" if args.go else " (DRY-RUN: 아무것도 안 보냄)"))
        while True:
            tC0 = time.monotonic()
            tEl = tC0 - tStart
            if args.secs > 0 and tEl > args.secs:  # 10-23 user: secs 0 = no time limit
                tStopReason = f"max time {args.secs:.0f}s"
                break
            tReason = ctl.poll()
            if tReason:
                tStopReason = tReason
                break
            if tStreamer is not None and tStreamer.mError:
                tStopReason = tStreamer.mError
                break
            if tPoller.mError and not args.assume_ready:
                tStopReason = f"robot read: {tPoller.mError}"
                break
            # ---- observe (latest of every ring; DiT: + the samples nearest t_obs - 50 ms)
            tObsNs = time.monotonic_ns()
            _, (tQm, _, tEst) = tPoller.mRing.latest()
            if (tEst or tQm is None) and args.go and args.keep_going:
                # 10-23 user: an arm drive dropping out (left wrist) must not end the run -> wait for the bridge to re-enable it,
                # clear the latch, re-seed both arms on their measured pose and continue
                tRecoveries += 1
                say(f"[{tEl:5.1f}s] 브리지 E-STOP/읽기 지연 -> 복구 대기 ({tRecoveries})")
                time.sleep(RECOVER_WAIT_S)
                try:
                    if tLiveBridge.status().get("estop"):
                        tLiveBridge.s.post(tLiveBridge.base_url + "/clear_estop", timeout=2.0)
                    for tSide in SIDES:
                        unpark(tLiveBridge, tSide)
                except Exception as tExc:
                    say(f"복구 실패, 재시도: {type(tExc).__name__}")
                _, (tQr, _, tEstR) = tPoller.mRing.latest()
                if tQr is not None and not tEstR:
                    tQAcc = {tSide: tQr[tSide].copy() for tSide in SIDES}
                    tStreamer.setTimeline([dict(tQAcc)], ROW_DT_S, tGCmd)
                    say("복구됨 -> 측정 자세에서 계속")
                tCycle += 1
                continue
            if tEst or tQm is None:
                tStopReason = "bridge E-STOP / stale read"
                break
            tGlobalNs, (tGlobalImg, tGlobalJpg) = tGlobalCam.mRing.latest()
            tWristLatest = {tSide: tWristCams[tSide].mRing.latest() for tSide in SIDES}
            tImgAge = [(tObsNs - tGlobalNs) / NS_PER_MS] + [(tObsNs - tWristLatest[tSide][0]) / NS_PER_MS for tSide in SIDES]
            tStaleImg = (
                max(tImgAge) > args.max_img_age_ms and not args.replay_global and args.policy != POLICY_LABEL
            )  # 10-27: label replay uses no images
            if tStaleImg and args.go and not args.keep_going:
                tStopReason = f"stale image {np.round(tImgAge)} ms"
                break
            if tStaleImg and args.go:  # keep_going: skip this plan, hold, re-observe
                time.sleep(STALE_IMAGE_SKIP_S)
                tCycle += 1
                continue
            tImgs = {"global": tGlobalImg, "left_wrist": tWristLatest["left"][1], "right_wrist": tWristLatest["right"][1]}
            # 11-06 UMI: the reference time is the head image's CAPTURE (arrival - measured camera delay); the robot state is the
            # 25 Hz ring sample nearest to it, like the training pairs (state at the image's time), not the newest read
            tRefNs = tObsNs
            if tCamLagNs:
                tRefNs = tGlobalNs - tCamLagNs
                tNearQ = tPoller.mRing.near(tRefNs, HIST_TOL_S)
                if tNearQ is not None and tNearQ[1][0] is not None:
                    tQm = tNearQ[1][0]
            tTm = {tSide: fk(tArms, tSide, tQm[tSide]) for tSide in SIDES}
            tState = buildState(tTm["left"], tTm["right"], tAnchor["left"], tAnchor["right"], tGCmd["left"], tGCmd["right"])
            tPNow = None
            if tPGrasp is not None:
                rig.mJawFallback = dict(tGCmd)
                tPNow = tPGrasp.at(tObsNs)
                tState = np.array(tState, np.float64)
                tState[GRIP_COLUMNS] = tPNow
            tPrevObs, tHistUsed = None, False
            if tDit:
                tHistNs = tRefNs - int(ROW_DT_S * NS_PER_S)
                tHistQ = tPoller.mRing.near(tHistNs, HIST_TOL_S)
                tHistGlobal = tGlobalCam.mRing.near(tHistNs, HIST_TOL_S + GLOBAL_HIST_EXTRA_TOL_S)
                tHistState = None
                if (
                    tHistQ is not None and tHistQ[1][0] is not None
                ):  # state at t - 50 ms from the 25 Hz robot ring (same anchor / jaw command)
                    tTh = {tSide: fk(tArms, tSide, tHistQ[1][0][tSide]) for tSide in SIDES}
                    tHistState = (
                        tHistQ[0],
                        buildState(tTh["left"], tTh["right"], tAnchor["left"], tAnchor["right"], tGCmd["left"], tGCmd["right"]),
                    )
                    if tPGrasp is not None:
                        tHistState = (tHistState[0], np.array(tHistState[1], np.float64))
                        tHistState[1][GRIP_COLUMNS] = tPGrasp.at(tHistNs)
                tHistWrist = {tSide: tWristCams[tSide].mRing.near(tHistNs, HIST_TOL_S) for tSide in SIDES}
                if tHistState is not None and tHistGlobal is not None and all(tV is not None for tV in tHistWrist.values()):
                    tPrevObs = (
                        {"global": tHistGlobal[1][0], "left_wrist": tHistWrist["left"][1], "right_wrist": tHistWrist["right"][1]},
                        tHistState[1],
                    )
                    tHistUsed = True
            tT1 = time.monotonic()
            # ---- policy
            tRtc = None
            if tRtcMode != "off":
                tIr, tQr = tStreamer.anchor(tRefNs / NS_PER_S)
                tQp = tStreamer.plannedAfter(tIr, CHUNK_ROWS)
                if tQp:
                    tTaInv = {tSide: inv_T(fk(tArms, tSide, tQr[tSide])) for tSide in SIDES}
                    tYr = np.zeros((CHUNK_ROWS, ACTION_DIM))
                    for tK in range(CHUNK_ROWS):
                        tQk = tQp[min(tK, len(tQp) - 1)]
                        tGk = tRtcJaw.get(tIr + 1 + tK, tGCmd)
                        for tSide in SIDES:
                            tYr[tK, POSE_SLICE[tSide]] = T_to_pose9(tTaInv[tSide] @ fk(tArms, tSide, tQk[tSide]))
                            tYr[tK, GRIP_INDEX[tSide]] = tGk[tSide]
                    tDelay = int(np.ceil(max(tLatHist) / tRowDt)) + 1 if tLatHist else RTC_DEFAULT_DELAY
                    tRtc = dict(
                        Y=tYr,
                        W=rtcWeights(CHUNK_ROWS, tDelay, len(tQp)),
                        g0=np.array([tGCmd["left"], tGCmd["right"]], np.float64),
                        mode=tRtcMode,
                        beta=getattr(args, "rtc_beta", RTC_BETA_DEFAULT),
                    )
            tPredictKw = dict(rtc=tRtc) if tRtc is not None else {}
            tChunkNorm, tChunk = tRunner.predict(
                tRunner.item(tImgs, tState, tPrevObs), seed=(0 if args.dit_fixed_noise else tCycle), **tPredictKw
            )
            tT2 = time.monotonic()
            if (
                tRtc is not None
            ):  # how well the new chunk continues the previous plan (W = 1 rows) and the overlap (0 < W < 1), mm
                tHard = tRtc["W"] >= 1.0
                tSoft = (tRtc["W"] > 0) & (tRtc["W"] < 1.0)
                tE = np.array(
                    [
                        [
                            np.linalg.norm(tChunk[tK, POSE_SLICE[tSide]][:3] - tRtc["Y"][tK, POSE_SLICE[tSide]][:3]) * MM_PER_M
                            for tSide in SIDES
                        ]
                        for tK in range(CHUNK_ROWS)
                    ]
                ).mean(1)
                tRtcLog.append(
                    dict(
                        t=round(tEl, 2),
                        d=int(tHard.sum()),
                        n_prev=int((tRtc["W"] > 0).sum()),
                        hard_mm=round(float(tE[tHard].mean()), 1) if tHard.any() else None,
                        soft_mm=round(float(tE[tSoft].mean()), 1) if tSoft.any() else None,
                        pol_ms=round((tT2 - tT1) * MS_PER_S),
                    )
                )
            if (
                tDit and args.speed != 1.0 and not tLook
            ):  # 10-29: DiT 20 Hz rows slowed along the path (X-VLA: in the runner; look-ahead: tick = row / speed)
                tChunk = retime(tChunk, ROW_DT_S, args.speed)
            tFlags = [f"STOP:stale_image_{max(tImgAge):.0f}ms"] if tStaleImg else []
            if not np.isfinite(tChunk).all():
                tFlags.append("STOP:nan_chunk")
            tTc = decodeChunk(tChunk, tTm["left"], tTm["right"])  # absolute targets anchored at T(t_obs)
            if tLook:
                tIAnc, tQAnc = tStreamer.anchor(tRefNs / NS_PER_S)  # tick of the observation (capture) + the commanded q there
            tPreview.submit(tEl, tTc, tQm)
            tWsOut = {
                tSide: float(
                    np.mean(np.any((tTc[tSide][:, :3, 3] < tWs[tSide][0]) | (tTc[tSide][:, :3, 3] > tWs[tSide][1]), axis=1))
                )
                for tSide in SIDES
            }
            if args.temporal_ensemble > 0:
                tEns.append((tEl, tTc))
                tEns = [tE_ for tE_ in tEns if tEl - tE_[0] < CHUNK_ROWS * ROW_DT_S]
            # ---- re-anchor on the newest measured TCP (latency compensation)
            _, (tQn, _, tEst2) = tPoller.mRing.latest()
            if (tEst2 or tQn is None) and args.go and args.keep_going:  # recovered at the top of the next cycle
                tCycle += 1
                continue
            if tEst2 or tQn is None:
                tStopReason = "bridge E-STOP / stale read"
                break
            tTn = {tSide: fk(tArms, tSide, tQn[tSide]) for tSide in SIDES}
            if tLabel:  # 10-27 label replay: chain on the COMMANDED pose (no tracking-lag loss)
                tTn = {tSide: fk(tArms, tSide, tQAcc[tSide]) for tSide in SIDES}
            # 10-29 X-VLA: the arm trails the commanded path by 50-80 mm at segment end -> a plan anchored on the MEASURED TCP
            # starts behind
            # the streamer and every new segment first went backward, then forward again.  Chain on the commanded pose (resync
            # handles drift).
            if tRetimed and args.go:
                tTn = {tSide: fk(tArms, tSide, tQAcc[tSide]) for tSide in SIDES}
            if tLook:
                tTn = {tSide: fk(tArms, tSide, tQAnc[tSide]) for tSide in SIDES}
            # 10-23 user: the left wrist drives (j4/j5) drop out and the bridge re-enables them in place -> the commanded path
            # (q_acc)
            # no longer matches the arm and the next IK fails.  Re-seed that arm on its measured q, hold it there this cycle,
            # continue.
            if args.go:
                # look-ahead: the timeline q at the physical time NOW (11-06: not mLast -- with --latency-match it is sent ~60 ms
                # ahead and the gap to the measured pose grew past resync_deg on a fast first move)
                tQRef = tStreamer.anchor(time.monotonic())[1] if tLook else tQAcc
                tResyncSides = [
                    tSide for tSide in SIDES if float(np.degrees(np.abs(tQRef[tSide] - tQn[tSide])).max()) > args.resync_deg
                ]
                if tResyncSides:
                    tResyncs += 1
                    for tSide in tResyncSides:
                        tQAcc[tSide] = tQn[tSide].copy()
                    say(
                        f"[{tEl:5.1f}s] 측정 자세와 명령이 {args.resync_deg}도 넘게 어긋남 ({', '.join(tResyncSides)}) -> 측정 자세로 다시 맞추고 계속 ({tResyncs}/{args.max_resync})"
                    )
                    if tResyncs > args.max_resync and not args.keep_going:
                        tStopReason = f"resync {tResyncs} 회 초과"
                        break
                    tStreamer.setTimeline([dict(tQAcc)], ROW_DT_S, tGCmd)
                    time.sleep(RESYNC_HOLD_S)
                    tCycle += 1
                    continue
            tAgeS = (time.monotonic_ns() - tRefNs) / NS_PER_S
            tK0 = int(np.clip(np.floor(tAgeS / ROW_DT_S), 0, CHUNK_ROWS - args.commit))
            if tLabel or tLook:
                tK0 = 0  # label cursor / look-ahead: rows are keyed by tick, the timeline drops the late ones
            if tAgeS > args.max_plan_age_s:
                tFlags.append(f"HOLD:stale_plan_{tAgeS * MS_PER_S:.0f}ms")  # never commit a plan this old
            tSpd = (args.slow_start_speed if tEl < args.slow_start_s else args.speed) * (WARN_SLOW_FACTOR if tWarnSlow else 1.0)
            if tRetimed:
                tSpd = (
                    WARN_SLOW_FACTOR if tWarnSlow else 1.0
                )  # slowness is already in the retimed chunk (X-VLA runner / DiT retime)
            # 10-23: with long commits the jaw must follow the committed SEGMENT, not chunk row 1 (a 16-step GO never opened: k1
            # 0.2-0.3,
            # later rows 0.7) -> --grip-at mid (default) = the row in the middle of the committed steps; max = open if any
            # committed row opens
            tSeg = tChunk[tK0 : tK0 + args.commit]
            tGripPred = {
                tSide: float(
                    {
                        "first": tChunk[args.grip_step - 1, GRIP_INDEX[tSide]],
                        "mid": tSeg[len(tSeg) // 2, GRIP_INDEX[tSide]],
                        "last": tSeg[-1, GRIP_INDEX[tSide]],
                        "max": tSeg[:, GRIP_INDEX[tSide]].max(),
                    }["mid" if args.grip_at == GRIP_AT_ROW else args.grip_at]
                )
                for tSide in SIDES
            }
            tGNew = jawCommand(args, tGripPred, tActive, tGCmd)
            # 11-06 user (jaw stayed open while the chunk's later rows closed it): --grip-at row = the jaw follows every committed row
            # in time like the joints (same binary / linear mapping per row); the mid-row value stays the logged / state jaw command
            tGRows = None
            if args.grip_at == GRIP_AT_ROW:
                # 11-06 user (jaw closed after the arm already lifted): --grip-lead-rows L = row j's jaw takes chunk row j + L
                tLead = 0 if tLat.get("jaw_ms") else int(getattr(args, "grip_lead_rows", 0))  # measured jaw send-ahead replaces it
                tGRows = [
                    jawCommand(
                        args,
                        {tSide: float(tChunk[min(tK0 + tJ + tLead, CHUNK_ROWS - 1), GRIP_INDEX[tSide]]) for tSide in SIDES},
                        tActive,
                        tGCmd,
                    )
                    for tJ in range(len(tSeg))
                ]
            tPrevRes = tPreview.mResult
            if tPrevRes is not None:
                tFlags += [tF for tF in tPrevRes["flags"] if tF.startswith("HOLD")]
            # ---- committed steps: candidate -> limiter -> IK -> safety
            tQPrev = {tSide: (tQAcc[tSide] if (args.go or tLabel) else tQn[tSide]).copy() for tSide in SIDES}
            tCands, tLims, tQIks, tQAccs, tIkMm, tIkDeg = [], [], [], [], [], []
            if tLook:
                tQPrev = {tSide: tQAnc[tSide].copy() for tSide in SIDES}
            for tJ in range(1, args.commit + 1):
                tK = tK0 + tJ
                # T_now @ inv(dT_k0) @ dT_k
                tCand = {
                    tSide: tTn[tSide] @ inv_T(tTc[tSide][tK0 - 1] if tK0 > 0 else tTm[tSide]) @ tTc[tSide][tK - 1]
                    for tSide in SIDES
                }
                if args.temporal_ensemble > 0 and not tLook:  # look-ahead blends in the timeline instead
                    tCand = {tSide: ensemble(tEns, tEl + tK * ROW_DT_S, tSide, args.temporal_ensemble) for tSide in SIDES}
                if tAmp != 1.0:  # 11-03 user "움직임이 너무 커": shrink each committed step's motion from T_now
                    tCand = {tSide: tTn[tSide] @ scaleTransform(inv_T(tTn[tSide]) @ tCand[tSide], tAmp) for tSide in SIDES}
                if tCalib is not None:  # plane follows the row xy; only the not-yet-carried part is added
                    tRamp = 1.0 if tZOffDone else min(1.0, tJ / Z_RAMP_ROWS)
                    for tSide in SIDES:
                        tZOff[tSide] = tZOffM + heightCalib.dzAt(tCalib, tSide, tCand[tSide][:2, 3])
                        tCand[tSide] = tCand[tSide].copy()
                        tCand[tSide][2, 3] += (tZOff[tSide] - tZBase[tSide]) * tRamp
                elif not tZOffDone:
                    for tSide in SIDES:
                        tCand[tSide] = tCand[tSide].copy()
                        tCand[tSide][2, 3] += tZOff[tSide] * min(1.0, tJ / Z_RAMP_ROWS)
                tLim, tQj, tQa, tEm, tEd = {}, {}, {}, {}, {}
                for tSide in SIDES:
                    if tSide not in tActive:  # dry arm: hold the last commanded joints
                        tLim[tSide] = tTn[tSide]
                        tQj[tSide] = tQPrev[tSide]
                        tQa[tSide] = tQPrev[tSide]
                        tEm[tSide] = tEd[tSide] = 0.0
                        continue
                    if tZFloorM is not None or tContact is not None:  # 11-06 user "너무 바닥을 찍어": robot-frame TCP floor
                        tXy = tCand[tSide][:2, 3]  # 11-07: static floor (z_floor set) and/or the measured contact floor
                        tFloor = max(floorBase(tSide, tXy) if tZFloorM is not None else -np.inf,
                                     tContact.floor(tSide, tXy) if tContact is not None else -np.inf)
                        if tCand[tSide][2, 3] < tFloor:
                            tFlags.append(f"LIM:z_floor_{tSide[0]}_{(tFloor - tCand[tSide][2, 3]) * MM_PER_M:.0f}mm")
                            tCand[tSide] = tCand[tSide].copy()
                            tCand[tSide][2, 3] = tFloor
                    tLim[tSide], tQj[tSide], tQa[tSide], tEm[tSide], tEd[tSide] = _limitAndSolve(
                        tSide,
                        args,
                        tTn[tSide],
                        tCand[tSide],
                        tSpd,
                        tRowDt,
                        tJ,
                        tArms[tSide],
                        tGNew[tSide],
                        tWs[tSide],
                        tQPrev[tSide],
                        tFlags,
                    )
                    tTq = fk(tArms, tSide, tQa[tSide])
                    tTableZ = tRobotArgs.table_z + (
                        heightCalib.dzAt(tCalib, tSide, tTq[:2, 3]) if tCalib is not None else 0.0
                    )  # 11-01 measured table
                    if tTq[2, 3] < tTableZ + args.tcp_table_margin:
                        tFlags.append(f"STOP:tcp_table_{tSide[0]}")
                tCheck = tCol.check(tQa, tableZ=tRobotArgs.table_z, armClear=args.arm_clear)
                tGap = linkGap(tArms, tQa)
                if not tCheck["ok_arm"]:
                    tFlags.append(f"STOP:arm_collision_{tCheck['pair']}_{tCheck['arm_clear_m'] * MM_PER_M:.0f}mm")
                if not tCheck["ok_table"]:
                    tFlags.append(f"STOP:link_table_{tCheck['table_clear_m'] * MM_PER_M:.0f}mm")
                if tGap < args.warn_gap:
                    tFlags.append(f"WARN:link_gap_{tGap * MM_PER_M:.0f}mm")
                if tJ == args.commit and (not tZOffDone or tCalib is not None):
                    tZOffDone = True
                    tZBase = dict(tZOff)
                    tFlags.append(f"INFO:z_offset_L{tZOff['left'] * MM_PER_M:+.0f}_R{tZOff['right'] * MM_PER_M:+.0f}mm")
                tCands.append(tCand)
                tLims.append(tLim)
                tQIks.append(tQj)
                tQAccs.append(tQa)
                tIkMm.append(tEm)
                tIkDeg.append(tEd)
                tQPrev = tQa
            tWarnSlow = any(tF.startswith("WARN:link_gap") for tF in tFlags)
            tT3 = time.monotonic()
            tFlags = relaxFlags(tFlags, args, tActive)
            tStop = [tF for tF in tFlags if tF.startswith(BLOCKING)]
            # ---- commit (go) / virtual commit (dry-run)
            if tStop:
                tViolations += (
                    1  # keep_going: this plan is NOT sent (the streamer holds the last accepted pose), the run continues
                )
                if tStep:
                    if tTakeStep():
                        say(f"[{tEl:5.1f}s] 스텝 거부 (안전 플래그): {', '.join(tStop)}")
                elif args.go and not args.keep_going:
                    tStopReason = "SAFETY " + ", ".join(tStop)
            elif args.go and tStep and not tTakeStep():
                tFlags.append("WAIT:step")  # step mode: plan shown, nothing sent, the streamer holds the last accepted pose
            elif tLook:
                tNd, tNb = tStreamer.merge(tIAnc, [tQAccs[tJ] for tJ in range(args.commit)], tGNew, tGRows)
                tDropped += tNd
                tBlended += tNb
                tQAcc = tQAccs[-1]
                tGCmd = tGNew
                if tRtcMode != "off":  # jaw rows of the committed segment, keyed by tick (the RTC target for the next plan)
                    tRtcJaw.update(
                        {
                            tIAnc + 1 + tJ: {tSide: float(tChunk[tJ, GRIP_INDEX[tSide]]) for tSide in SIDES}
                            for tJ in range(args.commit)
                        }
                    )
                    for tI in [tI for tI in tRtcJaw if tI < tIAnc - RTC_JAW_KEEP_TICKS]:
                        del tRtcJaw[tI]
            elif args.go:
                tStreamer.setTimeline([tQAccs[tJ] for tJ in range(args.commit)], ROW_DT_S, tGNew, tGRows)
                tQAcc = tQAccs[-1]
                tGCmd = tGNew
                if tStep:
                    say(
                        f"[{tEl:5.1f}s] 한 스텝 실행: {args.commit} 스텝 ({args.commit * ROW_DT_S * MS_PER_S:.0f} ms 분량), 집게 L {tGNew['left']:.0f} R {tGNew['right']:.0f}"
                    )
            elif tLabel:
                tQAcc = tQAccs[-1]  # label dry-run: virtual chaining on the commanded path
            if not args.go and not tStop:
                tGCmd = tGNew  # dry-run: the state gripper follows the virtual command
            if tLabel and not tStop:
                tRunner.mCursor = min(tRunner.mDurS, tRunner.mCursor + args.commit * ROW_DT_S * args.label_rate)
                if tRunner.done():
                    tStopReason = "label end"
            tT4 = time.monotonic()
            # ---- log
            tSmooth = {
                f"dchunk_k{tK}_mm": (
                    float(np.mean([np.linalg.norm(tTc[tSide][tK - 1][:3, 3] - tPrevTc[tSide][tK - 1][:3, 3]) for tSide in SIDES]))
                    * MM_PER_M
                    if tPrevTc is not None
                    else np.nan
                )
                for tK in SMOOTHNESS_ROWS
            }
            tPrevTc = tTc
            tPreviewWarn = [tF for tF in tPrevRes["flags"] if tF.startswith("WARN")] if tPrevRes else []
            tEntry = dict(
                t=tEl,
                lat_obs_ms=(tT1 - tC0) * MS_PER_S,
                lat_pol_ms=(tT2 - tT1) * MS_PER_S,
                lat_ctrl_ms=(tT3 - tT2) * MS_PER_S,
                lat_total_ms=(tT4 - tC0) * MS_PER_S,
                obs_to_commit_ms=(time.monotonic_ns() - tObsNs) / NS_PER_MS,
                k0=tK0,
                hist_used=tHistUsed,
                q_meas=np.r_[tQm["left"], tQm["right"]],
                q_now=np.r_[tQn["left"], tQn["right"]],
                T_meas=np.stack([tTm["left"], tTm["right"]]),
                T_now=np.stack([tTn["left"], tTn["right"]]),
                state=tState,
                chunk_norm=tChunkNorm,
                chunk=tChunk,
                T_chunk=np.stack([tTc["left"], tTc["right"]]),
                grip_raw=tChunk[:, GRIP_COLUMNS].copy(),
                T_cand=np.stack([[tX["left"], tX["right"]] for tX in tCands]),
                T_lim=np.stack([[tX["left"], tX["right"]] for tX in tLims]),
                q_ik=np.stack([np.r_[tX["left"], tX["right"]] for tX in tQIks]),
                q_acc=np.stack([np.r_[tX["left"], tX["right"]] for tX in tQAccs]),
                ik_mm=[[tX["left"], tX["right"]] for tX in tIkMm],
                ik_deg=[[tX["left"], tX["right"]] for tX in tIkDeg],
                pv_ik_ok=(tPrevRes["ik_ok"] if tPrevRes else [np.nan, np.nan]),
                pv_ik_max_mm=(tPrevRes["ik_max_mm"] if tPrevRes else [np.nan, np.nan]),
                pv_min_clear=(tPrevRes["min_clear"] if tPrevRes else np.nan),
                pv_age_s=(tEl - tPrevRes["t"] if tPrevRes else np.nan),
                g_pred=[tGripPred["left"], tGripPred["right"]],
                g_cmd=[tGNew["left"], tGNew["right"]],
                speed=tSpd,
                flags=tFlags + tPreviewWarn,
                arm_clear=tCheck["arm_clear_m"],
                table_clear=tCheck["table_clear_m"],
                link_gap=tGap,
                ws_chunk_out=[tWsOut["left"], tWsOut["right"]],
                img_age_ms=tImgAge,
                stop=bool(tStop),
                label_tau=(tRunner.mCursor if tLabel else np.nan),
                gimg_jpg=tGlobalJpg,
                wl=tImgs["left_wrist"],
                wr=tImgs["right_wrist"],
                **tSmooth,
            )
            if tPNow is not None:  # 11-06 PGRASP runs only: the p_grasp fed as state 18, 19
                tEntry["pgrasp"] = list(tPNow)
            for tKey, tValue in tEntry.items():
                tLog[tKey].append(tValue)
            tD1 = {tSide: np.linalg.norm(tTc[tSide][0][:3, 3] - tTm[tSide][:3, 3]) * MM_PER_M for tSide in SIDES}
            tD32 = {tSide: (tTc[tSide][CHUNK_ROWS - 1][:3, 3] - tTm[tSide][:3, 3]) * MM_PER_M for tSide in SIDES}
            if tCycle % args.print_every == 0 or tStop:
                print(
                    f"[{tEl:5.1f}s] loop {tLog['lat_total_ms'][-1]:5.1f} ms (obs {tLog['lat_obs_ms'][-1]:4.1f} pol {tLog['lat_pol_ms'][-1]:5.1f} "
                    f"ctrl {tLog['lat_ctrl_ms'][-1]:4.1f}) k0 {tK0} | k1 L {tD1['left']:5.1f} R {tD1['right']:5.1f} mm | "
                    f"k32 L {np.round(tD32['left']).astype(int)} R {np.round(tD32['right']).astype(int)} mm | g L {tGripPred['left']:.2f} R {tGripPred['right']:.2f} | "
                    + (f"p_grasp L {tPNow[0]:.2f} R {tPNow[1]:.2f} | " if tPNow is not None else "") +
                    f"preview IK {tPrevRes['ik_ok'] if tPrevRes else '-'} clr {tCheck['arm_clear_m'] * MM_PER_M:.0f} mm | "
                    f"{' '.join(tF for tF in tFlags if not tF.startswith('LIM'))}",
                    flush=True,
                )
            ctl.publish(
                dict(
                    t=tEl,
                    cyc=tCycle,
                    Tm=tTm,
                    Tc=tTc,
                    grip=tChunk[:, GRIP_COLUMNS].copy(),
                    flags=tLog["flags"][-1],
                    lat_pol_ms=tLog["lat_pol_ms"][-1],
                    lat_total_ms=tLog["lat_total_ms"][-1],
                    hz=tCycle / max(tEl, MIN_ELAPSED_S),
                    g_cmd=tGNew,
                    speed=tSpd,
                    stop=tStop,
                    go=args.go,
                    k0=tK0,
                    **({} if tPNow is None else dict(pgrasp=[round(float(tX), 2) for tX in tPNow])),
                )
            )
            tCycle += 1
            if tStop and args.go and not args.keep_going and not tStep:
                break
            if tStopReason == "label end":
                break
            # ---- pacing: next plan so that it is ready when the committed segment ends (ACT: 20 Hz)
            tLatEma = (tT4 - tC0) if tLatEma is None else LAT_EMA_KEEP * tLatEma + LAT_EMA_GAIN * (tT4 - tC0)
            tNextT = tC0 + args.commit * ROW_DT_S - (tLatEma if tDit else 0.0)
            if tLook:
                # re-plan when the queued ticks only cover the slowest recent plan + 1 tick (like indy7 ChunkLookahead).  10-29: a
                # fixed
                # half-chunk floor (LeRobot async chunk_size_threshold 0.5) kept only the first half of every chunk: DiT never got
                # past its
                # dip, X-VLA at speed 0.2 (333 ms rows, plan ~3 ticks) hovered while the preview's chunk END went to the T-shirt.
                # At speed 1
                # X-VLA still re-plans at ~16/32.
                tLatHist.append(tT4 - tC0)
                tNeed = int(np.ceil(max(tLatHist) / tRowDt)) + 1
                tNextT = tStreamer.mTBase + (tStreamer.lastIndex - tNeed) * tRowDt
            time.sleep(max(0.0, tNextT - time.monotonic()))
    except KeyboardInterrupt:
        tStopReason = "Ctrl+C"
    except (SystemExit, DeployError) as tExc:
        tStopReason = str(tExc)
    except Exception as tExc:
        tStopReason = f"exception {type(tExc).__name__}: {tExc}"
        import traceback

        traceback.print_exc()
    finally:
        if args.go:
            # 11-01 user "estop 으로 계속 영점이 틀어져, 보통 스톱은 없어?": every end used to latch an E-STOP (instant hold at the measured
            # pose = a velocity step).  An operator stop / time limit / label end now freezes the stream at the pose it is
            # commanding
            # NOW, the bridge's acceleration-limited profile brakes onto it, and nothing is latched.  Safety stops, bridge errors
            # and
            # exceptions still E-STOP.
            tReasonText = str(tStopReason or "")
            tSoft = (
                tReasonText.startswith(SOFT_STOP_PREFIXES)
                and tStreamer is not None
                and tStreamer.mError is None
                and tStreamer.is_alive()
            )
            if tSoft:
                tStreamer.setTimeline([], tStreamer.mDt, tStreamer.mG)
                time.sleep(SOFT_STOP_SETTLE_S)
                tStreamer.mStop = True
                if tLiveBridge is not None:
                    tLiveBridge.close()
                say(f"부드럽게 정지 (감속 후 현재 명령 자세 유지, E-STOP 아님): {tStopReason}")
            else:
                if tStreamer is not None:
                    tStreamer.mStop = True
                if tLiveBridge is not None:
                    tLiveBridge.estop()
                    tLiveBridge.close()
                say(f"정지 + 현재 자세 유지 (E-STOP 홀드): {tStopReason}")
        tPreview.mStop = True
    tTauLog, tPoller.mTauLog, tPoller.mCmdFn = tPoller.mTauLog, None, None
    if tTauLog:  # 11-07: 25 Hz measured q, joint torque and the streamer's commanded q of this run (gravity-model data)
        tNan = np.full(6, np.nan)
        np.savez_compressed(
            tRunDir / "qtau.npz",
            t_ns=np.array([tR[0] for tR in tTauLog], np.int64),
            q=np.array([[tR[1][tSide] for tSide in SIDES] for tR in tTauLog], float),
            tau=np.array([[tR[2][tSide] if tR[2].get(tSide) is not None else tNan for tSide in SIDES] for tR in tTauLog], float),
            q_cmd=np.array([[tR[3][tSide] if tR[3] else tNan for tSide in SIDES] for tR in tTauLog], float),
        )
    if tContact is not None:
        tContact.mStop = True
        (tRunDir / "contact.json").write_text(json.dumps(dict(
            events=tContact.mEvents, points=tContact.mPoints,
            guard_holds=tStreamer.mGuardHolds if tStreamer is not None else 0, gap_mm=tContact.mGap * 1e3, tau_nm=tContact.mTau,
            trace=tContact.mTrace), indent=0))
        say(f"바닥 접촉 {len(tContact.mEvents)} 회, 스트리머 홀드 {tStreamer.mGuardHolds if tStreamer else 0} 틱")
    say(f"끝: {tStopReason}")
    tSummary = summarize(tLog, args, tStopReason, tViolations, tRunDir, tWs, tReadyOff, tCamsCfg, tPreview, tCkpt, tRunner)
    if tLook:  # 10-29 look-ahead counters (summarize() has no access to the loop state)
        tHardMm = [tR["hard_mm"] for tR in tRtcLog if tR["hard_mm"] is not None]
        tLookahead = dict(
            row_dt_ms=round(tRowDt * MS_PER_S, 1),
            rows_dropped=tDropped,
            rows_blended=tBlended,
            dry_stream_s=round(tStreamer.mDryTicks / args.stream_hz, 2) if tStreamer is not None else None,
            rtc=tRtcMode,
            rtc_hard_mm_mean=(round(float(np.mean(tHardMm)), 1) if any(tR["hard_mm"] is not None for tR in tRtcLog) else None),
            rtc_pol_ms_mean=(round(float(np.mean([tR["pol_ms"] for tR in tRtcLog])), 0) if tRtcLog else None),
            rtc_cycles=tRtcLog,
        )
        (tRunDir / "lookahead.json").write_text(json.dumps(tLookahead, indent=1))
        say(f"look-ahead: {tLookahead}")
        if isinstance(tSummary, dict):
            tSummary["lookahead"] = tLookahead
    if tSummary is not None:
        tSummary["stop_reason"] = tStopReason
    return tSummary
