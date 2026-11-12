"""`start`: plan (dry) or drive (--go) READY -> a training episode's start pose, every waypoint checked (deploy setup step).

Target = the start TCPs (pose_abs at frame 0) of ONE training episode (default: the medoid episode, so both arms and T_LR come
from the same real start), + --start-lift in z. Path: Cartesian line READY -> target + 10 cm (IK seeded step to step), then
straight down; every waypoint checked (IK error, joint limits, capsule arm-arm, link-table, TCP above table). Dry-run prints the
plan and writes start_pose.json (TAGRIG_STATE_DIR); --go streams it at --deg-s (umi_replay_bi.stream_path) and then simply stops
streaming (the bridge holds the last goal).
"""

from __future__ import annotations
import json
import time

import numpy as np

from laundry_shared.contracts.action_contract import pose9_to_T

from ..core import height_calib as heightCalib
from ..core.collision import Collision
from ..core.contract import POSE_SLICE, SIDES, TRAIN_ROOT
from ..core.errors import HardwareError, RunRefused
from ..core.kinematics import fk, makeArms
from ..core.label_runner import LabelRunner
from .paths import START_FILE, say

DEFAULT_ROOT = "tagrig_laundry_v1_fused_train"
XYZ_COLUMNS = [0, 1, 2, 9, 10, 11]  # pose_abs columns of both TCP positions (medoid start)
READY_TOL_DEG = 5
BRIDGE_TIMEOUT_S = 0.5
APPROACH_LIFT_M = 0.10  # the approach leg ends this far above the target, then goes straight down
APPROACH_STEPS, DESCENT_STEPS = 40, 20
MAX_IK_MM = 10
MIN_ARM_CLEAR_M = 0.01
MIN_TCP_ABOVE_TABLE_M = 0.005
XY_SHIFT_KEY = "xy_B_minus_robot"
MM_PER_M = 1e3
STAMP_FMT = "%Y-%m-%d %H:%M:%S"


def _leg(arm, qa, Ta, Tb, n: int = APPROACH_STEPS) -> list:
    """Cartesian line Ta -> Tb (slerp + lerp), IK seeded step to step -> [(q, pos err m, rot err deg)]."""
    from scipy.spatial.transform import Rotation, Slerp

    tSlerp = Slerp([0, 1], Rotation.from_matrix([Ta[:3, :3], Tb[:3, :3]]))
    tOut, tQ = [], qa.copy()
    for tU in np.linspace(0, 1, n):
        tT = np.eye(4)
        tT[:3, :3] = tSlerp(tU).as_matrix()
        tT[:3, 3] = (1 - tU) * Ta[:3, 3] + tU * Tb[:3, 3]
        tQ, tPe, tRe = arm.mFree.ik(tQ, tT)
        tOut.append((tQ.copy(), tPe, tRe))
    return tOut


def startCmd(args) -> None:
    from laundry_shared.visual_teleop import config as vcfg
    from laundry_shared.visual_teleop.robot.bridge import HttpBridge
    from laundry_shared.visual_teleop.tools.robot_moves import READY_Q_DEG, unpark

    if getattr(args, "label", ""):  # 10-27 label replay: the label's own first pose (A' vs Cg test)
        tLabel = LabelRunner(args.label)
        tEpisode = args.label
        tI0 = 0
        tP0 = {tSide: tLabel.poseAt(tSide, 0.0) for tSide in SIDES}
    else:  # 10-28 --root (GOLD-281)
        tRoot = TRAIN_ROOT / "data/lerobot" / (getattr(args, "root", "") or DEFAULT_ROOT)
        # 11-06: the parquet columns only -- LeRobotDataset() goes to the HF hub for a set pulled without its videos (train_nov)
        import pandas as pd

        tHf = pd.concat([pd.read_parquet(tF, columns=["pose_abs", "frame_index", "episode_index"])
                         for tF in sorted((tRoot / "data").rglob("*.parquet"))], ignore_index=True)
        tP = np.stack(tHf["pose_abs"].to_numpy()).astype(float)
        tFrame = tHf["frame_index"].to_numpy()
        tEp = tHf["episode_index"].to_numpy()
        tStarts = np.flatnonzero(tFrame == 0)
        tMedian = np.median(tP[tStarts][:, XYZ_COLUMNS], 0)
        tMedoid = int(tEp[tStarts[np.argmin(np.linalg.norm(tP[tStarts][:, XYZ_COLUMNS] - tMedian, axis=1))]])
        tEpisode = tMedoid if args.episode < 0 else args.episode
        tI0 = int(np.flatnonzero((tEp == tEpisode) & (tFrame == 0))[0])
        tP0 = {tSide: pose9_to_T(tP[tI0, POSE_SLICE[tSide]]) for tSide in SIDES}
    tArms, tRobotArgs = makeArms()
    tCol = Collision(tArms)
    tQReady = {tSide: np.radians(READY_Q_DEG[tSide]) for tSide in SIDES}
    if args.go:
        tCfg = vcfg.load()
        tBridge = HttpBridge(tCfg.robot["base_url"], live=True, timeout_s=BRIDGE_TIMEOUT_S)
        tObs = tBridge.observe()
        if tObs.estop:
            raise HardwareError("로봇 E-STOP/미연결/읽기 지연")
        if max(np.degrees(np.abs(tObs.q[tSide] - tQReady[tSide])).max() for tSide in SIDES) > READY_TOL_DEG:
            raise RunRefused("두 팔이 READY 가 아님")
    tTarget = {}
    tCalib = None
    if getattr(
        args, "height_calib", False
    ):  # 11-01 user "xy 도 맞추어줘": land the start TCP where the label frame has it (robot FK = B - shift)
        tCalib = heightCalib.loadCalibration()
    for tSide in SIDES:
        tT = tP0[tSide].copy()
        tT[2, 3] += args.start_lift
        if tCalib is not None and XY_SHIFT_KEY in tCalib["fit"].get(tSide, {}):
            tT[:2, 3] -= np.asarray(tCalib["fit"][tSide][XY_SHIFT_KEY])
        tTarget[tSide] = tT
    if tCalib is not None:
        say(
            "시작 자세 xy 보정: "
            + ", ".join(
                f"{tSide} {np.round(np.asarray(tCalib['fit'][tSide].get(XY_SHIFT_KEY, [0, 0])) * -1e3, 1).tolist()} mm"
                for tSide in SIDES
            )
        )
    tPath = {}
    tReport = {}
    for tSide in SIDES:
        tTr = fk(tArms, tSide, tQReady[tSide])
        tUp = tTarget[tSide].copy()
        tUp[2, 3] += APPROACH_LIFT_M
        tLeg1 = _leg(tArms[tSide], tQReady[tSide], tTr, tUp)
        tLeg2 = _leg(tArms[tSide], tLeg1[-1][0], tUp, tTarget[tSide], DESCENT_STEPS)
        tPath[tSide] = tLeg1 + tLeg2
        tReport[tSide] = dict(
            target_p=np.round(tTarget[tSide][:3, 3], 3).tolist(),
            ik_max_mm=round(max(tX[1] for tX in tPath[tSide]) * MM_PER_M, 1),
            rot_max_deg=round(max(tX[2] for tX in tPath[tSide]), 1),
            limits_ok=bool(all(np.all(tX[0] >= tArms[tSide].mLo) and np.all(tX[0] <= tArms[tSide].mHi) for tX in tPath[tSide])),
            q_start_deg=np.round(np.degrees(tPath[tSide][-1][0]), 1).tolist(),
        )
    tN = max(len(tV) for tV in tPath.values())
    tWaypoints = [{tSide: tPath[tSide][min(tI, len(tPath[tSide]) - 1)][0] for tSide in SIDES} for tI in range(tN)]
    tChecks = [tCol.check(tW, tableZ=tRobotArgs.table_z) for tW in tWaypoints]
    tTcpZ = min(fk(tArms, tSide, tW[tSide])[2, 3] for tW in tWaypoints for tSide in SIDES)
    tJointStep = max(
        float(np.degrees(np.abs(tWaypoints[tI + 1][tSide] - tWaypoints[tI][tSide])).max())
        for tI in range(tN - 1)
        for tSide in SIDES
    )
    tMinClear = min(tC["arm_clear_m"] for tC in tChecks)
    tMinTable = min(tC["table_clear_m"] for tC in tChecks)
    tOk = (
        all(tR["ik_max_mm"] <= MAX_IK_MM and tR["limits_ok"] for tR in tReport.values())
        and tMinClear >= MIN_ARM_CLEAR_M
        and tMinTable >= 0
        and tTcpZ >= tRobotArgs.table_z + MIN_TCP_ABOVE_TABLE_M
    )
    tLength = sum(
        max(float(np.degrees(np.abs(tWaypoints[tI + 1][tSide] - tWaypoints[tI][tSide])).max()) for tSide in SIDES)
        for tI in range(tN - 1)
    )
    tOut = dict(
        episode=tEpisode,
        frame0_index=tI0,
        start_lift_m=args.start_lift,
        arms=tReport,
        min_arm_clear_mm=round(tMinClear * MM_PER_M),
        min_table_clear_mm=round(tMinTable * MM_PER_M),
        min_tcp_z_mm=round(tTcpZ * MM_PER_M),
        max_joint_step_deg=round(tJointStep, 1),
        duration_s=round(tLength / args.deg_s, 1),
        ok=bool(tOk),
        q={tSide: tPath[tSide][-1][0].tolist() for tSide in SIDES},
        T={tSide: tTarget[tSide].tolist() for tSide in SIDES},
        created=time.strftime(STAMP_FMT),
    )
    print(json.dumps({tKey: tValue for tKey, tValue in tOut.items() if tKey not in ("q", "T")}, indent=1, ensure_ascii=False))
    if not tOk:
        raise RunRefused("시작 자세 경로 불가 -- 이동 안 함")
    START_FILE.write_text(json.dumps(tOut, indent=1))
    print("저장", START_FILE)
    if not args.go:
        print("DRY: 이동 안 함. 움직이려면 사람이 옆에서: start --go")
        return
    from laundry_shared.visual_teleop.tools.umi_replay_bi import stream_path

    for tSide in SIDES:
        unpark(tBridge, tSide)
    # [11-02 user: "또 입이 벌려져 있어"] the observed jaw was re-sent here, and a parked (torque-off) jaw relaxes open -> close both
    # jaws on the way to the start (training episodes start closed; g 0 = closed)
    tG = {tSide: 0.0 for tSide in SIDES}
    say("학습 시작 자세로 천천히 이동 (집게 닫음)")
    stream_path(tBridge, [{tSide: tBridge.observe().q[tSide] for tSide in SIDES}] + tWaypoints, tG, deg_s=args.deg_s)
    say("시작 자세 도착 (브리지가 마지막 목표를 유지)")
    tBridge.close()
