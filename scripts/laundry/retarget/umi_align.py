"""UMI <-> robot gripper ORIENTATION alignment (2026-10-20, absolute replay; was scripts/retarget/umi_align.py).

Pipeline stage: one-time calibration of the replay track. The robot holds READY (fixed R_B_E from the FK of READY_Q_DEG); the user
holds the UMI parallel to the gripper (same pointing direction, same jaw-opening direction) in each '멈추세요' of a tag_rig_record
--proto align take. Per hold (last 3 s, TAG frames only): R_E_U = R_B_E^T R_B_U. Mean over holds, spread = how well the hand matched
(expect a few deg). Input: <take>/meta.json + umi_traj_<hand>.npz. Output (--write):
configs/laundry/calibration/umi_gripper_align_v001.json (right; umi_gripper_align_left_v001.json for the left hand): R_U_E (UMI TCP
frame -> robot TCP frame) so that the robot target orientation for a UMI pose is R_B_E* = R_B_U R_U_E.
    python scripts/laundry/retarget/umi_align.py datasets/tag_rig/<align take> [--write]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from laundry_common.paths import CONFIGS

ALIGN_FILES = {"right": "calibration/umi_gripper_align_v001.json", "left": "calibration/umi_gripper_align_left_v001.json"}
HOLD_TEXT = "맞춰 대고"
# the last HOLD_TAIL_S of each hold are used
HOLD_TAIL_S = 3.0
NS_PER_S = 1e9
MIN_HOLD_FRAMES = 5
MIN_HOLDS = 2
BOTH_ARMS_DOF = 14
JSON_INDENT = 1
DECIMALS = 1


class AlignError(RuntimeError):
    """Fewer than MIN_HOLDS holds had enough tag frames."""


def readyRotation(hand: str) -> R:
    """R_B_E of `hand` at READY (FK of READY_Q_DEG with the Pink IK model of the robot config)."""
    from laundry_shared.ego_teleop.robot.rebot_client import SIDE_SLICE
    from laundry_shared.visual_teleop import config as vcfg
    from laundry_shared.visual_teleop.robot.ik_pink import PinkIkSolver, PinkParams
    from laundry_shared.visual_teleop.tools.robot_moves import READY_Q_DEG
    tConfig = vcfg.load()
    tJoints = np.zeros(BOTH_ARMS_DOF)
    tJoints[SIDE_SLICE[hand]] = READY_Q_DEG[hand]
    tSolver = PinkIkSolver(hand, PinkParams(**tConfig.ik["pink"]))
    tSolver.sync(tJoints)
    return R.from_matrix(tSolver.fk(np.radians(READY_Q_DEG[hand]))[:3, :3])


def holdRotations(take: Path, hand: str, readyR: R, holds: list[int] | None) -> list[R]:
    """R_E_U of every selected hold (1-based numbers in `holds`, all when empty) with at least MIN_HOLD_FRAMES tag frames."""
    tMeta = json.loads((take / "meta.json").read_text())
    tTraj = np.load(take / f"umi_traj_{hand}.npz")
    tT = tTraj["t_ns"]
    tTagged = tTraj["ntags"] > 0
    tStart = int(tMeta["t_start_monotonic_ns"])
    tOut = []
    tHoldNumber = 0
    for tStep in tMeta["protocol"]:
        if HOLD_TEXT not in tStep["text"]:
            continue
        tHoldNumber += 1
        if holds and tHoldNumber not in holds:
            continue
        tEnd = tStep["t0"] + tStep["dur"]
        tMask = tTagged & (tT >= tStart + (tEnd - HOLD_TAIL_S) * NS_PER_S) & (tT <= tStart + tEnd * NS_PER_S)
        if tMask.sum() < MIN_HOLD_FRAMES:
            print(f"  {tStep['text'][-3:]}: 태그 프레임 부족 ({int(tMask.sum())})")
            continue
        tRBU = R.from_matrix(tTraj["R_B_tcp"][tMask]).mean()
        tOut.append(readyR.inv() * tRBU)
        print(f"  {tStep['text'][-3:]}: 프레임 {int(tMask.sum())}, R_E_U rotvec {np.round(np.degrees(tOut[-1].as_rotvec()), DECIMALS)}°")
    return tOut


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tParser.add_argument("take")
    tParser.add_argument("--hand", default="right")
    tParser.add_argument("--write", action="store_true")
    tParser.add_argument("--holds", type=int, nargs="*", help="쓸 회차 (1부터), 기본 전부")
    tParser.add_argument("--out")
    return tParser


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    tOutFile = tArgs.out or str(CONFIGS / ALIGN_FILES["right" if tArgs.hand == "right" else "left"])
    tTake = Path(tArgs.take)
    try:
        tHolds = holdRotations(tTake, tArgs.hand, readyRotation(tArgs.hand), tArgs.holds)
        if len(tHolds) < MIN_HOLDS:
            raise AlignError("정지 구간이 2개 이상 필요")
    except AlignError as tError:
        print(tError, file=sys.stderr)
        return 1
    tMean = R.concatenate(tHolds).mean()
    tSpread = [float(np.degrees((tMean.inv() * tHold).magnitude())) for tHold in tHolds]
    print(f"평균 R_E_U rotvec {np.round(np.degrees(tMean.as_rotvec()), DECIMALS)}°, 회차별 차이 {np.round(tSpread, DECIMALS)}° "
          f"(중앙 {np.median(tSpread):.1f}°)")
    if tArgs.write:
        tRecord = dict(R_U_E=tMean.inv().as_matrix().tolist(), R_E_U=tMean.as_matrix().tolist(), spread_deg=tSpread, take=str(tTake),
                       hand=tArgs.hand, note="robot target orientation = R_B_U * R_U_E (R_B_U = umi_traj R_B_tcp)")
        Path(tOutFile).write_text(json.dumps(tRecord, indent=JSON_INDENT))
        print("기록", tOutFile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
