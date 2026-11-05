"""Stage-A bimanual robotization of a human UMI take (2026-10-21, GPT: human_canonical_tcp -> robot_feasible_tcp before IK; was
scripts/retarget/umi_robotize_bi.py).

Pipeline stage: D4 -> D5 robotization of the replay track (training labels use laundry_dataset's robotize IK instead).
113243 replayed as-is failed: the right hand crossed the body midline (y up to +0.19 m), hands came within 3-11 cm, both jaws were
pressed below the table. Per demo window (segments.json rep) and per hand, the POSITION is warped continuously (never clipped):
  y   right hand: affine compression toward its outer edge so its p99 stays at <= -sep/2; left hand mirrored (p1 >= +sep/2);
      then a smooth soft-limit for whatever still crosses (no hard clamp)
  z   smooth soft floor at table + clearance (softplus, width --soft-w), so contact segments are lifted without a kink
  xy  (--min-reach) pushed radially out to at least --min-reach from the arm's own base, softly
Orientation is left to the IK (umi_replay_bi --ori-cost relaxes it). Input: <take>/umi_traj_<side><src>.npz + segments.json.
Output: <take>/umi_traj_<side><src>_robot.npz (same keys, p_B replaced, p_B_human kept, robotize_* per-frame quality) and a printed
summary of how much was changed. Feasibility (IK / link-link gap / table) is then checked by
    umi_replay_bi <take> --traj _robot --rep N --offline
    python scripts/laundry/retarget/umi_robotize_bi.py datasets/tag_rig/<take> [--sep 0.16] [--table-z 0.003 --clear 0.035]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SIDES = ("right", "left")
MIN_WINDOW_FRAMES = 10
# y extent of the side warp: these percentiles (outer / inner edge)
LOW_PERCENTILE, HIGH_PERCENTILE = 1, 99
MIN_EXTENT_M = 1e-6
MIN_RADIUS_M = 1e-6
# the reach floor is twice as soft as the y / z warps
REACH_SOFTNESS = 2
# 10-21 (GPT, training on robot_feasible_tcp): per-frame quality so a dataset builder can weight / drop frames whose action label
# moved far from what the wrist video shows. class 0 < 20 mm (train as is), 1 < 50 mm (lower weight), 2 >= 50 mm (replace / drop)
QUALITY_EDGES_M = [0.02, 0.05]
# printed: share of frames moved more than this
CHANGED_M = 0.005
MM_PER_M = 1000
PERCENT = 100
P95 = 95
DEFAULT_SEP_M = 0.16
DEFAULT_TABLE_Z_M = 0.003
DEFAULT_CLEAR_M = 0.035
DEFAULT_SOFT_W_M = 0.01
DEFAULT_MIN_REACH_M = 0.30
DEFAULT_BASE_Y_M = 0.315
DEFAULT_MIN_SCALE = 0.4


def softFloor(z, f, w):
    return f + w * np.logaddexp(0.0, (z - f) / w)


def softCeil(y, c, w):
    return -softFloor(-y, -c, w)


def _sideWarp(y, side, sep, softW, minScale):
    """(warped y, scale) of one window: compress toward the hand's outer edge, then a soft limit at -sep/2 (right) / +sep/2 (left)."""
    tHalf = sep / 2
    if side == "right":
        tOuter, tInner = np.percentile(y, LOW_PERCENTILE), np.percentile(y, HIGH_PERCENTILE)
        tScale = 1.0 if tInner <= -tHalf else float(np.clip((-tHalf - tOuter) / max(tInner - tOuter, MIN_EXTENT_M), minScale, 1.0))
        return softCeil(tOuter + tScale * (y - tOuter), -tHalf, softW), tScale
    tOuter, tInner = np.percentile(y, HIGH_PERCENTILE), np.percentile(y, LOW_PERCENTILE)
    tScale = 1.0 if tInner >= tHalf else float(np.clip((tOuter - tHalf) / max(tOuter - tInner, MIN_EXTENT_M), minScale, 1.0))
    return softFloor(tOuter + tScale * (y - tOuter), tHalf, softW), tScale


def robotizeSide(traj: dict, side: str, segments: list[dict], reps: list, args) -> None:
    """Warp traj["p_B"] of one hand per demo window in place (adds p_B_human, robotize*, printing per window)."""
    tT = traj["t_ns"].astype(np.int64)
    tP = traj["p_B"].astype(float).copy()
    tHuman = tP.copy()
    for tRep in reps:
        tSeg = [tX for tX in segments if tX["rep"] == tRep][0]
        tT0 = tSeg["hold_pre"][0]
        tT1 = (tSeg["hold_post"] or tSeg["back"])[1]
        tMask = (tT >= tT0) & (tT <= tT1)
        if tMask.sum() < MIN_WINDOW_FRAMES:
            continue
        tY = tP[tMask, 1]
        tY2, tScale = _sideWarp(tY, side, args.sep, args.soft_w, args.min_scale)
        tP[tMask, 1] = tY2
        tP[tMask, 2] = softFloor(tP[tMask, 2], args.table_z + args.clear, args.soft_w)
        # 10-21: left IK failed on targets 18-39 cm from its own base, low (x 0.14-0.35, z 0.04-0.19) -> push radially out
        if args.min_reach > 0:
            tBase = np.array([0.0, args.base_y if side == "left" else -args.base_y])
            tV = tP[tMask, :2] - tBase
            tRadius = np.linalg.norm(tV, axis=1)
            tRadius2 = softFloor(tRadius, args.min_reach, args.soft_w * REACH_SOFTNESS)
            tP[tMask, :2] = tBase + tV * (tRadius2 / np.maximum(tRadius, MIN_RADIUS_M))[:, None]
        tD = np.linalg.norm(tP[tMask] - tHuman[tMask], axis=1)
        print(f"{side} {tRep}회: y 범위 사람 [{np.percentile(tY, LOW_PERCENTILE):+.3f}, {np.percentile(tY, HIGH_PERCENTILE):+.3f}] -> 로봇 "
              f"[{np.percentile(tP[tMask, 1], LOW_PERCENTILE):+.3f}, {np.percentile(tP[tMask, 1], HIGH_PERCENTILE):+.3f}] "
              f"(압축 {tScale:.2f}), "
              f"z 최저 {tHuman[tMask, 2].min():+.3f} -> {tP[tMask, 2].min():+.3f} | 위치 변경 중앙 {np.median(tD) * MM_PER_M:.0f} mm "
              f"p95 {np.percentile(tD, P95) * MM_PER_M:.0f} mm, 5 mm 넘게 바뀐 프레임 {np.mean(tD > CHANGED_M) * PERCENT:.0f}%")
    tDisp = np.linalg.norm(tP - tHuman, axis=1)
    tLift = tP[:, 2] - tHuman[:, 2]
    traj.update(p_B_human=tHuman, p_B=tP, robotize=json.dumps(vars(args)), robotize_disp_m=tDisp, robotize_lift_z_m=tLift,
                robotize_quality=np.digitize(tDisp, QUALITY_EDGES_M).astype(np.int8))


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tParser.add_argument("take")
    tParser.add_argument("--sep", type=float, default=DEFAULT_SEP_M, help="두 손 y 사이 최소 간격 (m), 중앙선 = 0")
    tParser.add_argument("--table-z", type=float, default=DEFAULT_TABLE_Z_M)
    tParser.add_argument("--clear", type=float, default=DEFAULT_CLEAR_M, help="TCP 의 테이블 위 최소 높이 (m)")
    tParser.add_argument("--soft-w", type=float, default=DEFAULT_SOFT_W_M)
    tParser.add_argument("--min-reach", type=float, default=DEFAULT_MIN_REACH_M, help="베이스에서 수평 최소 거리 (m), 0 = 끔")
    tParser.add_argument("--base-y", type=float, default=DEFAULT_BASE_Y_M, help="팔 베이스 y (좌 +, 우 -)")
    tParser.add_argument("--src", default="", help="입력 궤적 접미사 (예: _fused) -> 출력 umi_traj_<손><src>_robot.npz")
    tParser.add_argument("--min-scale", type=float, default=DEFAULT_MIN_SCALE)
    return tParser


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    tTake = Path(tArgs.take)
    tSegments = json.loads((tTake / "segments.json").read_text())
    tReps = sorted({tSeg["rep"] for tSeg in tSegments})
    tTrajs = {tSide: dict(np.load(tTake / f"umi_traj_{tSide}{tArgs.src}.npz", allow_pickle=True)) for tSide in SIDES}
    for tSide, tTraj in tTrajs.items():
        robotizeSide(tTraj, tSide, tSegments, tReps, tArgs)
        np.savez(tTake / f"umi_traj_{tSide}{tArgs.src}_robot.npz", **tTraj)
        tQuality = tTraj["robotize_quality"]
        print(f"{tSide} 프레임 품질: 작은 수정(<20 mm) {np.mean(tQuality == 0) * PERCENT:.0f}%, 중간(<50) {np.mean(tQuality == 1) * PERCENT:.0f}%, "
              f"큰 수정(>=50) {np.mean(tQuality == 2) * PERCENT:.0f}%")
    print(f"저장 {tTake}/umi_traj_{{right,left}}_robot.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
