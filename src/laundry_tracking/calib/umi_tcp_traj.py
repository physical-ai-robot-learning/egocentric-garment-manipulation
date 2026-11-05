"""UMI TCP trajectory from a tag_rig_record take (2026-10-20, replay parity A = "UMI measured").  Per colour frame with >= 1 tag of that
UMI: T_C_rig (laundry_shared rebot_prep/parity/tag_rig_fuse, rotation imu|tag, optional depth range) -> TCP point
p_C = T_C_rig * tcp_in_rig_m (touch-TCP calibration, umi_tag_rig_v002.json), then into the robot base with T_B_C
(robot_cam_v001.json) when it exists.  For --proto axes takes it also writes the stroke segments (hand, axis, direction, repeat,
outward / return / hold windows in host ns).
Robot base convention used for the expected directions: +X 앞, +Y 왼쪽, +Z 위 (reBot URDF base, right READY TCP ~ (0.45, -0.30, 0.28)).
-> <take>/umi_traj_<right|left>.npz (t_ns, p_C, R_C_tcp, ntags, single_new_tag, guard_rot_deg[, p_B, R_B_tcp]) + <take>/segments.json
   (unwritable take -> <take>_derived/)
10-23 v002: single_new_tag=True marks frames whose pose came ONLY from the lone 30 mm camera-back tag (220/221) after the IMU flip
guard (tag_rig_fuse step 4); guard_rot_deg = its rotation residual vs the gyro prediction (NaN elsewhere).  Down-weight these.
    python -m laundry_tracking.calib.umi_tcp_traj datasets/tag_rig/<take> --hand 오른손 [--rot imu|tag] [--depth] [--robot-cal <json>]
(was scripts/pose/umi_tcp_traj.py of ego_collector; module path laundry_tracking.calib.umi_tcp_traj unchanged by the 11-04 refactor)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from laundry_common import paths
from laundry_shared.rebot_prep.parity.tag_rig_fuse import SIDE, fuse_take

DEFAULT_RIG = paths.CONFIGS / "calibration/umi_tag_rig_v002.json"
DEFAULT_ROBOT_CAL = paths.CONFIGS / "calibration/robot_cam_v001.json"
DIRS = {"앞으로": ("X", +1), "뒤로": ("X", -1), "왼쪽으로": ("Y", +1), "오른쪽으로": ("Y", -1), "위로": ("Z", +1), "아래로": ("Z", -1),
        # 10-20 --proto rot: rotation about the robot-BASE axis through the jaw tip (right-hand rule; READY approach axis = base +X):
        # yaw = about +Z (turn left = +), pitch = about +Y (tip down = +), roll = about +X (clockwise seen from behind = +)
        "왼쪽돌리기": ("YAW", +1), "오른쪽돌리기": ("YAW", -1), "끝아래로": ("PITCH", +1), "끝위로": ("PITCH", -1),
        "시계방향비틀기": ("ROLL", +1), "반시계방향비틀기": ("ROLL", -1)}
PHYS = {"YAW": "sweep_left_right(x_E)", "PITCH": "bend_up_down(y_E)", "ROLL": "twist_jaw_axis(z_E)"}
FREE_TRIAL_CUE = ", 자유동작, "
FREE_CUE_PREFIX = "자유:"
SEGMENT_PROTOS = ("axes", "rot", "free", "grip", "task")


def segments(meta: dict) -> list[dict]:
    """axes protocol -> strokes.  sched entries: hold '<hand>, <ax>축, <r>회, 멈추세요' -> move '<hand> <dir> <n>센티 천천히' -> hold -> return"""
    tT0 = int(meta["t_start_monotonic_ns"])
    tSched = meta["protocol"]
    tOut = []

    def _ns(entry, frac=0.0):
        return int(tT0 + (entry["t0"] + frac * entry["dur"]) * 1e9)
    for tI, tE in enumerate(tSched):
        if FREE_TRIAL_CUE in tE["text"]:       # 10-20 --proto free: hold -> consecutive '자유:' cues -> hold (one segment per trial)
            tJ = tI + 1
            while tJ < len(tSched) and tSched[tJ]["text"].startswith(FREE_CUE_PREFIX):
                tJ += 1
            if tJ == tI + 1 or tJ >= len(tSched):
                continue
            tHoldPre = [_ns(tSched[tI]), _ns(tSched[tI], 1)]
            tHoldEnd = [_ns(tSched[tJ]), _ns(tSched[tJ], 1)]
            tOut.append(dict(hand=tE["text"].split(",")[0], axis="FREE", sign=1, dir="자유", phys="free",
                             rep=int(tE["text"].split(",")[2].strip().rstrip("회")),
                             cues=[tX["text"][4:] for tX in tSched[tI + 1:tJ]], hold_pre=tHoldPre,
                             out=[_ns(tSched[tI + 1]), _ns(tSched[tJ - 1], 1)], hold_mid=tHoldEnd, back=None, hold_post=tHoldEnd))
            continue
        tWords = tE["text"].split()
        if not ("센티 천천히" in tE["text"] or "도 천천히" in tE["text"]) or len(tWords) < 3 or tWords[1] not in DIRS:
            continue
        tHold = tSched[tI - 1]["text"].split(",")
        tAx, tSign = DIRS[tWords[1]]
        tOut.append(dict(hand=tWords[0], axis=tAx, sign=tSign, dir=tWords[1], phys=PHYS.get(tAx, "translation"),
                         rep=int(tHold[2].strip().rstrip("회")),
                         cm=float(tWords[2].replace("센티", "")) if "센티" in tWords[2] else None,
                         deg=float(tWords[2].replace("도", "")) if tWords[2].endswith("도") else None,
                         hold_pre=[_ns(tSched[tI - 1]), _ns(tSched[tI - 1], 1)], out=[_ns(tE), _ns(tE, 1)],
                         hold_mid=[_ns(tSched[tI + 1]), _ns(tSched[tI + 1], 1)],
                         back=[_ns(tSched[tI + 2]), _ns(tSched[tI + 2], 1)],
                         hold_post=[_ns(tSched[tI + 3]), _ns(tSched[tI + 3], 1)] if tI + 3 < len(tSched) else None))
    return tOut


def _buildParser() -> argparse.ArgumentParser:
    tAp = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tAp.add_argument("take")
    tAp.add_argument("--hand", default="오른손", choices=list(SIDE))
    tAp.add_argument("--rot", default="imu", choices=["imu", "tag"])
    tAp.add_argument("--depth", action="store_true")
    tAp.add_argument("--imu-fill", action="store_true", help="태그 끊긴 구간 회전을 IMU 로 채움 (위치는 보간, ntags=0 표시)")
    tAp.add_argument("--tag-size", type=float, default=0.048)
    tAp.add_argument("--rig", default=str(DEFAULT_RIG))
    tAp.add_argument("--robot-cal", default=str(DEFAULT_ROBOT_CAL))
    return tAp


def main(argv: list[str] | None = None) -> int:
    tArgs = _buildParser().parse_args(argv)
    tTake = Path(tArgs.take)
    tMeta = json.loads((tTake / "meta.json").read_text())
    tRig = json.loads(Path(tArgs.rig).read_text())
    tName, tSide = SIDE[tArgs.hand]
    tTcp = np.array(tRig[tName]["tcp_in_rig_m"])
    F = fuse_take(tTake, tArgs.hand, tRig, tArgs.tag_size, use_depth=tArgs.depth, rot_source=tArgs.rot, fill_gaps=tArgs.imu_fill)
    if F is None:
        print(f"{tArgs.hand} 유미 태그가 한 번도 안 보임", file=sys.stderr)
        return 1
    tT = F["T_C_rig"]
    tPC = tT[:, :3, :3] @ tTcp + tT[:, :3, 3]
    tRes = dict(t_ns=F["t_ns"], p_C=tPC, R_C_tcp=tT[:, :3, :3], ntags=F["ntags"], rot=tArgs.rot, depth=tArgs.depth,
                single_new_tag=F["single_new_tag"], guard_rot_deg=F["guard_rot_deg"])
    tGs = F["guard_stats"]
    if tGs["single_new_tag_frames"]:
        print(f"카메라 뒤 30mm 태그만 보인 프레임 {tGs['single_new_tag_frames']}: IMU 뒤집힘 방지 통과 {tGs['accepted']}, 거부 회전 {tGs['rejected_rot']} / "
              f"점프 {tGs['rejected_jump']} / 이전 자세 없음 {tGs['rejected_no_prior']} / IMU 없음 {tGs['rejected_no_imu']} (통과 프레임은 single_new_tag=True)")
    if tArgs.imu_fill and "t_all_ns" in F:
        # every colour frame: IMU-anchored orientation; TCP position interpolated across tag gaps (marked ntags=0 -> not a measurement)
        tTa = F["t_all_ns"]
        tPa = np.stack([np.interp(tTa, F["t_ns"], tPC[:, tK]) for tK in range(3)], 1)
        tNa = np.zeros(len(tTa), int)
        tIdx = np.searchsorted(tTa, F["t_ns"])
        tNa[tIdx] = F["ntags"]
        tGa = np.full(len(tTa), np.nan)
        tGa[tIdx] = F["guard_rot_deg"]
        tSa = np.zeros(len(tTa), bool)
        tSa[tIdx] = F["single_new_tag"]
        tRes = dict(t_ns=tTa, p_C=tPa, R_C_tcp=F["R_C_rig_all"], ntags=tNa, rot=tArgs.rot, depth=tArgs.depth, imu_fill=True,
                    single_new_tag=tSa, guard_rot_deg=tGa)
        print(f"IMU 로 채움: 프레임 {len(tTa)} 중 태그 있음 {int((tNa > 0).sum())}, 회전만 IMU {int((tNa == 0).sum())}")
    if Path(tArgs.robot_cal).exists():
        tTbc = np.array(json.loads(Path(tArgs.robot_cal).read_text())["T_B_C"])
        tRes.update(p_B=tRes["p_C"] @ tTbc[:3, :3].T + tTbc[:3, 3], R_B_tcp=tTbc[:3, :3] @ tRes["R_C_tcp"],
                    robot_cal=tArgs.robot_cal)
        print(f"로봇 베이스 좌표로 변환 ({tArgs.robot_cal})")
    else:
        print(f"[주의] 로봇 캘리브레이션 없음 ({tArgs.robot_cal}) -- 카메라 좌표만 저장 (replay_parity 는 --cam-axes 로 근사 비교만 가능)")
    tOut = tTake if os.access(tTake, os.W_OK) else tTake.parent / (tTake.name + "_derived")
    tOut.mkdir(exist_ok=True)
    np.savez(tOut / f"umi_traj_{tSide}.npz", **tRes)
    tGaps = np.diff(F["t_ns"]) / 1e9
    tPC = tRes["p_C"]
    print(f"{tArgs.hand}: 프레임 {len(tPC)}/{tMeta['frames']} ({len(tPC) / max(tMeta['frames'], 1) * 100:.0f}%), 태그 2개 이상 {np.mean(F['ntags'] >= 2) * 100:.0f}%, "
          f"최대 끊김 {tGaps.max() if len(tGaps) else 0:.2f}s, 태그-융합 회전차 중앙 {np.median(F['tag_vs_fused_deg']):.1f}° -> {tOut / f'umi_traj_{tSide}.npz'}")
    if tMeta.get("proto") in SEGMENT_PROTOS:
        tSeg = segments(tMeta)
        (tOut / "segments.json").write_text(json.dumps(tSeg, ensure_ascii=False, indent=1))
        print(f"구간 {len(tSeg)}개 -> {tOut / 'segments.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
