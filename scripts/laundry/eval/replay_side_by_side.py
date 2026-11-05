"""Side-by-side video of a human UMI demo and its robot replay, both seen by the Orbbec (2026-10-21, user request; was
scripts/evaluation/replay_side_by_side.py).

Pipeline stage: D7 comparison of the replay track (visual). Left = the take's colour frames (human demo), right = an Orbbec recording
made DURING the replay (tag_rig_record --proto robotcal). Both recorders and umi_replay_bi stamp host time.monotonic_ns, and the replay
log maps every command (t_cmd_ns) to its source time (src_t_ns), so each output frame at human time t shows the robot frame recorded
when the command for t was sent -> same speed as the human, whatever --speed the replay ran at. Bottom bar: jaw command (robot) per
arm. Output: <replay>/side_by_side.mp4 (or --out), H.264 when ffmpeg is on PATH.
    python scripts/laundry/eval/replay_side_by_side.py datasets/tag_rig/<take> datasets/replay/bi_<ts> datasets/tag_rig/<robot_rec>
        [--out x.mp4] [--fps 15] [--slow 1.0]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

NS_PER_S = 1e9
BAR_H = 40
# a robot frame further than this from the wanted time is flagged on the video
MAX_ROBOT_FRAME_GAP_NS = 0.2e9
HUMAN_COLOUR = (0, 255, 0)
ROBOT_COLOUR = (0, 200, 255)
MISSING_COLOUR = (0, 0, 255)
JAW_OPEN_COLOUR = (0, 200, 0)
JAW_CLOSED_COLOUR = (0, 0, 230)
# a jaw command counts as open above this
JAW_OPEN_MIN = 0.5
ASPECT_W, ASPECT_H = 16, 9
DEFAULT_FPS = 15.0
DEFAULT_SLOW = 1.0
DEFAULT_WIDTH = 960
FFMPEG_ARGS = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23"]


def nearest(ts: np.ndarray, t: float) -> int:
    """Index of the stamp in sorted `ts` closest to `t` (never index 0 unless it is closer than index 1)."""
    tI = int(np.clip(np.searchsorted(ts, t), 1, len(ts) - 1))
    return tI if abs(ts[tI] - t) < abs(ts[tI - 1] - t) else tI - 1


def _robotFrameTimes(robotRec: Path) -> np.ndarray:
    if (robotRec / "t_ns.npy").exists():
        return np.load(robotRec / "t_ns.npy").astype(float)
    # 10-21: recorder suspended (Ctrl+Z) before saving t_ns -> jpg write times (wall) mapped to monotonic, ~10-30 ms late
    tOffset = time.time_ns() - time.monotonic_ns()
    tFiles = sorted(robotRec.glob("color_*.jpg"))
    tTimes = np.array([os.stat(tFile).st_mtime_ns - tOffset for tFile in tFiles], float)
    print(f"t_ns.npy 없음 -> 파일 저장 시각으로 대신 ({len(tTimes)} 프레임)")
    return tTimes


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tParser.add_argument("take")
    tParser.add_argument("replay")
    tParser.add_argument("robot_rec")
    tParser.add_argument("--out")
    tParser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    tParser.add_argument("--slow", type=float, default=DEFAULT_SLOW, help="출력 재생 배속의 역수 (2 = 두 배 느리게)")
    tParser.add_argument("--w", type=int, default=DEFAULT_WIDTH)
    tParser.add_argument("--robot-time", action="store_true", help="로봇 실제 재생 시간 그대로 (사람 영상을 그만큼 느리게) -- 10-21 사용자 요청 94 s 전체")
    return tParser


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    tHuman, tReplay, tRobotRec = Path(tArgs.take), Path(tArgs.replay), Path(tArgs.robot_rec)
    tHumanT = np.load(tHuman / "t_ns.npy").astype(float)
    tRobotT = _robotFrameTimes(tRobotRec)
    tSides = [tSide for tSide in ("right", "left") if (tReplay / tSide / "seg_00.npz").exists()]
    tSegments = {tSide: np.load(tReplay / tSide / "seg_00.npz") for tSide in tSides}
    tFirst = tSegments[tSides[0]]
    tSrc, tCmd = tFirst["src_t_ns"].astype(float), tFirst["t_cmd_ns"].astype(float)
    if tCmd[0] < tRobotT[0] or tCmd[-1] > tRobotT[-1]:
        print(f"주의: 로봇 녹화가 재생 전체를 덮지 않음 (녹화 {tRobotT[0] / NS_PER_S:.1f}-{tRobotT[-1] / NS_PER_S:.1f}s, "
              f"재생 {tCmd[0] / NS_PER_S:.1f}-{tCmd[-1] / NS_PER_S:.1f}s)")
    tOut = Path(tArgs.out) if tArgs.out else tReplay / "side_by_side.mp4"
    tW = tArgs.w
    tH = tW * ASPECT_H // ASPECT_W
    tRaw = tOut.with_suffix(".raw.mp4")
    tWriter = cv2.VideoWriter(str(tRaw), cv2.VideoWriter_fourcc(*"mp4v"), tArgs.fps, (2 * tW, tH + BAR_H))
    tFrameStep = NS_PER_S / (tArgs.fps * tArgs.slow)
    tTimes = np.arange(tSrc[0], tSrc[-1], tFrameStep)     # one output frame per 1/(fps*slow) s of HUMAN time
    if tArgs.robot_time:
        tTimes = np.interp(np.arange(tCmd[0], tCmd[-1], tFrameStep), tCmd, tSrc)     # same frames, robot clock spacing
    for tT in tTimes:
        tIh = nearest(tHumanT, tT)
        tRobotTime = float(np.interp(tT, tSrc, tCmd))
        tIv = nearest(tRobotT, tRobotTime)
        tFh = cv2.imread(str(tHuman / f"color_{tIh:05d}.jpg"))
        tFv = cv2.imread(str(tRobotRec / f"color_{tIv:05d}.jpg"))
        if tFh is None or tFv is None:
            continue
        tFh, tFv = cv2.resize(tFh, (tW, tH)), cv2.resize(tFv, (tW, tH))
        tElapsed = (tT - tSrc[0]) / NS_PER_S
        cv2.putText(tFh, f"HUMAN demo  t={tElapsed:5.1f}s", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, HUMAN_COLOUR, 2)
        cv2.putText(tFv, f"ROBOT replay  t={tElapsed:5.1f}s  (real {(tRobotTime - tCmd[0]) / NS_PER_S:5.1f}s)", (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, ROBOT_COLOUR, 2)
        if abs(tRobotT[tIv] - tRobotTime) > MAX_ROBOT_FRAME_GAP_NS:
            cv2.putText(tFv, "NO ROBOT FRAME", (16, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, MISSING_COLOUR, 2)
        tBar = np.zeros((BAR_H, 2 * tW, 3), np.uint8)
        tJ = int(np.clip(np.searchsorted(tSrc, tT), 0, len(tSrc) - 1))
        # robot jaw commands, under the ROBOT half: left then right (10-22 user)
        for tK, tSide in enumerate([tX for tX in ("left", "right") if tX in tSides]):
            tG = float(tSegments[tSide]["g_cmd"][tJ])
            tColour = JAW_OPEN_COLOUR if tG > JAW_OPEN_MIN else JAW_CLOSED_COLOUR
            cv2.putText(tBar, f"robot {tSide} jaw {'OPEN' if tG > JAW_OPEN_MIN else 'CLOSED'}", (tW + 16 + tK * tW // 2, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, tColour, 2)
        tWriter.write(np.vstack([np.hstack([tFh, tFv]), tBar]))
    tWriter.release()
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tRaw), *FFMPEG_ARGS, str(tOut)], check=True)
        tRaw.unlink()
    else:
        tRaw.rename(tOut)
    print(f"저장 {tOut} ({len(tTimes)} 프레임, {len(tTimes) / tArgs.fps:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
