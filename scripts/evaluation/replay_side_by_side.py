"""Side-by-side video of a human UMI demo and its robot replay, both seen by the Orbbec (2026-10-21, user request).
Left = the take's colour frames (human demo), right = an Orbbec recording made DURING the replay (tag_rig_record --proto robotcal).
Both recorders and umi_replay_bi stamp host time.monotonic_ns, and the replay log maps every command (t_cmd_ns) to its source time
(src_t_ns), so each output frame at human time t shows the robot frame recorded when the command for t was sent -> same speed as the
human, whatever --speed the replay ran at. Bottom bar: jaw command (robot) per arm.
    ~/xvla-mac/bin/python scripts/evaluation/replay_side_by_side.py datasets/tag_rig/<take> datasets/replay/bi_<ts> datasets/tag_rig/<robot_rec>
        [--out x.mp4] [--fps 15] [--slow 1.0]"""
from __future__ import annotations
import argparse, shutil, subprocess
from pathlib import Path
import cv2
import numpy as np


def nearest(ts: np.ndarray, t: float) -> int:
    i = int(np.clip(np.searchsorted(ts, t), 1, len(ts) - 1)); return i if abs(ts[i] - t) < abs(ts[i - 1] - t) else i - 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take"); ap.add_argument("replay"); ap.add_argument("robot_rec"); ap.add_argument("--out")
    ap.add_argument("--fps", type=float, default=15.0); ap.add_argument("--slow", type=float, default=1.0, help="출력 재생 배속의 역수 (2 = 두 배 느리게)")
    ap.add_argument("--w", type=int, default=960)
    ap.add_argument("--robot-time", action="store_true", help="로봇 실제 재생 시간 그대로 (사람 영상을 그만큼 느리게) -- 10-21 사용자 요청 94 s 전체"); a = ap.parse_args()
    H, R, V = Path(a.take), Path(a.replay), Path(a.robot_rec)
    th = np.load(H / "t_ns.npy").astype(float)
    if (V / "t_ns.npy").exists(): tv = np.load(V / "t_ns.npy").astype(float)
    else:       # 10-21: recorder suspended (Ctrl+Z) before saving t_ns -> jpg write times (wall) mapped to monotonic, ~10-30 ms late
        import os, time; off = time.time_ns() - time.monotonic_ns(); fs = sorted(V.glob("color_*.jpg"))
        tv = np.array([os.stat(f).st_mtime_ns - off for f in fs], float); print(f"t_ns.npy 없음 -> 파일 저장 시각으로 대신 ({len(tv)} 프레임)")
    sides = [s for s in ("right", "left") if (R / s / "seg_00.npz").exists()]
    Z = {s: np.load(R / s / "seg_00.npz") for s in sides}; z0 = Z[sides[0]]
    src, tc = z0["src_t_ns"].astype(float), z0["t_cmd_ns"].astype(float)
    if tc[0] < tv[0] or tc[-1] > tv[-1]: print(f"주의: 로봇 녹화가 재생 전체를 덮지 않음 (녹화 {tv[0] / 1e9:.1f}-{tv[-1] / 1e9:.1f}s, 재생 {tc[0] / 1e9:.1f}-{tc[-1] / 1e9:.1f}s)")
    out = Path(a.out) if a.out else R / "side_by_side.mp4"; w = a.w; h = w * 9 // 16; bar = 40
    tmp = out.with_suffix(".raw.mp4"); vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (2 * w, h + bar))
    ts = np.arange(src[0], src[-1], 1e9 / (a.fps * a.slow))     # one output frame per 1/(fps*slow) s of HUMAN time
    if a.robot_time: ts = np.interp(np.arange(tc[0], tc[-1], 1e9 / (a.fps * a.slow)), tc, src)     # same frames, robot clock spacing
    for k, t in enumerate(ts):
        ih = nearest(th, t); trob = float(np.interp(t, src, tc)); iv = nearest(tv, trob)
        fh = cv2.imread(str(H / f"color_{ih:05d}.jpg")); fv = cv2.imread(str(V / f"color_{iv:05d}.jpg"))
        if fh is None or fv is None: continue
        fh, fv = cv2.resize(fh, (w, h)), cv2.resize(fv, (w, h)); el = (t - src[0]) / 1e9
        cv2.putText(fh, f"HUMAN demo  t={el:5.1f}s", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(fv, f"ROBOT replay  t={el:5.1f}s  (real {(trob - tc[0]) / 1e9:5.1f}s)", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        if abs(tv[iv] - trob) > 0.2e9: cv2.putText(fv, "NO ROBOT FRAME", (16, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        b = np.zeros((bar, 2 * w, 3), np.uint8); j = int(np.clip(np.searchsorted(src, t), 0, len(src) - 1))
        for n_, s in enumerate([x for x in ("left", "right") if x in sides]):     # robot jaw commands, under the ROBOT half: left then right (10-22 user)
            g = float(Z[s]["g_cmd"][j]); c = (0, 200, 0) if g > 0.5 else (0, 0, 230)
            cv2.putText(b, f"robot {s} jaw {'OPEN' if g > 0.5 else 'CLOSED'}", (w + 16 + n_ * w // 2, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)
        vw.write(np.vstack([np.hstack([fh, fv]), b]))
    vw.release()
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out)], check=True); tmp.unlink()
    else: tmp.rename(out)
    print(f"저장 {out} ({len(ts)} 프레임, {len(ts) / a.fps:.1f}s)")


if __name__ == "__main__":
    main()
