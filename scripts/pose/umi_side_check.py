#!/usr/bin/env python3
"""[2026-11-01] user "laundry pinching 100개 더 쌓자, 연결은 다 했어, 확인해줘": which UMI camera / jaw motor / IMU belongs to which hand.
Runs --secs seconds: both wrist cameras (by USB location, mean grey), both Feetech jaw encoders (torque off, read only), both XIAO IMU
serial ports (bytes/s).  The operator covers the LEFT-hand camera lens and opens/closes the LEFT jaw (and shakes the LEFT unit) during
the run; the summary names the device that changed most.  Nothing is driven (motor torque is switched OFF, as in tag_rig_record).
  .venv/bin/python scripts/umi_side_check.py --cams 0x0111 0x0311 --motors /dev/cu.usbmodem5B3D0497471 /dev/cu.usbmodem5B3D0480821 \
      --imus /dev/cu.usbmodem11401 /dev/cu.usbmodem3131101 --secs 40"""
import argparse, sys, threading, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cams", nargs="+", default=[]); ap.add_argument("--motors", nargs="+", default=[])
    ap.add_argument("--imus", nargs="+", default=[]); ap.add_argument("--secs", type=float, default=40.0); a = ap.parse_args()
    from visual_teleop.tools.umi_jaw import open_cam
    from visual_teleop.tools.tag_rig_record import MotorLog
    import serial
    cams = {loc: open_cam(-1, 640, 360, location=loc) for loc in a.cams}
    mots = {p: MotorLog("x", p) for p in a.motors}
    for m in mots.values(): m.start()
    imu_n = {p: 0 for p in a.imus}; stop = [False]
    def imu_reader(p):
        try:
            s = serial.Serial(p, 115200, timeout=0.2)
            while not stop[0]: imu_n[p] += len(s.read(4096))
        except Exception as e: print(f"IMU {p} 오류: {e}", flush=True)
    for p in a.imus: threading.Thread(target=imu_reader, args=(p,), daemon=True).start()
    log = {k: [] for k in list(cams) + list(mots) + list(imu_n)}; t0 = time.monotonic(); last_n = dict(imu_n)
    print("시작 -- 왼손: 카메라 렌즈 3초 가리기 + 집게 여닫기 + 유닛 흔들기", flush=True)
    while time.monotonic() - t0 < a.secs:
        row = []
        for loc, c in cams.items():
            ok, f = c.read(); g = float(f.mean()) if ok else np.nan; log[loc].append(g); row.append(f"cam {loc} {g:5.0f}")
        for p, m in mots.items(): log[p].append(m.last if m.last is not None else np.nan); row.append(f"motor {p[-11:-1]} {m.last}")
        for p in imu_n: r = (imu_n[p] - last_n[p]) / 0.5; log[p].append(r); row.append(f"imu {p[-8:]} {r:5.0f} B/s")
        last_n = dict(imu_n); print(f"{time.monotonic() - t0:4.1f}s  " + " | ".join(row), flush=True); time.sleep(0.5)
    stop[0] = True
    for m in mots.values(): m.stop_
    print("\n요약 (변화가 큰 쪽 = 왼손):", flush=True)
    for loc in cams: v = np.array(log[loc]); print(f"  카메라 {loc}: 밝기 최소 {np.nanmin(v):.0f} / 중앙 {np.nanmedian(v):.0f}  (가려지면 크게 떨어짐)")
    for p in mots: v = np.array(log[p], float); print(f"  모터 {p}: 위치 범위 {np.nanmax(v) - np.nanmin(v):.0f} ticks  (min {np.nanmin(v):.0f}, max {np.nanmax(v):.0f})")
    for p in imu_n: v = np.array(log[p]); print(f"  IMU {p}: 평균 {np.mean(v):.0f} B/s  (0 이면 데이터 안 옴)")
    for m in mots.values(): m.stop_ = True
    time.sleep(0.3)


if __name__ == "__main__":
    main()
