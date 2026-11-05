"""[2026-11-01] user "laundry pinching 100개 더 쌓자, 연결은 다 했어, 확인해줘": which UMI camera / jaw motor / IMU belongs to which hand.
Runs --secs seconds: both wrist cameras (by USB location, mean grey), both Feetech jaw encoders (torque off, read only), both XIAO IMU
serial ports (bytes/s).  The operator covers the LEFT-hand camera lens and opens/closes the LEFT jaw (and shakes the LEFT unit) during
the run; the summary names the device that changed most.  Nothing is driven (motor torque is switched OFF, as in tag_rig_record).
    python -m laundry_tracking.calib.umi_side_check --cams 0x0111 0x0311 --motors /dev/cu.usbmodem5B3D0497471 /dev/cu.usbmodem5B3D0480821 \\
        --imus /dev/cu.usbmodem11401 /dev/cu.usbmodem3131101 --secs 40
(was scripts/umi_side_check.py of ego_collector; module path laundry_tracking.calib.umi_side_check unchanged by the 11-04 refactor)
"""
from __future__ import annotations

import argparse
import sys
import threading
import time

import numpy as np

CAM_W, CAM_H = 640, 360
IMU_BAUD = 115200
IMU_READ_TIMEOUT_S = 0.2
IMU_READ_BYTES = 4096
POLL_S = 0.5
STOP_SETTLE_S = 0.3


def _buildParser() -> argparse.ArgumentParser:
    tAp = argparse.ArgumentParser()
    tAp.add_argument("--cams", nargs="+", default=[])
    tAp.add_argument("--motors", nargs="+", default=[])
    tAp.add_argument("--imus", nargs="+", default=[])
    tAp.add_argument("--secs", type=float, default=40.0)
    return tAp


def main(argv: list[str] | None = None) -> int:
    tArgs = _buildParser().parse_args(argv)
    from laundry_shared.visual_teleop.tools.umi_jaw import open_cam
    from laundry_shared.visual_teleop.tools.tag_rig_record import MotorLog
    import serial
    tCams = {tLoc: open_cam(-1, CAM_W, CAM_H, location=tLoc) for tLoc in tArgs.cams}
    tMotors = {tP: MotorLog("x", tP) for tP in tArgs.motors}
    for tM in tMotors.values():
        tM.start()
    tImuBytes = {tP: 0 for tP in tArgs.imus}
    tStop = [False]

    def _imuReader(port):
        try:
            tSerial = serial.Serial(port, IMU_BAUD, timeout=IMU_READ_TIMEOUT_S)
            while not tStop[0]:
                tImuBytes[port] += len(tSerial.read(IMU_READ_BYTES))
        except Exception as tErr:                                         # report and stop this reader; the others keep running
            print(f"IMU {port} 오류: {tErr}", flush=True)
    for tP in tArgs.imus:
        threading.Thread(target=_imuReader, args=(tP,), daemon=True).start()
    tLog = {tK: [] for tK in list(tCams) + list(tMotors) + list(tImuBytes)}
    tT0 = time.monotonic()
    tLastN = dict(tImuBytes)
    print("시작 -- 왼손: 카메라 렌즈 3초 가리기 + 집게 여닫기 + 유닛 흔들기", flush=True)
    while time.monotonic() - tT0 < tArgs.secs:
        tRow = []
        for tLoc, tCam in tCams.items():
            tOk, tFrame = tCam.read()
            tGray = float(tFrame.mean()) if tOk else np.nan
            tLog[tLoc].append(tGray)
            tRow.append(f"cam {tLoc} {tGray:5.0f}")
        for tP, tM in tMotors.items():
            tLog[tP].append(tM.last if tM.last is not None else np.nan)
            tRow.append(f"motor {tP[-11:-1]} {tM.last}")
        for tP in tImuBytes:
            tRate = (tImuBytes[tP] - tLastN[tP]) / POLL_S
            tLog[tP].append(tRate)
            tRow.append(f"imu {tP[-8:]} {tRate:5.0f} B/s")
        tLastN = dict(tImuBytes)
        print(f"{time.monotonic() - tT0:4.1f}s  " + " | ".join(tRow), flush=True)
        time.sleep(POLL_S)
    tStop[0] = True
    for tM in tMotors.values():
        tM.stop_                                                          # no-op attribute read, kept as recorded (see the report)
    print("\n요약 (변화가 큰 쪽 = 왼손):", flush=True)
    for tLoc in tCams:
        tV = np.array(tLog[tLoc])
        print(f"  카메라 {tLoc}: 밝기 최소 {np.nanmin(tV):.0f} / 중앙 {np.nanmedian(tV):.0f}  (가려지면 크게 떨어짐)")
    for tP in tMotors:
        tV = np.array(tLog[tP], float)
        print(f"  모터 {tP}: 위치 범위 {np.nanmax(tV) - np.nanmin(tV):.0f} ticks  (min {np.nanmin(tV):.0f}, max {np.nanmax(tV):.0f})")
    for tP in tImuBytes:
        tV = np.array(tLog[tP])
        print(f"  IMU {tP}: 평균 {np.mean(tV):.0f} B/s  (0 이면 데이터 안 옴)")
    for tM in tMotors.values():
        tM.stop_ = True
    time.sleep(STOP_SETTLE_S)
    return 0


if __name__ == "__main__":
    sys.exit(main())
