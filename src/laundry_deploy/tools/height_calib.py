"""[2026-11-01] Records the robot <-> training-frame height calibration (core.height_calib holds the file and its evaluation).

user "off를 임의로 주기보다 로봇에 맞추어서" + "왼쪽 에이프릴 태그로 면을 맞추자": an arm is parked (torque off), the operator puts the closed
jaw tip on a tag centre (or anywhere on the table), presses Enter, and the TCP = FK(measured q) with the same arms / FK as the deploy
is recorded. Per arm: dz = TCP z - anchor z (tags: also dx, dy), and a table plane from all touches of that arm.
  measure:  python -m laundry_deploy.tools.height_calib measure      (Korean prompts; result -> height_calib.json, older kept as .prev)
  point:    python -m laundry_deploy.tools.height_calib point l 400  (one touch, non-interactive -> height_calib_pending.json)
  finish:   python -m laundry_deploy.tools.height_calib finish       (pending touches -> height_calib.json)
  camxy / anchorxy: per-arm xy shifts (camera ray / anchors file), show: print the stored calibration
Was laundry_deploy/height_calib.py.
"""

from __future__ import annotations
import argparse
import json
import shutil
import sys
import time

import numpy as np

from ..core import height_calib as heightCalib
from ..core.contract import SIDES
from ..core.kinematics import fk, makeArms

TOUCH_READS = 10  # 0.5 s of reads: a hand-held arm wobbles a little
TOUCH_READ_S = 0.05
WOBBLE_WARN_DEG = 0.5
BRIDGE_TIMEOUT_S = 0.5
ARM_KEYS = {"l": "left", "r": "right"}
TAG_TARGETS = ("400", "401", "t")
STREAM_URL = "http://127.0.0.1:8770/stream/global"  # the deploy UI's Orbbec stream
STREAM_TIMEOUT_S = 3
STREAM_READ_BYTES = 400000
TAG_OF_ARM = (("left", "400"), ("right", "401"))
MM_PER_M = 1e3
STAMP_FMT = "%Y-%m-%d %H:%M:%S"
CLOCK_FMT = "%H:%M:%S"


def _readBridge():
    from laundry_shared.visual_teleop import config as vcfg
    from laundry_shared.visual_teleop.robot.bridge import HttpBridge

    return HttpBridge(vcfg.load().robot["base_url"], live=False, timeout_s=BRIDGE_TIMEOUT_S)


def _touch(bridge, arms: dict, side: str):
    """Median of TOUCH_READS reads of one arm -> (q, TCP position, spread deg)."""
    tQs = []
    for _ in range(TOUCH_READS):
        tQs.append(np.asarray(bridge.observe().q[side], np.float64))
        time.sleep(TOUCH_READ_S)
    tQ = np.median(tQs, 0)
    tP = fk(arms, side, tQ)[:3, 3]
    return tQ, tP, round(float(np.degrees(np.ptp(tQs, 0)).max()), 2)


def _readPending() -> list:
    return json.loads(heightCalib.PENDING_FILE.read_text()) if heightCalib.PENDING_FILE.exists() else []


def _writeCalibration(points: list) -> dict:
    if heightCalib.CALIB_FILE.exists():
        shutil.copy(heightCalib.CALIB_FILE, heightCalib.CALIB_FILE.with_suffix(".json.prev"))
    tCal = dict(
        created=time.strftime(STAMP_FMT),
        anchors=str(heightCalib.ANCHORS_FILE),
        table_z=heightCalib.TAG_TABLE_Z_M,
        points=points,
        fit=heightCalib.fitCalibration(points),
    )
    heightCalib.CALIB_FILE.write_text(json.dumps(tCal, indent=1))
    return tCal


def cameraXyShift() -> None:
    """[11-01] user "UI 에서 TCP 가 벗어나": the Orbbec (robot_cam_v005 = the B frame of the labels) sees the tag centres; their pixel ray
    on the table (z = 0.001) is the true B position, the pivot touches (pending p400 / p401) give the robot-FK position of the same
    point -> per-arm xy shift B - robot, stored into height_calib.json (plane untouched). Was camxy()."""
    import urllib.request

    import cv2

    from laundry_common import paths as egopaths

    tCalibDir = egopaths.DATA_CALIB
    tK = json.loads((tCalibDir / "robot_cam_v001.json").read_text())["intrinsics"]
    tKm = np.array([[tK["fx"], 0, tK["cx"]], [0, tK["fy"], tK["cy"]], [0, 0, 1]])
    tBaseFromCam = np.array(json.loads((tCalibDir / "robot_cam_v005.json").read_text())["T_B_C"])
    tBytes = urllib.request.urlopen(STREAM_URL, timeout=STREAM_TIMEOUT_S).read(STREAM_READ_BYTES)
    tI = tBytes.find(b"\xff\xd8", len(tBytes) // 2)
    tJ = tBytes.find(b"\xff\xd9", tI)
    tImg = cv2.imdecode(np.frombuffer(tBytes[tI : tJ + 2], np.uint8), 0)
    tCorners, tIds, _ = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)).detectMarkers(
        tImg
    )
    tPoints = _readPending()
    tCal = heightCalib.loadCalibration()
    for tCorner, tTagId in zip(tCorners if tIds is not None else [], tIds.ravel() if tIds is not None else []):
        tTouches = [tP for tP in tPoints if tP["target"] == f"p{tTagId}"]
        if not tTouches:
            continue
        tRay = np.linalg.inv(tKm) @ np.r_[tCorner[0].mean(0), 1]
        tOrigin = tBaseFromCam[:3, 3]
        tDir = tBaseFromCam[:3, :3] @ tRay
        tPb = tOrigin + (heightCalib.TAG_TABLE_Z_M - tOrigin[2]) / tDir[2] * tDir
        tPr = np.mean([tP["p_robot"] for tP in tTouches], 0)
        tSide = tTouches[0]["arm"]
        tShift = (tPb - tPr)[:2]
        tCal["fit"][tSide]["xy_cam_minus_robot"] = tShift.tolist()
        tCal["fit"][tSide]["xy_cam_from"] = f"tag {tTagId} camera ray vs {len(tTouches)} pivots"
        print(
            f"{tSide}: tag {tTagId} B {np.round(tPb * MM_PER_M, 1)} robot {np.round(tPr * MM_PER_M, 1)} -> xy shift {np.round(tShift * MM_PER_M, 1)} mm, "
            f"z check {(tPb - tPr)[2] * MM_PER_M:+.1f} vs plane {-heightCalib.dzAt(tCal, tSide, tPr[:2]) * MM_PER_M:+.1f} mm"
        )
    heightCalib.CALIB_FILE.write_text(json.dumps(tCal, indent=1))


def anchorXyShift() -> None:
    """[11-01] user "xy 도 맞추어줘": camera-free xy shift for the robot COMMANDS = anchors-file tag centre (B, 10-26; the re-printed tags
    were stuck over the old ones) - robot FK of the touches on that tag (pivots p400/p401 pending + tag touches 400/401) ->
    xy_B_minus_robot. Was anchorxy()."""
    tAnchors = json.loads(heightCalib.ANCHORS_FILE.read_text())["tags"]
    tCal = heightCalib.loadCalibration()
    tPoints = _readPending() + tCal["points"]
    for tSide, tTagId in TAG_OF_ARM:
        tP = np.array([tX["p_robot"] for tX in tPoints if tX["arm"] == tSide and tX["target"] in (tTagId, "p" + tTagId)])
        if not len(tP):
            continue
        tShift = np.asarray(tAnchors[tTagId]["p_B"])[:2] - tP[:, :2].mean(0)
        tCal["fit"][tSide]["xy_B_minus_robot"] = tShift.tolist()
        tCal["fit"][tSide]["xy_B_from"] = (
            f"anchors tag {tTagId} vs {len(tP)} touches (spread {np.round(tP[:, :2].std(0) * MM_PER_M, 1).tolist()} mm)"
        )
        print(
            f"{tSide}: tag {tTagId} anchors {np.round(np.asarray(tAnchors[tTagId]['p_B'])[:2] * MM_PER_M, 1)} robot {np.round(tP[:, :2].mean(0) * MM_PER_M, 1)} "
            f"(n {len(tP)}) -> xy shift {np.round(tShift * MM_PER_M, 1)} mm"
        )
    heightCalib.CALIB_FILE.write_text(json.dumps(tCal, indent=1))


def measure() -> None:
    """Interactive recording: 'l 400' / 'r 401' / 'l t', 'u' = undo the last touch, 'q' = save and end."""
    tAnchors = json.loads(heightCalib.ANCHORS_FILE.read_text())["tags"]
    tArms, _ = makeArms()
    tBridge = _readBridge()
    tStatus = tBridge.status()
    tParked = tStatus.get("parked") or []
    print(f"브리지: {tStatus.get('status')}  park(토크 꺼짐): {tParked}", flush=True)
    if set(tParked) != set(SIDES):
        print("주의: 두 팔 다 park(토크 꺼짐) 상태가 아님 -- UI 에서 '휴식(park)' 먼저", flush=True)
    print(
        "측정 방법: 집게를 닫은 채로 집게 끝(두 손가락 끝 가운데)을 대상에 살짝 대고 Enter.\n"
        "  대상: 400 / 401 = 태그 중심 (70 mm 태그), t = 테이블 아무 곳.  팔: l / r.  예) 'l 400', 'r t', 'u' = 마지막 취소, 'q' = 저장 후 끝",
        flush=True,
    )
    tPoints = []
    while True:
        try:
            tCmd = input("> ").strip().lower().split()
        except EOFError:
            tCmd = ["q"]
        if not tCmd:
            continue
        if tCmd[0] == "q":
            break
        if tCmd[0] == "u":
            if tPoints:
                print("취소:", tPoints.pop()["target"], flush=True)
            continue
        if tCmd[0] not in ARM_KEYS or len(tCmd) < 2 or tCmd[1] not in TAG_TARGETS:
            print("형식: 'l 400' / 'r 401' / 'l t'", flush=True)
            continue
        tSide = ARM_KEYS[tCmd[0]]
        tTarget = heightCalib.TABLE_TARGET if tCmd[1] == "t" else tCmd[1]
        tQ, tP, tSpread = _touch(tBridge, tArms, tSide)
        tExpected = (
            list(tAnchors[tTarget]["p_B"])
            if tTarget != heightCalib.TABLE_TARGET
            else [float(tP[0]), float(tP[1]), heightCalib.TAG_TABLE_Z_M]
        )
        tPoints.append(
            dict(
                arm=tSide,
                target=tTarget,
                q=tQ.tolist(),
                p_robot=tP.tolist(),
                p_expected=tExpected,
                q_spread_deg=tSpread,
                t=time.strftime(CLOCK_FMT),
            )
        )
        tD = (tP - np.asarray(tExpected)) * MM_PER_M
        print(
            f"  {tSide} {tTarget}: TCP ({tP[0] * 1e3:.0f}, {tP[1] * 1e3:.0f}, {tP[2] * 1e3:.1f}) mm | 데이터 기준과 차이 "
            + (f"dx {tD[0]:+.1f} dy {tD[1]:+.1f} " if tTarget != heightCalib.TABLE_TARGET else "")
            + f"dz {tD[2]:+.1f} mm"
            + (f"  (흔들림 {tPoints[-1]['q_spread_deg']}°)" if tPoints[-1]["q_spread_deg"] > WOBBLE_WARN_DEG else ""),
            flush=True,
        )
    if not tPoints:
        print("측정 없음 -- 저장 안 함")
        return
    show(_writeCalibration(tPoints))


def point(arm: str, target: str) -> None:
    """One touch, non-interactive (the operator touches, the caller records): appends to height_calib_pending.json."""
    tAnchors = json.loads(heightCalib.ANCHORS_FILE.read_text())["tags"]
    tArms, _ = makeArms()
    tBridge = _readBridge()
    tSide = ARM_KEYS[arm]
    tQ, tP, tSpread = _touch(tBridge, tArms, tSide)
    tTarget = heightCalib.TABLE_TARGET if target == "t" else target
    # 11-01 pivot: "p400" = jaw tip held on the tag-400 centre while the wrist is turned (TCP / joint-offset fit, kin_calib.py)
    tExpected = (
        list(tAnchors[tTarget.lstrip("p")]["p_B"])
        if tTarget != heightCalib.TABLE_TARGET
        else [float(tP[0]), float(tP[1]), heightCalib.TAG_TABLE_Z_M]
    )
    tPoints = _readPending()
    tPoints.append(
        dict(
            arm=tSide,
            target=tTarget,
            q=tQ.tolist(),
            p_robot=tP.tolist(),
            p_expected=tExpected,
            q_spread_deg=tSpread,
            t=time.strftime(CLOCK_FMT),
        )
    )
    heightCalib.PENDING_FILE.write_text(json.dumps(tPoints, indent=1))
    tD = (tP - np.asarray(tExpected)) * MM_PER_M
    print(
        f"#{len(tPoints)} {tSide} {tTarget}: TCP ({tP[0] * 1e3:.0f}, {tP[1] * 1e3:.0f}, {tP[2] * 1e3:.1f}) mm | "
        + (f"dx {tD[0]:+.1f} dy {tD[1]:+.1f} " if tTarget != heightCalib.TABLE_TARGET else "")
        + f"dz {tD[2]:+.1f} mm | 흔들림 {tPoints[-1]['q_spread_deg']}°"
    )


def finish() -> None:
    tPoints = _readPending()
    if not tPoints:
        print("기록된 점 없음")
        return
    tCal = _writeCalibration(tPoints)
    heightCalib.PENDING_FILE.rename(heightCalib.PENDING_FILE.with_suffix(f".done_{time.strftime('%H%M%S')}.json"))
    show(tCal)


def show(cal: dict | None = None) -> None:
    cal = cal or heightCalib.loadCalibration()
    if not cal:
        print("height_calib.json 없음 -- 먼저 measure")
        return
    print(f"높이 캘리브레이션 {cal['created']}  ({len(cal['points'])} 점)")
    for tSide, tFit in cal["fit"].items():
        tTags = "  ".join(f"태그 {tKey}: d {tV['d_mm']} mm (n {tV['n']})" for tKey, tV in tFit["tags"].items())
        print(
            f"  {tSide}: 테이블 높이 차 평균 {tFit['dz_mean_mm']:+.1f} mm, 기울기 {tFit['tilt_deg']}°, 잔차 최대 {tFit['resid_mm']} mm, n {tFit['n']}  {tTags}"
        )
    print("  (+ = 실제 테이블이 학습 좌표보다 높음 -> 배포 시 그만큼 올림)")


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser()
    tParser.add_argument("cmd", choices=["measure", "show", "point", "finish", "camxy", "anchorxy"])
    tParser.add_argument("args", nargs="*")
    return tParser


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    if tArgs.cmd == "measure":
        measure()
    elif tArgs.cmd == "point":
        point(*tArgs.args)  # point l 400 | point r 401 | point l t
    elif tArgs.cmd == "finish":
        finish()
    elif tArgs.cmd == "camxy":
        cameraXyShift()
    elif tArgs.cmd == "anchorxy":
        anchorXyShift()
    else:
        show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
