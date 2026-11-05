"""The `cams` and `wave` sub-commands of the deploy CLI (setup before any run): find each robot wrist camera's OpenCV index from
its USB location, confirm left / right by covering or waving at a camera, and pin the result into act_tagrig_cams.json.
"""

from __future__ import annotations
import json
import subprocess
import threading
import time

import cv2
import numpy as np

from ..core.contract import SIDES
from ..core.errors import HardwareError
from .cameras import ARDUCAM_VENDOR, WristCam, camKey, recorderRunning, umiLocationWarning, uvcList
from .paths import CAMS_FILE, DESKTOP, say

WARMUP_S = 1.5
COVER_S = 5
COVER_MIN_DROP = 30  # grey-level drop of the covered (left) camera
COVER_RATIO = 3  # ... and at least this many times the other camera's drop
SNAPSHOT_WH = (960, 540)
LABEL_RED, LABEL_YELLOW = (0, 0, 255), (0, 255, 255)
WAVE_HOST = "127.0.0.1"
WAVE_FRAME_S = 0.07  # MJPEG frame period of the wave page
WAVE_JPEG_QUALITY = 70
WAVE_POLL_S = 0.5
STAMP_FMT = "%Y-%m-%d %H:%M:%S"
CAMS_CFG_INDENT = 1


def _writeCfg(cfg: dict) -> None:
    CAMS_FILE.write_text(json.dumps(cfg, indent=CAMS_CFG_INDENT, ensure_ascii=False))


def camsCmd(args) -> None:
    """Find each robot wrist camera's OpenCV index from its USB location (brightness probe), save act_tagrig_cams.json, save a
    labelled pair to the Desktop; --cover: the user covers the robot LEFT wrist camera -> the darker location must be the configured
    left."""
    if recorderRunning():
        raise HardwareError(f"녹화 프로세스가 카메라를 쓰는 중: {recorderRunning()}")
    tCfg = json.loads(CAMS_FILE.read_text())
    tList = uvcList()
    print(tList)
    umiLocationWarning(tCfg)
    from laundry_shared.visual_teleop.tools.stereo_check import index_for_location

    tCandidates = [int(tX) for tX in args.candidates.split(",")]
    for tSide in SIDES:
        tLoc = tCfg[tSide]["location"]
        if tLoc.lower() not in tList.lower():
            raise HardwareError(f"{tSide}: USB 위치 {tLoc} 에 카메라 없음 -- 연결 확인 (uvc-util 목록 위)")
        tCfg[tSide]["index"] = index_for_location(tLoc, candidates=tCandidates)
        print(f"{tSide} 손목 = {tLoc} -> OpenCV index {tCfg[tSide]['index']}", flush=True)
    tCfg["cam_key"] = camKey()
    tCfg["probed"] = time.strftime(STAMP_FMT)
    _writeCfg(tCfg)
    tCams = {tSide: WristCam(tSide, tCfg[tSide]["location"], tCfg[tSide]["index"], prep=False) for tSide in SIDES}
    for tCam in tCams.values():
        tCam.start()
    time.sleep(WARMUP_S)
    if args.cover:
        tBase = {tSide: float(tCams[tSide].get()[0].mean()) for tSide in SIDES}
        say("로봇 왼팔 손목 카메라를 손으로 가리세요 (5초)")
        time.sleep(COVER_S)
        tCovered = {tSide: float(tCams[tSide].get()[0].mean()) for tSide in SIDES}
        tDrop = {tSide: tBase[tSide] - tCovered[tSide] for tSide in SIDES}
        tOk = tDrop["left"] > COVER_MIN_DROP and tDrop["left"] > COVER_RATIO * max(tDrop["right"], 1)
        print(
            f"밝기 변화 {tDrop} -> {'왼쪽 맞음' if tOk else '확인 실패/반대 -- act_tagrig_cams.json 의 left/right location 을 확인'}",
            flush=True,
        )
        tCfg["cover_test"] = dict(drop=tDrop, ok=bool(tOk), t=time.strftime(STAMP_FMT))
        _writeCfg(tCfg)
    tImages = []
    for tSide in SIDES:
        tFrame = cv2.resize(tCams[tSide].get()[0], SNAPSHOT_WH)
        cv2.putText(
            tFrame,
            f"ROBOT {tSide.upper()} wrist  {tCfg[tSide]['location']} idx {tCfg[tSide]['index']}",
            (10, 40),
            0,
            1.1,
            LABEL_RED,
            3,
        )
        tImages.append(tFrame)
        tCams[tSide].mStop = True
    tPath = DESKTOP / f"act_robot_wrists_{time.strftime('%H%M%S')}.jpg"
    cv2.imwrite(str(tPath), np.hstack(tImages))
    print("저장", tPath)


def _waveFrame(cams: list, cfg: dict) -> bytes:
    tImages = []
    for tCam in cams:
        tFrame = tCam.mRaw
        tFrame = np.zeros((SNAPSHOT_WH[1], SNAPSHOT_WH[0], 3), np.uint8) if tFrame is None else cv2.resize(tFrame, SNAPSHOT_WH)
        cv2.rectangle(tFrame, (0, 0), (SNAPSHOT_WH[0], 60), (0, 0, 0), -1)
        cv2.putText(
            tFrame, f"USB {tCam.mLoc}  ({cfg[tCam.mSide].get('name', '')}, idx {tCam.mIndex})", (10, 42), 0, 1.0, LABEL_YELLOW, 2
        )
        tImages.append(tFrame)
    tView = np.hstack(tImages)
    cv2.putText(
        tView,
        "wave at the robot LEFT wrist camera, then the RIGHT one -> tell which USB location moved",
        (10, 530),
        0,
        0.9,
        LABEL_RED,
        2,
    )
    return cv2.imencode(".jpg", tView, [cv2.IMWRITE_JPEG_QUALITY, WAVE_JPEG_QUALITY])[1].tobytes()


def _confirmWave(args) -> None:
    tCfg = json.loads(CAMS_FILE.read_text())
    tAnswer = dict(tX.split("=") for tX in args.confirm)
    tLocs = {tLine.split()[2].lower(): " ".join(tLine.split()[4:]) for tLine in uvcList().splitlines() if ARDUCAM_VENDOR in tLine}
    for tSide in SIDES:
        tLoc = tAnswer[tSide].lower()
        if tLoc not in tLocs:
            raise HardwareError(f"{tSide}={tLoc}: 지금 연결된 위치가 아님 {list(tLocs)}")
        if tCfg.get(tSide, {}).get("location", "").lower() != tLoc:
            tCfg[tSide] = dict(location=tLoc, name=tLocs[tLoc], index=None)
    if tAnswer["left"].lower() == tAnswer["right"].lower():
        raise HardwareError("left 와 right 가 같은 위치")
    tCfg["wave_test"] = dict(confirmed=True, left=tAnswer["left"], right=tAnswer["right"], by=args.by, t=time.strftime(STAMP_FMT))
    _writeCfg(tCfg)
    print("고정:", tCfg["wave_test"], "-- index 가 바뀌었으면 `cams` 를 다시")


def waveCmd(args) -> None:
    """Both robot wrist streams side by side, labelled with their USB location, in a browser (MJPEG :PORT).  The user waves at the
    robot LEFT arm's camera, then the RIGHT one; --confirm left=<loc> right=<loc> pins the answer into act_tagrig_cams.json."""
    if args.confirm:
        _confirmWave(args)
        return
    if recorderRunning():
        raise HardwareError(f"녹화 프로세스가 카메라를 쓰는 중: {recorderRunning()}")
    tCfg = json.loads(CAMS_FILE.read_text())
    tCams = []
    umiLocationWarning(tCfg)
    for tSide in SIDES:
        if tCfg[tSide].get("index") is None:
            raise HardwareError("index 없음 -- 먼저 `cams`")
        tCam = WristCam(tSide, tCfg[tSide]["location"], tCfg[tSide]["index"], args.exposure, prep=False)
        tCam.start()
        tCams.append(tCam)
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class WaveHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                self.end_headers()
                try:
                    while True:
                        self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + _waveFrame(tCams, tCfg) + b"\r\n")
                        time.sleep(WAVE_FRAME_S)
                except (BrokenPipeError, ConnectionResetError):
                    return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<html><body style='margin:0;background:#111'><img src='/stream' style='width:100%'></body></html>")

        def log_message(self, *args):
            pass

    threading.Thread(target=ThreadingHTTPServer((WAVE_HOST, args.port), WaveHandler).serve_forever, daemon=True).start()
    tUrl = f"http://localhost:{args.port}/"
    print(f"wave 테스트 화면: {tUrl}  (Ctrl+C = 끝)", flush=True)
    subprocess.Popen(["open", tUrl])
    tT0 = time.monotonic()
    try:
        while args.secs <= 0 or time.monotonic() - tT0 < args.secs:
            time.sleep(WAVE_POLL_S)
    except KeyboardInterrupt:
        pass
    tPath = DESKTOP / f"act_wave_test_{time.strftime('%H%M%S')}.jpg"
    tPath.write_bytes(_waveFrame(tCams, tCfg))
    print("마지막 화면", tPath)
    for tCam in tCams:
        tCam.mStop = True
