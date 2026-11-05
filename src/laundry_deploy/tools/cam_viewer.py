"""[2026-10-28] Camera-only viewer (no robot): the Orbbec from orbbec_server (:8092/frame) + every AVFoundation camera the Mac
sees, one page at http://localhost:8780 (Korean labels). For checking the wrist cameras remotely before a deploy session.
Deploy stage: pre-session camera check.
    python -m laundry_deploy.tools.cam_viewer [--port 8780] [--max-index 6]
"""

from __future__ import annotations
import argparse
import sys
import threading
import time

import cv2
import numpy as np

CAPTURE_WH = (1920, 1080)  # requested capture size of every camera
VIEW_WIDTH_PX = 640  # tiles are resized to this width, keeping the aspect
JPEG_QUALITY = 70
READ_RETRY_S = 0.05
STREAM_PERIOD_S = 0.12  # MJPEG frame period of every tile
ORBBEC_TIMEOUT_S = 1.0
LABEL_ORG = (8, 26)
LABEL_SCALE = 0.7
LABEL_COLOUR = (0, 255, 255)


def _label(img, text: str) -> None:
    cv2.putText(img, text, LABEL_ORG, 0, LABEL_SCALE, LABEL_COLOUR, 2)


class Cam(threading.Thread):
    """One AVFoundation camera: latest frame as a labelled 640-wide JPEG (mJpg); mOk = it opened."""

    def __init__(self, idx: int) -> None:
        super().__init__(daemon=True)
        self.mIdx = idx
        self.mJpg = None
        self.mWh = None
        self.mFrames = 0
        self.mCap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        self.mOk = self.mCap.isOpened()
        if self.mOk:
            self.mCap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WH[0])
            self.mCap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_WH[1])

    def run(self) -> None:
        while self.mOk:
            tRead, tFrame = self.mCap.read()
            if not tRead:
                time.sleep(READ_RETRY_S)
                continue
            self.mWh = (tFrame.shape[1], tFrame.shape[0])
            self.mFrames += 1
            tViewWh = (VIEW_WIDTH_PX, int(VIEW_WIDTH_PX * tFrame.shape[0] / tFrame.shape[1]))
            tView = cv2.resize(tFrame, tViewWh, interpolation=cv2.INTER_AREA)
            _label(tView, f"index {self.mIdx}  {self.mWh[0]}x{self.mWh[1]}")
            self.mJpg = cv2.imencode(".jpg", tView, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()


def buildParser() -> argparse.ArgumentParser:
    tAp = argparse.ArgumentParser()
    tAp.add_argument("--port", type=int, default=8780)
    tAp.add_argument("--max-index", type=int, default=6)
    tAp.add_argument("--orbbec", default="http://127.0.0.1:8092/frame")
    return tAp


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    import requests
    from flask import Flask, Response

    tCams = {}
    for tI in range(tArgs.max_index):
        tCam = Cam(tI)
        if tCam.mOk:
            tCam.start()
            tCams[tI] = tCam
    tApp = Flask("cam_viewer")
    tSession = requests.Session()
    import logging

    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    def orbbecJpg():
        try:
            tContent = tSession.get(tArgs.orbbec, timeout=ORBBEC_TIMEOUT_S).content
            tG = cv2.imdecode(np.frombuffer(tContent, np.uint8), cv2.IMREAD_COLOR)
            tG = cv2.resize(tG, (VIEW_WIDTH_PX, int(VIEW_WIDTH_PX * tG.shape[0] / tG.shape[1])))
            _label(tG, "Orbbec")
            return cv2.imencode(".jpg", tG, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()
        except Exception:
            return None

    def mjpeg(fn):
        def gen():
            while True:
                tJpg = fn()
                if tJpg:
                    yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + tJpg + b"\r\n"
                time.sleep(STREAM_PERIOD_S)

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")

    tApp.add_url_rule("/s/orbbec", "o", lambda: mjpeg(orbbecJpg))
    for tI in tCams:
        tApp.add_url_rule(f"/s/{tI}", f"c{tI}", (lambda c: lambda: mjpeg(lambda: c.mJpg))(tCams[tI]))
    tPage = (
        '<!doctype html><meta charset="utf-8"><title>카메라 확인</title>'
        '<body style="background:#15171a;color:#ddd;font:14px sans-serif">'
        '<h3>카메라 확인 (로봇 명령 없음)</h3><div style="display:flex;flex-wrap:wrap;gap:8px">'
        '<div>Orbbec<br><img src="/s/orbbec" width="640"></div>'
        + "".join(f'<div>AVFoundation index {i}<br><img src="/s/{i}" width="640"></div>' for i in tCams)
        + "</div></body>"
    )
    tApp.add_url_rule("/", "i", lambda: tPage)
    print(f"[cam_viewer] http://localhost:{tArgs.port}  cameras {list(tCams)}", flush=True)
    tApp.run(host="127.0.0.1", port=tArgs.port, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
