#!/usr/bin/env python3
"""[2026-10-28] Camera-only viewer (no robot): Orbbec from orbbec_server (:8092/frame) + every AVFoundation camera the Mac sees,
one page at http://localhost:8780 (Korean labels).  For checking the wrist cameras remotely before a deploy session.
    .venv-mac/bin/python deploy/cam_viewer.py [--port 8780] [--max-index 6]"""
import argparse, threading, time

import cv2
import numpy as np
import requests


class Cam(threading.Thread):
    def __init__(self, idx):
        super().__init__(daemon=True); self.idx = idx; self.jpg = None; self.wh = None; self.n = 0
        self.cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION); self.ok = self.cap.isOpened()
        if self.ok: self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    def run(self):
        while self.ok:
            r, f = self.cap.read()
            if not r: time.sleep(0.05); continue
            self.wh = (f.shape[1], f.shape[0]); self.n += 1
            v = cv2.resize(f, (640, int(640 * f.shape[0] / f.shape[1])), interpolation=cv2.INTER_AREA)
            cv2.putText(v, f"index {self.idx}  {self.wh[0]}x{self.wh[1]}", (8, 26), 0, 0.7, (0, 255, 255), 2)
            self.jpg = cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8780); ap.add_argument("--max-index", type=int, default=6)
    ap.add_argument("--orbbec", default="http://127.0.0.1:8092/frame"); a = ap.parse_args()
    from flask import Flask, Response
    cams = {}
    for i in range(a.max_index):
        c = Cam(i)
        if c.ok: c.start(); cams[i] = c
    app = Flask("cam_viewer"); s = requests.Session()
    import logging; logging.getLogger("werkzeug").setLevel(logging.ERROR)

    def orbbec_jpg():
        try:
            g = cv2.imdecode(np.frombuffer(s.get(a.orbbec, timeout=1).content, np.uint8), cv2.IMREAD_COLOR)
            g = cv2.resize(g, (640, int(640 * g.shape[0] / g.shape[1]))); cv2.putText(g, "Orbbec", (8, 26), 0, 0.7, (0, 255, 255), 2)
            return cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
        except Exception: return None

    def mjpeg(fn):
        def gen():
            while True:
                j = fn()
                if j: yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n"
                time.sleep(0.12)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")

    app.add_url_rule("/s/orbbec", "o", lambda: mjpeg(orbbec_jpg))
    for i in cams: app.add_url_rule(f"/s/{i}", f"c{i}", (lambda c: lambda: mjpeg(lambda: c.jpg))(cams[i]))
    page = ('<!doctype html><meta charset="utf-8"><title>카메라 확인</title><body style="background:#15171a;color:#ddd;font:14px sans-serif">'
            '<h3>카메라 확인 (로봇 명령 없음)</h3><div style="display:flex;flex-wrap:wrap;gap:8px">'
            '<div>Orbbec<br><img src="/s/orbbec" width="640"></div>' +
            "".join(f'<div>AVFoundation index {i}<br><img src="/s/{i}" width="640"></div>' for i in cams) + "</div></body>")
    app.add_url_rule("/", "i", lambda: page)
    print(f"[cam_viewer] http://localhost:{a.port}  cameras {list(cams)}", flush=True)
    app.run(host="127.0.0.1", port=a.port, threaded=True)


if __name__ == "__main__":
    main()
