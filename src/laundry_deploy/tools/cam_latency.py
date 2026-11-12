"""[2026-11-06] Camera capture delay, measured the UMI way (Chi et al. 2024).  Default --mode blink: this Mac's screen goes
full-screen black / white at random 120-500 ms intervals and the delay is the cross-correlation peak (tools.latency_probe.estimateLag)
of the drawn levels against the brightness of the pixels that flicker in the camera image (the screen, found by per-pixel variance)
- any size / angle works, the head Orbbec sees the laptop from ~1.2 m (a QR there was ~40 px, never decoded).
--mode qr: a window on this Mac shows a QR code of the host
monotonic clock (redrawn every display frame), the camera films the screen, and delay = the frame's deploy timestamp - the clock
in the decoded QR - the display delay (--display-ms, about one 60 Hz refresh).  The timestamps are exactly the deploy ones: the
wrist cameras are read and stamped like session.cameras.WristCam (monotonic_ns right after read()), the head Orbbec through
GlobalCam (orbbec_server's X-Capture-Monotonic-Ns).  Result: cam_ms.<cam> in LATENCY_FILE (run_loop --latency-match uses
cam_ms.global for the capture-time state).

Stop the UI first (it holds the cameras), hold the screen 20-40 cm in front of the camera, sharp and filling ~1/3 of the image.
    python -m laundry_deploy.tools.cam_latency --cam global|left_wrist|right_wrist [--secs 10]
    python -m laundry_deploy.tools.cam_latency --index N      (any UVC camera, e.g. a data-collection wrist cam; --name for the key)
"""

from __future__ import annotations
import argparse
import json
import sys
import threading
import time

import cv2
import numpy as np

from ..core.errors import DeployError, HardwareError
from ..session.paths import CAMS_FILE, LATENCY_FILE

QR_PX = 760  # fills a laptop screen: the head Orbbec sees the screen from ~1 m
WINDOW = "cam latency clock"
DEFAULT_DISPLAY_MS = 16.7  # one 60 Hz refresh between drawing the QR and the panel showing it
MIN_DECODED = 20
KEEP_W = 960
BLINK_W = 160  # analysis resolution of the blink mode
BLINK_PIXEL_PCT = 97.0
BLINK_MIN_STD = 20.0
BLINK_MIN_PIXELS = 15
BLINK_MIN_CORR = 0.6  # the slow black -> white ramp keeps it below ~0.85 even on a clean run
BLINK_MIN_S, BLINK_MAX_S = 0.12, 0.5
BLINK_WINDOW_PX = (3000, 2000)  # clipped to the screen
BLINK_SKIP_S = 1.5
EDGE_SEARCH_FRAMES = 20
MIN_EDGES = 10
MS_PER_S = 1e3
NS_PER_MS = 1e6
CAM_SIDE = {"left_wrist": "left", "right_wrist": "right"}


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser()
    tParser.add_argument("--cam", choices=["global", "left_wrist", "right_wrist"])
    tParser.add_argument("--mode", choices=["blink", "qr"], default="blink")
    tParser.add_argument("--index", type=int, help="any UVC camera index (stamped at read like the collectors)")
    tParser.add_argument("--name", default="", help="key for --index (default uvc<index>)")
    tParser.add_argument("--secs", type=float, default=10.0)
    tParser.add_argument("--display-ms", type=float, default=DEFAULT_DISPLAY_MS)
    tParser.add_argument("--period-ms", type=float, default=50.0, help="each code stays this long (0 = every redraw)")
    tParser.add_argument("--global-url", default="http://127.0.0.1:8092")
    return tParser


def qrImage(text: str) -> np.ndarray:
    tQr = cv2.QRCodeEncoder.create().encode(text)
    tImg = cv2.resize(tQr, (QR_PX, QR_PX), interpolation=cv2.INTER_NEAREST)
    tImg = cv2.copyMakeBorder(tImg, 40, 90, 40, 40, cv2.BORDER_CONSTANT, value=255)
    cv2.putText(tImg, text[-9:-6], (40, QR_PX + 110), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3)
    return tImg


def shrink(frame: np.ndarray) -> np.ndarray:
    tGrey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tW = tGrey.shape[1]
    return tGrey if tW <= KEEP_W else cv2.resize(tGrey, (KEEP_W, int(tGrey.shape[0] * KEEP_W / tW)), interpolation=cv2.INTER_AREA)


def blinkLag(draws: list, frames: list, displayMs: float) -> dict:
    """draws = [(t_s, level 0/1)], frames = [(t_ns, grey)] -> delay of the camera behind the drawn levels"""
    from .latency_probe import estimateLag

    tF = np.stack([cv2.resize(tG, (BLINK_W, int(tG.shape[0] * BLINK_W / tG.shape[1]))) for _, tG in frames]).astype(np.float32)
    tStd = tF.std(0)
    tMask = (tStd >= np.percentile(tStd, BLINK_PIXEL_PCT)) & (tStd > BLINK_MIN_STD)
    if tMask.sum() < BLINK_MIN_PIXELS:
        raise DeployError(f"깜빡이는 화면이 카메라에 안 보임 (픽셀 {int(tMask.sum())}) -- 화면이 보이게 다시")
    tY = tF[:, tMask].mean(1)
    np.savez(LATENCY_FILE.with_name("cam_latency_trace.npz"), t_draw=np.array([tD[0] for tD in draws]),
             level=np.array([tD[1] for tD in draws]), t_frame=np.array([tT / 1e9 for tT, _ in frames]), y=tY, mask=tMask)
    tTd = np.array([tD[0] for tD in draws])
    tLv = np.array([float(tD[1]) for tD in draws])
    tTf = np.array([tT / 1e9 for tT, _ in frames])
    tLag, tCorr = estimateLag(tTd, tLv, tTf, tY)
    # the delay itself from the white -> black edges only: a MacBook mini-LED panel ramps black -> white over ~100 ms (backlight),
    # which biased the cross-correlation; first frame past the midpoint = delay + U(0, frame period) -> minus half a period
    tMid = 0.5 * (np.percentile(tY, 5) + np.percentile(tY, 95))
    tPer = float(np.median(np.diff(tTf)))
    tFall = []
    for tT, tL in zip(tTd, tLv):
        if tL != 0:
            continue
        tK = np.searchsorted(tTf, tT)
        tHit = [tI for tI in range(tK, min(tK + EDGE_SEARCH_FRAMES, len(tTf))) if tY[tI] < tMid]
        if tHit:
            tFall.append(tTf[tHit[0]] - tT)
    if len(tFall) < MIN_EDGES:
        raise DeployError(f"검은색 전환을 {len(tFall)} 번만 봄 (< {MIN_EDGES}) -- 더 길게/화면이 보이게")
    tFall = np.asarray(tFall)
    return dict(median_ms=round((float(tFall.mean()) - 0.5 * tPer) * MS_PER_S - displayMs, 1), corr=round(tCorr, 3),
                xcorr_ms=round(tLag * MS_PER_S - displayMs, 1), fall_first_frame_ms=round(float(tFall.mean()) * MS_PER_S, 1),
                fall_p10_p90_ms=[round(float(np.percentile(tFall, q)) * MS_PER_S, 1) for q in (10, 90)], edges=len(tFall),
                frame_period_ms=round(tPer * MS_PER_S, 1), pixels=int(tMask.sum()), n=len(frames), display_ms=displayMs, mode="blink",
                created=time.strftime("%Y-%m-%d %H:%M:%S"))


class UvcGrabber(threading.Thread):
    """read() + monotonic_ns() right after it, every frame kept (same stamping as WristCam / the collectors)"""

    def __init__(self, index: int) -> None:
        super().__init__(daemon=True)
        from ..session.cameras import _openWrist

        self.mCap = _openWrist(index)
        if not self.mCap.isOpened():
            raise HardwareError(f"카메라 index {index} 를 열 수 없음 (UI 가 잡고 있으면 먼저 종료)")
        self.mFrames, self.mStop = [], False

    def run(self) -> None:
        while not self.mStop:
            tOk, tFrame = self.mCap.read()
            tT = time.monotonic_ns()
            if tOk and tFrame is not None:  # grey, <= KEEP_W wide: 30 fps of 1080p colour would hold ~2 GB for 10 s
                self.mFrames.append((tT, shrink(tFrame)))
        self.mCap.release()


class GlobalGrabber(threading.Thread):
    """the deploy GlobalCam (orbbec_server capture stamps), every new JPEG kept"""

    def __init__(self, url: str) -> None:
        super().__init__(daemon=True)
        from ..session.cameras import GlobalCam

        self.mCam = GlobalCam(url)
        self.mFrames, self.mStop = [], False

    def run(self) -> None:
        self.mCam.start()
        tLast = None
        while not self.mStop:
            tT, tValue = self.mCam.mRing.latest()
            if tValue is not None and tT != tLast:
                tLast = tT
                self.mFrames.append((tT, shrink(cv2.imdecode(np.frombuffer(tValue[1], np.uint8), cv2.IMREAD_COLOR))))
            time.sleep(0.002)
        self.mCam.mStop = True


def camLatency(args) -> dict:
    if args.cam == "global":
        tGrab, tKey = GlobalGrabber(args.global_url), "global"
    elif args.cam:
        tIndex = json.loads(CAMS_FILE.read_text())[CAM_SIDE[args.cam]]["index"]
        tGrab, tKey = UvcGrabber(tIndex), args.cam
    elif args.index is not None:
        tGrab, tKey = UvcGrabber(args.index), args.name or f"uvc{args.index}"
    else:
        raise DeployError("--cam 또는 --index")
    if args.mode == "blink":
        print(f"{tKey}: 화면 전체가 깜빡입니다 -- 카메라에 화면이 보이게 ({args.secs:.0f}초)")
        # a plain window as large as the screen allows: macOS full screen animates into its own Space for ~1 s and the first try
        # caught a half-black, half-white panel (corr 0.56)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, *BLINK_WINDOW_PX)
        cv2.moveWindow(WINDOW, 0, 0)
        tLevels = [np.zeros((64, 64), np.uint8), np.full((64, 64), 255, np.uint8)]
        tRng = np.random.default_rng(0)
        tGrab.start()
        tT0 = time.monotonic()
        tNext, tLevel, tDraws = 0.0, 0, []
        while time.monotonic() - tT0 < args.secs:
            if time.monotonic() >= tNext:
                tLevel = 1 - tLevel
                cv2.imshow(WINDOW, tLevels[tLevel])
                cv2.waitKey(1)
                tDraws.append((time.monotonic(), tLevel))  # after imshow + one event pump: the window has the new level
                tNext = time.monotonic() + tRng.uniform(BLINK_MIN_S, BLINK_MAX_S)
            cv2.waitKey(1)
        tGrab.mStop = True
        tGrab.join(timeout=3.0)
        cv2.destroyWindow(WINDOW)
        cv2.waitKey(1)
        if tGrab.mFrames:  # what the camera saw (debug)
            cv2.imwrite(str(LATENCY_FILE.with_name(f"cam_latency_{tKey}.jpg")), tGrab.mFrames[len(tGrab.mFrames) // 2][1])
        tSkipNs = int((tT0 + BLINK_SKIP_S) * 1e9)  # window placement / camera exposure settling
        tOut = blinkLag([tD for tD in tDraws if tD[0] >= tT0 + BLINK_SKIP_S], [tF for tF in tGrab.mFrames if tF[0] >= tSkipNs],
                        args.display_ms)
        print(json.dumps({tKey: tOut}, indent=1))
        if tOut["corr"] < BLINK_MIN_CORR:
            raise DeployError(f"상관 {tOut['corr']} < {BLINK_MIN_CORR} -- 저장 안 함 (화면이 더 크게/정면으로)")
    else:
        print(f"{tKey}: 화면의 QR 을 카메라 앞 20-40 cm 에 비추세요 ({args.secs:.0f}초)")
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        tGrab.start()
        tT0 = time.monotonic()
        tNext, tImg, tDrawn = 0.0, None, 0
        while time.monotonic() - tT0 < args.secs:
            if time.monotonic() >= tNext:  # --period-ms > 0: hold each code that long so an exposure never blends two codes
                tImg = qrImage(str(time.monotonic_ns()))
                tNext = time.monotonic() + args.period_ms / MS_PER_S
                tDrawn += 1
            cv2.imshow(WINDOW, tImg)
            cv2.waitKey(1)
        tGrab.mStop = True
        tGrab.join(timeout=3.0)
        cv2.destroyWindow(WINDOW)
        tDet = cv2.QRCodeDetector()
        tLags = []
        for tT, tFrame in tGrab.mFrames:
            tText, _, _ = tDet.detectAndDecode(tFrame)
            if tText.isdigit():
                tLags.append((tT - int(tText)) / NS_PER_MS - args.display_ms - 0.5 * args.period_ms)  # code shown for period_ms
        print(f"프레임 {len(tGrab.mFrames)}, QR 읽음 {len(tLags)}")
        if tGrab.mFrames:  # what the camera saw (debug)
            cv2.imwrite(str(LATENCY_FILE.with_name(f"cam_latency_{tKey}.jpg")), tGrab.mFrames[len(tGrab.mFrames) // 2][1])
        if len(tLags) < MIN_DECODED:
            raise DeployError(f"QR 을 {len(tLags)} 장만 읽음 (< {MIN_DECODED}) -- 더 가깝게/선명하게 다시")
        tLags = np.asarray(tLags)
        # the QR redraws at the display rate while the camera exposes for a while: the lower envelope is the delay of a frame that saw a
        # freshly drawn code; the median carries + half a redraw
        tOut = dict(median_ms=round(float(np.median(tLags)), 1), p10_ms=round(float(np.percentile(tLags, 10)), 1),
                    p90_ms=round(float(np.percentile(tLags, 90)), 1), n=int(len(tLags)), display_ms=args.display_ms,
                    created=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(json.dumps({tKey: tOut}, indent=1))
    tAll = json.loads(LATENCY_FILE.read_text()) if LATENCY_FILE.exists() else {}
    tAll.setdefault("cam", {})[tKey] = tOut
    tAll.setdefault("cam_ms", {})[tKey] = tOut["median_ms"]
    LATENCY_FILE.write_text(json.dumps(tAll, indent=1, ensure_ascii=False))
    print("저장", LATENCY_FILE)
    return tOut


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    try:
        camLatency(tArgs)
    except DeployError as tExc:
        print(tExc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
