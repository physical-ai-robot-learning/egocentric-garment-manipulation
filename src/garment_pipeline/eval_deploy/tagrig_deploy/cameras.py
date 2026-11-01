"""Cameras: UVC device list and wrist-camera auto-mapping by device name, the recorder / UMI-port checks, the timestamped
ring buffer, the robot wrist Arducams (fixed exposure, white balance, stall watchdog), the Orbbec poller, and the `cams` /
`wave` sub-commands.
"""
from __future__ import annotations
import collections
import json
import os
import pathlib
import subprocess
import threading
import time

import cv2
import numpy as np

from garment_pipeline.eval_deploy import act_tagrig_core as core
from garment_pipeline.eval_deploy.tagrig_deploy.paths import CAMS_CFG, DESKTOP, say, UVC


# ---------------------------------------------------------------- cameras / robot state (ring buffers on the host monotonic clock)
def uvc_list():
    return subprocess.run([UVC, "-d"], capture_output=True, text=True).stdout


def auto_remap_cams(cfg, lst):
    """[10-29 user: "하드코딩되어 있으면 자동으로 바꿔줘"] the two wrist Arducams report different device NAMES (firmware strings, stay with
    the physical camera: left 'Arducam 1080P Low Light', right 'FisheyeCamLeft').  If a configured USB location is gone or the
    USB set changed, find each side's camera by its name, re-probe the OpenCV index, rewrite act_tagrig_cams.json; the wave/cover
    confirmation stays valid (same physical cameras), the old mapping goes to act_tagrig_cams.json.bak_auto."""
    rows = [l.split(None, 4) for l in lst.splitlines() if "0x0c45" in l]          # index vend:prod location uvc name
    by_name = {r[4].strip(): r[2] for r in rows if len(r) >= 5}
    moved = [s for s in core.SIDES if (cfg.get(s, {}).get("location") or "").lower() not in lst.lower()]
    if not moved and cfg.get("cam_key") == cam_key(): return cfg
    from visual_teleop.tools.stereo_check import index_for_location
    new = json.loads(json.dumps(cfg))
    for s in core.SIDES:
        nm = cfg.get(s, {}).get("name"); loc = by_name.get(nm)
        if loc is None: raise SystemExit(f"{s} 손목 카메라 '{nm}' 가 USB 목록에 없음 -- 연결 확인\n{lst}")
        new[s]["location"] = loc; new[s]["index"] = index_for_location(loc, candidates=[0, 1, 2, 3, 4, 5])
        say(f"손목 카메라 자동 매핑: {s} = '{nm}' -> USB {loc}, OpenCV index {new[s]['index']} (전: {cfg.get(s, {}).get('location')}, index {cfg.get(s, {}).get('index')})")
    CAMS_CFG.with_name(CAMS_CFG.name + ".bak_auto").write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
    new["cam_key"] = cam_key(); new["probed"] = time.strftime("%Y-%m-%d %H:%M:%S") + " (auto by name)"
    if "wave_test" in new: new["wave_test"]["left"], new["wave_test"]["right"] = new["left"]["location"], new["right"]["location"]
    CAMS_CFG.write_text(json.dumps(new, indent=1, ensure_ascii=False)); return new


def cam_key():
    return "\n".join(sorted(" ".join(l.split()[1:]) for l in uvc_list().splitlines() if "0x0c45" in l))


def recorder_running():
    r = subprocess.run(["pgrep", "-fl", "tag_rig_record|umi_jaw|handumi_collector|laundry_collector|wrist_rec|stereo_check"], capture_output=True, text=True).stdout.strip()
    return [l for l in r.splitlines() if str(os.getpid()) not in l.split()[0]
            and not any(f"umi_jaw {m}" in l for m in ("infer", "train"))]   # 10-26: offline jaw labelling opens no camera


def umi_location_warning(cfg):
    """10-23 user: the robot LEFT wrist camera sits on the USB port the left UMI camera used.  umi_jaw.LOCATIONS (UMI recorder) is NOT
    changed here -> warn loudly whenever a robot wrist location matches a UMI recorder prefix (the recorder would open the robot cam)"""
    try:
        from visual_teleop.tools.umi_jaw import LOCATIONS
    except Exception: return []
    out = []
    for s in core.SIDES:
        for h, pre in LOCATIONS.items():
            if cfg.get(s, {}).get("location", "").lower().startswith(pre.lower()):
                out.append(f"경고: 로봇 {s} 손목 카메라 USB {cfg[s]['location']} = UMI 녹화기 umi_jaw LOCATIONS['{h}'] = {pre} 와 같은 포트 -- "
                           f"이대로 UMI 녹화(tag_rig_record --wrist / umi_jaw record --hand {h})를 하면 로봇 카메라가 열림. UMI 녹화 전에 UMI 카메라로 바꿔 꽂으세요")
    for w in out: print(w, flush=True)
    return out


class Ring:
    def __init__(self, n=12): self.buf = collections.deque(maxlen=n); self.lock = threading.Lock()

    def put(self, t_ns, x):
        with self.lock: self.buf.append((t_ns, x))

    def latest(self):
        with self.lock: return self.buf[-1] if self.buf else (0, None)

    def near(self, t_ns, tol_s):
        """entry closest to t_ns if within tol, else None"""
        with self.lock: b = list(self.buf)
        if not b: return None
        t, x = min(b, key=lambda e: abs(e[0] - t_ns))
        return (t, x) if abs(t - t_ns) <= tol_s * 1e9 else None


class WristCam(threading.Thread):
    """robot wrist Arducam at a USB location: 1920x1080 @30, fixed exposure + IMAGE_CTRL (umi_jaw.fixed_exposure); every frame is
    pre-processed here (core.prep_wrist -> 384x216 RGB) so the planner never waits on resizing"""
    def __init__(self, side, loc, index, exposure=2, align=None, prep=True):
        super().__init__(daemon=True); self.side, self.loc, self.index, self.align, self.prep = side, loc, index, align, prep
        c = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION); c.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); c.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080); c.set(cv2.CAP_PROP_FPS, 30)
        if not c.isOpened(): raise SystemExit(f"{side} 손목 카메라 index {index} 를 열 수 없음")
        from visual_teleop.tools.umi_jaw import fixed_exposure
        self.expo = fixed_exposure(c, loc, exposure)
        if self.expo and self.expo.get("mean", 0) > 185:     # 10-29: the right camera washes out at the minimum exposure (mean ~226) -> brightness down
            import subprocess
            subprocess.run([UVC, "-L", loc, "-s", "brightness=-64"], capture_output=True)
            for _ in range(6): c.read()
            ok_, f_ = c.read(); say(f"[집게] {side} 너무 밝음 (평균 {self.expo['mean']:.0f}) -> brightness -64, 평균 {cv2.cvtColor(f_, cv2.COLOR_BGR2GRAY).mean():.0f}")
        self.c = c; self.raw = None; self.ring = Ring(); self.n = 0; self.stop = False; self.wb = None

    def white_balance(self, f):
        """[10-29] the right Arducam's colour drifts (yellow/green; UVC white-balance-temp has no visible effect on it) and its SHADOWS
        go orange while the table is grey -> per-channel levels: black point (p1) and mid-grey (pixels 90..235, mostly table) of every
        channel mapped onto common targets (min black, max grey), smoothed over ~2 s, applied by LUT"""
        if self.n % 10 == 0:          # knots per channel: black (p1), shadow (grey 25..80 pixels), mid-grey (90..235) -> neutral targets
            s_ = f[::8, ::8].reshape(-1, 3).astype(np.float32); g_ = s_.mean(1); m_ = s_[(g_ > 90) & (g_ < 235)]; d_ = s_[(g_ > 25) & (g_ < 80)]
            if len(m_) > 500:
                dk = d_.mean(0) if len(d_) > 200 else np.full(3, np.nan); new = np.r_[np.percentile(s_, 1, axis=0), dk, m_.mean(0)]
                self.wb = new if self.wb is None else np.where(np.isnan(self.wb), new, 0.85 * self.wb + 0.15 * np.where(np.isnan(new), self.wb, new))
        if self.wb is None: return f
        lo, dk, hi = self.wb[:3], self.wb[3:6], self.wb[6:]; x = np.arange(256, dtype=np.float32); lut = []
        for c in range(3):
            kx, ky = [lo[c], hi[c]], [lo.min(), hi.max()]
            if np.isfinite(dk).all() and lo[c] + 2 < dk[c] < hi[c] - 2: kx.insert(1, dk[c]); ky.insert(1, dk.mean())
            lut.append(np.clip(np.interp(x, [0.0] + kx + [255.0], [0.0] + ky + [255.0]), 0, 255).astype(np.uint8))
        return cv2.merge([cv2.LUT(ch_, lut[i]) for i, ch_ in enumerate(cv2.split(f))])

    def _reopen(self):
        """[10-29] the left Arducam stopped delivering frames 3x today (device still enumerated; a deploy run then starved on stale
        images and stood still) -> release and reopen the same camera index, re-apply the fixed exposure"""
        self.n_reopen = getattr(self, "n_reopen", 0) + 1
        try: self.c.release()
        except Exception: pass
        time.sleep(0.5)
        if self.loc.lower() not in uvc_list().lower():      # unplugged: opening a missing AVFoundation index segfaulted the UI (10-29)
            say(f"[손목] {self.side} 카메라가 USB 목록에 없음 ({self.loc}) -- 다시 꽂을 때까지 대기"); time.sleep(2.0); return
        try:
            idx = self.index                   # same OpenCV index (uvc-util's index is NOT the AVFoundation order); a replug -> UI restart remaps
            c = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION); c.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); c.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080); c.set(cv2.CAP_PROP_FPS, 30)
            if not c.isOpened(): raise RuntimeError(f"index {idx} not opened")
            from visual_teleop.tools.umi_jaw import fixed_exposure
            fixed_exposure(c, self.loc, 2); self.c = c; self.index = idx
            say(f"[손목] {self.side} 카메라 프레임 멈춤 -> 다시 엶 ({self.n_reopen}회, index {idx})")
        except Exception as e:
            say(f"[손목] {self.side} 카메라 다시 열기 실패 ({type(e).__name__}: {e}) -- 2초 뒤 재시도"); time.sleep(2.0)

    def run(self):
        self.t_last = time.monotonic()
        def watchdog():                       # a stalled AVFoundation read() can block forever: release the capture to unblock it
            while not self.stop:
                time.sleep(0.5)
                if time.monotonic() - self.t_last > 2.0 and not getattr(self, "_reopening", False):
                    self._stalled = True
                    try: self.c.release()
                    except Exception: pass
        threading.Thread(target=watchdog, daemon=True).start()
        while not self.stop:
            ok, f = self.c.read(); t = time.monotonic_ns()
            if ok and f is not None:
                f = self.white_balance(f); self.raw = f; self.n += 1; self.t_last = time.monotonic()
                if self.prep: self.ring.put(t, core.prep_wrist(f, align=self.align))
            elif getattr(self, "_stalled", False) or time.monotonic() - self.t_last > 2.0:
                self._reopening = True; self._reopen(); self._stalled = False; self.t_last = time.monotonic(); self._reopening = False
            else: time.sleep(0.01)
        self.c.release()

    def get(self):
        return self.raw, 0


class FakeWristCam(threading.Thread):
    """[10-28] DEPLOY_FAKE_WRIST=<side>: the wrist camera of <side> is missing (not enumerated) -> black 384x216 frames at 30 Hz on
    the same ring interface (user: try 3 policy steps without it).  Out of training distribution -- test only, flagged in the log."""
    def __init__(self, side):
        super().__init__(daemon=True); self.side = side; self.raw = np.zeros((1080, 1920, 3), np.uint8); self.ring = Ring(); self.n = 0; self.stop = False
        self.img = np.zeros((core.IMG_WH[1], core.IMG_WH[0], 3), np.uint8)

    def run(self):
        while not self.stop:
            self.ring.put(time.monotonic_ns(), self.img); self.n += 1; time.sleep(1 / 30)

    def get(self): return self.raw, 0


class GlobalCam(threading.Thread):
    """orbbec_server /frame (848x480 JPEG q95, capture time X-Capture-Monotonic-Ns) polled at ~30 Hz; or dir:<take> = recorded
    color_*.jpg replayed at 15 Hz (TEST ONLY: not the live scene)"""
    def __init__(self, src):
        super().__init__(daemon=True); self.src = src; self.ring = Ring(); self.jpg = None; self.n = 0; self.stop = False; self.last_idx = None
        if src.startswith("dir:"):
            d = pathlib.Path(src[4:]); self.files = sorted(d.glob("color_*.jpg")); self.k = 0
            if not self.files: raise SystemExit(f"{d} 에 color_*.jpg 없음")
        else:
            import requests
            self.s = requests.Session(); self.url = src.rstrip("/") + "/frame"
            try: r = self.s.get(self.url, timeout=1.0); ok = r.status_code == 200
            except Exception as e: ok = False; r = e
            if not ok: raise SystemExit(f"Orbbec 서버 {self.url} 응답 없음 ({r}) -- 먼저: sudo ~/ego_collector/.venv/bin/python ~/laundry_system/robot/orbbec_server.py")

    def run(self):
        while not self.stop:
            if self.src.startswith("dir:"):
                b = self.files[self.k % len(self.files)].read_bytes(); self.k += 1; t = time.monotonic_ns(); time.sleep(1 / 15)
            else:
                try:
                    r = self.s.get(self.url, timeout=0.5)
                    if r.status_code != 200 or r.headers.get("X-Frame-Index") == self.last_idx: time.sleep(0.01); continue
                    self.last_idx = r.headers.get("X-Frame-Index"); b = r.content; t = int(r.headers.get("X-Capture-Monotonic-Ns", time.monotonic_ns()))
                except Exception: time.sleep(0.05); continue
            f = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR); self.jpg = b; self.n += 1
            self.ring.put(t, (core.prep_global(f), b))


def cams_cmd(a):
    """find each robot wrist camera's OpenCV index from its USB location (brightness probe), save act_tagrig_cams.json, save a labelled
    pair to the Desktop; --cover: the user covers the robot LEFT wrist camera -> the darker location must be the configured left"""
    if recorder_running(): raise SystemExit(f"녹화 프로세스가 카메라를 쓰는 중: {recorder_running()}")
    cfg = json.loads(CAMS_CFG.read_text()); lst = uvc_list(); print(lst); umi_location_warning(cfg)
    from visual_teleop.tools.stereo_check import index_for_location
    cand = [int(x) for x in a.candidates.split(",")]
    for s in core.SIDES:
        loc = cfg[s]["location"]
        if loc.lower() not in lst.lower(): raise SystemExit(f"{s}: USB 위치 {loc} 에 카메라 없음 -- 연결 확인 (uvc-util 목록 위)")
        cfg[s]["index"] = index_for_location(loc, candidates=cand); print(f"{s} 손목 = {loc} -> OpenCV index {cfg[s]['index']}", flush=True)
    cfg["cam_key"] = cam_key(); cfg["probed"] = time.strftime("%Y-%m-%d %H:%M:%S"); CAMS_CFG.write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
    cams = {s: WristCam(s, cfg[s]["location"], cfg[s]["index"], prep=False) for s in core.SIDES}
    for c in cams.values(): c.start()
    time.sleep(1.5)
    if a.cover:
        base = {s: float(cams[s].get()[0].mean()) for s in core.SIDES}; say("로봇 왼팔 손목 카메라를 손으로 가리세요 (5초)"); time.sleep(5)
        cov = {s: float(cams[s].get()[0].mean()) for s in core.SIDES}; drop = {s: base[s] - cov[s] for s in core.SIDES}
        ok = drop["left"] > 30 and drop["left"] > 3 * max(drop["right"], 1)
        print(f"밝기 변화 {drop} -> {'왼쪽 맞음' if ok else '확인 실패/반대 -- act_tagrig_cams.json 의 left/right location 을 확인'}", flush=True)
        cfg["cover_test"] = dict(drop=drop, ok=bool(ok), t=time.strftime("%Y-%m-%d %H:%M:%S")); CAMS_CFG.write_text(json.dumps(cfg, indent=1, ensure_ascii=False))
    ims = []
    for s in core.SIDES:
        f = cv2.resize(cams[s].get()[0], (960, 540)); cv2.putText(f, f"ROBOT {s.upper()} wrist  {cfg[s]['location']} idx {cfg[s]['index']}", (10, 40), 0, 1.1, (0, 0, 255), 3); ims.append(f)
        cams[s].stop = True
    p = DESKTOP / f"act_robot_wrists_{time.strftime('%H%M%S')}.jpg"; cv2.imwrite(str(p), np.hstack(ims)); print("저장", p)


# ---------------------------------------------------------------- wave test (wrist camera left/right by USB location)
def wave_cmd(a):
    """both robot wrist streams side by side, labelled with their USB location, in a browser (MJPEG :PORT).  The user waves at the
    robot LEFT arm's camera, then the RIGHT one; --confirm left=<loc> right=<loc> pins the answer into act_tagrig_cams.json."""
    if a.confirm:
        cfg = json.loads(CAMS_CFG.read_text()); ans = dict(x.split("=") for x in a.confirm)
        locs = {l.split()[2].lower(): " ".join(l.split()[4:]) for l in uvc_list().splitlines() if "0x0c45" in l}
        for s in core.SIDES:
            loc = ans[s].lower()
            if loc not in locs: raise SystemExit(f"{s}={loc}: 지금 연결된 위치가 아님 {list(locs)}")
            if cfg.get(s, {}).get("location", "").lower() != loc: cfg[s] = dict(location=loc, name=locs[loc], index=None)
        if ans["left"].lower() == ans["right"].lower(): raise SystemExit("left 와 right 가 같은 위치")
        cfg["wave_test"] = dict(confirmed=True, left=ans["left"], right=ans["right"], by=a.by, t=time.strftime("%Y-%m-%d %H:%M:%S"))
        CAMS_CFG.write_text(json.dumps(cfg, indent=1, ensure_ascii=False)); print("고정:", cfg["wave_test"], "-- index 가 바뀌었으면 `cams` 를 다시"); return
    if recorder_running(): raise SystemExit(f"녹화 프로세스가 카메라를 쓰는 중: {recorder_running()}")
    cfg = json.loads(CAMS_CFG.read_text()); cams = []; umi_location_warning(cfg)
    for s in core.SIDES:
        if cfg[s].get("index") is None: raise SystemExit("index 없음 -- 먼저 `cams`")
        w = WristCam(s, cfg[s]["location"], cfg[s]["index"], a.exposure, prep=False); w.start(); cams.append(w)
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    def frame():
        ims = []
        for w in cams:
            f = w.raw
            f = np.zeros((540, 960, 3), np.uint8) if f is None else cv2.resize(f, (960, 540))
            cv2.rectangle(f, (0, 0), (960, 60), (0, 0, 0), -1); cv2.putText(f, f"USB {w.loc}  ({cfg[w.side].get('name', '')}, idx {w.index})", (10, 42), 0, 1.0, (0, 255, 255), 2)
            ims.append(f)
        v = np.hstack(ims); cv2.putText(v, "wave at the robot LEFT wrist camera, then the RIGHT one -> tell which USB location moved", (10, 530), 0, 0.9, (0, 0, 255), 2)
        return cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/stream"):
                self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f"); self.end_headers()
                try:
                    while True: self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + frame() + b"\r\n"); time.sleep(0.07)
                except (BrokenPipeError, ConnectionResetError): return
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers()
            self.wfile.write(b"<html><body style='margin:0;background:#111'><img src='/stream' style='width:100%'></body></html>")
        def log_message(self, *x): pass
    threading.Thread(target=ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever, daemon=True).start()
    url = f"http://localhost:{a.port}/"; print(f"wave 테스트 화면: {url}  (Ctrl+C = 끝)", flush=True)
    subprocess.Popen(["open", url]); t0 = time.monotonic()
    try:
        while a.secs <= 0 or time.monotonic() - t0 < a.secs: time.sleep(0.5)
    except KeyboardInterrupt: pass
    p = DESKTOP / f"act_wave_test_{time.strftime('%H%M%S')}.jpg"; p.write_bytes(frame()); print("마지막 화면", p)
    for w in cams: w.stop = True
