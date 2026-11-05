"""UMI jaw OPEN / CLOSED from the SINGLE UMI wrist fisheye (2026-10-20, gripper-timing parity). The old jaw_resnet18_v004 used both
stereo tracking cameras of the UMI; those two cameras now sit on the ROBOT (indices 0/3), the UMI keeps one fisheye (FisheyeCamLeft,
USB location 0x11140000), so the classifier is re-trained on that view. Same recipe as jaw_finetune (ImageNet ResNet18, brightness /
contrast / gamma / shift / blur augmentation, first --skip-s of every cue dropped), input = the bottom ROI where the jaws are.
  capture  spoken cycles (open / closed, keep moving the UMI around) -> datasets/umi_jaw/cap_<ts>.npz (crops, label 1 = open)
  train    captures -> configs/calibration/umi_jaw_resnet18_v001.pt (TorchScript, 1x3x224x224 -> logit open), cycle-held-out validation
  record   wrist video at --hz with host monotonic_ns until Ctrl+C (run beside tag_rig_record) -> datasets/umi_wrist/<ts>/
  infer    record dir -> umi_jaw.npz (t_ns, p_open, state with hysteresis, events)
    .venv/bin/python -m visual_teleop.tools.umi_jaw capture --hand right|left [--cycles 6 --phase-s 8]
    .venv/bin/python -m visual_teleop.tools.umi_jaw train --hand right|left datasets/umi_jaw/cap_right_*.npz [--val 0:1 1:3]
    .venv/bin/python -m visual_teleop.tools.umi_jaw record [--index 2]          (Ctrl+C to stop)
    .venv/bin/python -m visual_teleop.tools.umi_jaw infer datasets/umi_wrist/<ts>"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path
import cv2
import numpy as np
from laundry_common import paths as egopaths

ROOT = Path(__file__).resolve().parents[4]
LOCATIONS = {"right": "0x0314", "left": "0x0114"}   # 11-03 re-cable (user-checked 11:58: first guess was swapped): right UMI cam 0x0314, left 0x0114
_LOCATIONS_1102 = {"right": "0x0311", "left": "0x0314"}   # 11-02: left UMI cam now on the robot-left-cam port (0x0111 before)  # 11-01 14:40 re-cable for laundry pinching x100: right cam now 0x03110000 (left 0x01110000 cover test 88 -> 21, right unchanged 146; left jaw motor 5B3D049747 moved -591..-118, right 5B3D048082 still)
_LOCATIONS_1030 = {"right": "0x03131", "left": "0x0111"}  # 10-30 10:25 re-cable after the robot deploy: left cam 0x01110000 shares hub 1 with the LEFT IMU 63ACF722, right cam 0x03131100 shares the 0x03131500 adapter with the RIGHT motor 5B3D048082 (10:31 cover test PASS: covering the LEFT hand lens dropped 0x0111 132 -> 61-83, 0x03131 unchanged 179-187)
_LOCATIONS_1029b = {"right": "0x03110", "left": "0x03140"}  # 10-29 15:57 physical test: the LEFT hand unit = camera 0x03140 + motor 5B3D049747 + IMU 63ACF722 (all three swapped)
_LOCATIONS_1029a = {"right": "0x03140", "left": "0x03110"}  # 10-29: left UMI camera now at 0x03110000 (robot wrist cams unplugged; 0x0311 was the robot FisheyeCamLeft port on 10-28 -- unplug the robot cams before UMI recording)
_LOCATIONS_1023 = {"right": "0x03140", "left": "0x01140"}  # 10-23: UMI cams re-plugged after the robot deploy test -> left back at 0x01140000 (0x0111 now = XIAO IMU)
_LOCATIONS_1022 = {"right": "0x03140", "left": "0x0111"}   # 10-22: hub swap moved the left UMI camera 0x01140000 -> 0x01110000
_LOCATIONS_BEFORE_1022 = {"right": "0x03140", "left": "0x01140"}   # 10-21 12:30 (0x03140 so it does not also match the robot cam that sat at 0x03141000 on 10-20): right UMI replugged 0x0113 -> 0x03140000 (user: index 2 = right UMI); 0x0113 now = robot wrist FisheyeCamLeft
_LOCATIONS_BEFORE_1230 = {"right": "0x0113", "left": "0x0311"}   # 10-21: USB location PREFIX (hub branch). The left camera re-enumerated as 0x03110000 / 0x03111000 between runs   # 10-21 12:45: left replugged 0x0311 -> 0x01140000 | 14:40: 0x03141000 = ROBOT right wrist cam (Arducam name too) -- never widen the right UMI prefix to 0x0314
LOCATION = LOCATIONS["right"]
_LOC_NOTE = "LOCATION before 10-20 17:40 was 0x03110000 on the RIGHT UMI (jaw model v001 + takes up to 20261020_160629 used it)"                     # 10-20 user: the UMI wrist fisheye = Arducam at 0x03110000 (FisheyeCamLeft 0x01114000 = other robot, 0x03141000 = robot)
ROI = (0.60, 1.00, 0.28, 0.78)              # y0, y1, x0, x1 as fractions of the frame: jaws bottom-centre
# 10-21: per hand. The left UMI's camera shows the jaws only in the bottom 30 % (x 0.25-0.6 when open); with ROI it was mostly background
ROIS = {"right": (0.60, 1.00, 0.20, 0.80), "left": (0.60, 1.00, 0.20, 0.80)}   # 10-22: both wrist cameras re-angled (user) so the whole jaws sit bottom-centre; v002 models were trained on the OLD views
_ROIS_BEFORE_1022 = {"right": ROI, "left": (0.70, 1.00, 0.24, 0.68)}
CROP_WH = (224, 112)
MODEL = ROOT / "configs/laundry/calibration/umi_jaw_resnet18_v001.pt"
# 10-21: per-hand models -- camera 0x0311 moved from the right to the left UMI and the jaws were adjusted, v001 no longer fits either hand
MODELS_V002 = {h: ROOT / f"configs/laundry/calibration/umi_jaw_resnet18_v002_{h}.pt" for h in ("right", "left")}
MODELS = {h: ROOT / f"configs/laundry/calibration/umi_jaw_resnet18_v003_{h}.pt" for h in ("right", "left")}   # 10-22: re-angled cameras; held-out capture accuracy L ~0.95 (bright table) / R 0.97
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None], np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


def say(t: str) -> None:
    print("[집게]", t, flush=True)
    try: subprocess.Popen(["say", "-v", "Yuna", t])
    except Exception: pass


def _cam_key() -> str:
    """10-20: the UVC cameras that matter (vendor 0x0c45 = the Arducam / fisheye wrist cameras), location + name, sorted. The Orbbec
    line of uvc-util changes when the sudo recorder opens it, which invalidated the whole-text cache and made two probes collide."""
    lst = subprocess.run([str(egopaths.UVC_UTIL), "-d"], capture_output=True, text=True).stdout
    return "\n".join(sorted(" ".join(l.split()[1:]) for l in lst.splitlines() if "0x0c45" in l))


def open_cam(index: int, w=1920, h=1080, location: str | None = None):
    location = location or LOCATION
    if index < 0:     # 10-21: the robot's wrist fisheye (FisheyeCamLeft) got plugged into the right UMI's hub branch 0x0113 while the UMI
        # camera was unplugged -> a whole jaw capture recorded the robot gripper. The UMI wrist cameras are the "Arducam 1080P Low Light" ones.
        lst = subprocess.run([str(egopaths.UVC_UTIL), "-d"], capture_output=True, text=True).stdout
        hit = [l for l in lst.splitlines() if "0x0c45" in l and l.split()[2].startswith(location)]
        if not any("Arducam" in l for l in hit):
            sys.exit(f"[집게] USB 위치 {location} 에 UMI 손목 카메라(Arducam)가 없음: {[' '.join(l.split()[2:]) for l in hit] or '아무것도 없음'} -- UMI 카메라 USB 를 확인하세요")
    cache = ROOT / f"datasets/umi_wrist/cam_index_{location}.json"
    if index < 0 and cache.exists():        # 10-20: the brightness probe takes ~30 s; reuse it while no camera was replugged
        try:
            j = json.loads(cache.read_text())
            if j.get("location") == location and j.get("cam_key") == _cam_key(): index = int(j["index"]); print(f"[집게] UMI 손목 카메라 {location} = index {index} (저장된 값)", flush=True)
        except Exception: pass
    if index < 0:
        from laundry_shared.visual_teleop.tools.stereo_check import index_for_location
        index = index_for_location(location); print(f"[집게] UMI 손목 카메라 {location} = index {index}", flush=True)
        try:
            cache.parent.mkdir(parents=True, exist_ok=True); cache.write_text(json.dumps(dict(location=location, index=index, cam_key=_cam_key(), t=time.time())))
        except Exception: pass
    c = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION); c.set(cv2.CAP_PROP_FRAME_WIDTH, w); c.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not c.isOpened(): sys.exit(f"카메라 index {index} 를 열 수 없음")
    t0 = time.monotonic()                     # 10-20: a replug renumbered the cameras and capture saved 0 frames -> fail fast
    while time.monotonic() - t0 < 5:
        ok, f = c.read()
        if ok and f is not None and f.std() > 3: return c
    c.release(); sys.exit(f"카메라 index {index} 에서 5초 동안 영상이 안 나옴 -- --index 없이 (USB 위치로 자동) 다시")


def crop(bgr: np.ndarray, roi=ROI) -> np.ndarray:
    h, w = bgr.shape[:2]; y0, y1, x0, x1 = roi
    g = cv2.cvtColor(bgr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)], cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, CROP_WH, interpolation=cv2.INTER_AREA)


def to_input(c: np.ndarray) -> np.ndarray:
    x = cv2.resize(c, (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
    return (np.repeat(x[None], 3, 0) - MEAN) / STD


def augment(c, rng):
    v = c.astype(np.float32); v = np.clip(v * rng.uniform(0.6, 1.4) + rng.uniform(-40, 40), 0, 255); v = 255 * (v / 255) ** rng.uniform(0.7, 1.4)
    dx, dy = rng.integers(-8, 9, 2); v = cv2.warpAffine(v, np.float32([[1, 0, dx], [0, 1, dy]]), (v.shape[1], v.shape[0]), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.3: v = cv2.GaussianBlur(v, (5, 5), rng.uniform(0.5, 1.5))
    return v.astype(np.uint8)


def capture(a):
    roi = ROIS[a.hand]; cam = open_cam(a.index, location=LOCATIONS[a.hand]); out = ROOT / "datasets/umi_jaw"; out.mkdir(parents=True, exist_ok=True)
    for _ in range(20): cam.read()
    ok, f = cam.read(); cv2.imwrite(str(out / f"roi_check_{a.hand}.jpg"), cv2.rectangle(f.copy(), (int(roi[2] * f.shape[1]), int(roi[0] * f.shape[0])), (int(roi[3] * f.shape[1]) - 1, int(roi[1] * f.shape[0]) - 1), (0, 0, 255), 4))
    say(f"{'오른손' if a.hand == 'right' else '왼손'} 집게 학습 데이터입니다. 안내에 따라 집게를 열고 닫으면서, 유미를 여기저기 천천히 움직이세요. 흰 티셔츠 위에서도, 허공에서도 하세요. 열 때는 반쯤 열기와 끝까지 열기를 섞고, 닫을 때는 절반은 티셔츠를 꽉 물고 닫으세요"); time.sleep(4)
    C, Y, P = [], [], []
    for cyc in range(a.cycles):
        for lab, txt in ((1, "집게 여세요"), (0, "집게 닫으세요")):
            say(f"{txt}, {cyc + 1}회"); t0 = time.monotonic(); nxt = t0
            while time.monotonic() - t0 < a.phase_s:
                ok, f = cam.read()
                if not ok or time.monotonic() < nxt: continue
                nxt += 1 / a.hz; C.append(crop(f, roi)); Y.append(lab); P.append(time.monotonic() - t0)
    say("끝났습니다"); cam.release(); p = out / f"cap_{a.hand}_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez_compressed(p, crops=np.array(C), label=np.array(Y), t_in_phase=np.array(P), cycles=a.cycles, phase_s=a.phase_s, roi=np.array(roi), hand=a.hand, location=LOCATIONS[a.hand])
    print(f"[집게] 저장 {p} ({len(C)} 프레임), ROI 확인 그림 {out / f'roi_check_{a.hand}.jpg'}", flush=True)


def train(a):
    import torch, torchvision
    a.out = a.out or str(MODELS[a.hand])
    for p in a.captures:
        zz = np.load(p); h = str(zz["hand"]) if "hand" in zz.files else ""
        if h and h != a.hand: sys.exit(f"[집게] {p} 는 {h} 손 캡처입니다 (--hand {a.hand})")
    X, Y, G = [], [], []
    for gi, p in enumerate(a.captures):
        z = np.load(p); per = 2; ph = np.cumsum(np.r_[0, np.diff(z["t_in_phase"]) < 0]); cyc = ph // per
        keep = z["t_in_phase"] >= a.skip_s
        X += list(z["crops"][keep]); Y += list(z["label"][keep]); G += list(gi * 100 + cyc[keep])
    Y, G = np.array(Y), np.array(G); pair = lambda s: int(s.split(":")[0]) * 100 + int(s.split(":")[1])
    val = np.isin(G, [pair(v) for v in a.val]) if a.val else np.isin(G % 100, [G.max() % 100])      # default: last cycle of every capture
    dev = "mps" if torch.backends.mps.is_available() else "cpu"; rng = np.random.default_rng(0); torch.manual_seed(0)
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1); net.fc = torch.nn.Linear(512, 1); net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4); tr = np.where(~val)[0]; lossf = torch.nn.BCEWithLogitsLoss()
    Xv = torch.tensor(np.stack([to_input(X[i]) for i in np.where(val)[0]])) if val.any() else None
    print(f"[집게] 학습 {len(tr)} / 검증 {int(val.sum())} 프레임 (열림 비율 {Y.mean():.2f}), {dev}", flush=True)
    for ep in range(a.epochs):
        net.train(); rng.shuffle(tr); tot = 0.0
        for b in range(0, len(tr), 32):
            idx = tr[b:b + 32]; x = torch.tensor(np.stack([to_input(augment(X[i], rng)) for i in idx])).to(dev); y = torch.tensor(Y[idx], dtype=torch.float32).to(dev)
            loss = lossf(net(x).squeeze(1), y); opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(idx)
        msg = f"[집게] epoch {ep + 1}: loss {tot / len(tr):.3f}"
        if Xv is not None:
            net.eval()
            with torch.no_grad(): pr = torch.cat([net(Xv[i:i + 64].to(dev)).squeeze(1).cpu() for i in range(0, len(Xv), 64)]).numpy() > 0
            msg += f" | 검증(학습에 안 쓴 회차) 정확도 {np.mean(pr == Y[val]):.3f}"
        print(msg, flush=True)
    net.eval().cpu(); torch.jit.trace(net, torch.zeros(1, 3, 224, 224)).save(a.out); print(f"[집게] 저장 {a.out}", flush=True)


IMAGE_CTRL = {"backlight-compensation": 0, "gamma": 72, "contrast": 48, "brightness": -20}   # 10-22, see fixed_exposure


def fixed_exposure(cam, location: str, value: int):
    """10-22: take 123614 locked exposure 63 at the start pose (a darker view there) -> the whole take was washed out (mean grey 211,
    contrast 39 vs 5-6 / 170 / 66 on the good takes). A fixed value measured on the table under the room lights avoids that."""
    U = str(egopaths.UVC_UTIL); lst = subprocess.run([U, "-d"], capture_output=True, text=True).stdout
    line = next((l for l in lst.splitlines() if location.lower() in l.lower()), None)
    if line is None: print(f"[집게] 노출 고정 실패: uvc-util 에 {location} 없음", flush=True); return None
    u = line.split()[0]
    # 11-01: after a replug the Arducam reported mode 1 / exposure 2 but streamed at mean ~229 (washed out); a pass through auto (8) and back
    # to manual made the fixed exposure take effect again (mean 122) -> always toggle before setting it
    subprocess.run([U, "-I", u, "-s", "auto-exposure-mode=8"], capture_output=True)
    for _ in range(10): cam.read()
    subprocess.run([U, "-I", u, "-s", "auto-exposure-mode=1"], capture_output=True)
    subprocess.run([U, "-I", u, "-s", f"exposure-time-abs={int(value)}"], capture_output=True)
    # 10-22 user "wrist view is hazy": over a white table even the minimum exposure leaves mean grey ~210. Backlight compensation off,
    # gamma 72 (min), contrast 48, brightness -20 raised whole-frame contrast 58-66 -> 85-94 (sweep on both wrist cameras)
    for k_, v_ in IMAGE_CTRL.items(): subprocess.run([U, "-I", u, "-s", f"{k_}={v_}"], capture_output=True)
    for _ in range(6): cam.read()
    ok, f = cam.read(); g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    print(f"[집게] 노출 고정 {location}: exposure-time-abs {int(value)} (고정값), 평균 {g.mean():.0f}, 상위1% {np.percentile(g, 99):.0f}", flush=True)
    return dict(exposure=int(value), p99=float(np.percentile(g, 99)), mean=float(g.mean()), mode="fixed", image_ctrl=IMAGE_CTRL)


def lock_exposure(cam, location: str, lo=225.0, hi=245.0):
    """10-20 (MASt3R lost 92 % of a white-T-shirt take: auto exposure washed the cloth out). Manual exposure, chosen on the live scene so
    the top 1 % of grey levels sits in [lo, hi] (no clipping, ~0.5-1 stop under auto). Bisection over exposure-time-abs (1..5000)."""
    U = str(egopaths.UVC_UTIL); lst = subprocess.run([U, "-d"], capture_output=True, text=True).stdout
    line = next((l for l in lst.splitlines() if location.lower() in l.lower()), None)
    if line is None: print(f"[집게] 노출 고정 실패: uvc-util 에 {location} 없음", flush=True); return None
    u = line.split()[0]; subprocess.run([U, "-I", u, "-s", "auto-exposure-mode=1"], capture_output=True)
    def p99(e):
        subprocess.run([U, "-I", u, "-s", f"exposure-time-abs={int(e)}"], capture_output=True)
        for _ in range(6): cam.read()
        ok, f = cam.read(); g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY); return float(np.percentile(g, 99)), float(g.mean())
    a_, b_ = 5.0, 800.0; best = None
    for _ in range(9):
        e = (a_ * b_) ** 0.5; v, mean = p99(e); best = (e, v, mean)
        if v > hi: b_ = e
        elif v < lo: a_ = e
        else: break
    e, v, mean = best; p99(e)
    print(f"[집게] 노출 고정 {location}: exposure-time-abs {int(e)}, 상위1% {v:.0f}, 평균 {mean:.0f}", flush=True); return dict(exposure=int(e), p99=v, mean=mean)


def record(a):
    """wrist video for SLAM + jaw (10-20 GPT plan: wrist RGB 30 Hz + IMU + Orbbec on one host monotonic clock). Frames are written by a
    writer thread (the read loop never blocks on JPEG encoding); t_ns = host monotonic_ns right after cam.read(). SIGINT / SIGTERM stop
    it cleanly (tag_rig_record --wrist starts and stops it). <out>/started appears with the first frame."""
    import queue, signal, threading
    loc = LOCATIONS[a.hand]; roi = ROIS[a.hand]; cam = open_cam(a.index, location=loc); cam.set(cv2.CAP_PROP_FPS, a.hz)
    expo = fixed_exposure(cam, loc, a.exposure_abs) if a.exposure == "fixed" else lock_exposure(cam, loc) if a.exposure == "lock" else None
    out = Path(a.out) if a.out else ROOT / "datasets/umi_wrist" / time.strftime("%Y%m%d_%H%M%S"); out.mkdir(parents=True, exist_ok=True)
    W, H = (int(v) for v in a.size.split("x")); q: queue.Queue = queue.Queue(maxsize=240); stop = threading.Event()
    def writer():
        while not (stop.is_set() and q.empty()):
            try: k_, f_ = q.get(timeout=0.2)
            except queue.Empty: continue
            cv2.imwrite(str(out / f"wrist_{k_:05d}.jpg"), f_, [cv2.IMWRITE_JPEG_QUALITY, 92])
    ths = [threading.Thread(target=writer, daemon=True) for _ in range(3)]; [t_.start() for t_ in ths]
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    PV = {"jpg": None}
    if a.port:       # 10-20 user: "녹화되는걸 preview로" -- MJPEG of the wrist camera (jaw ROI box) at http://localhost:<port>/stream
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        class Hd(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.startswith("/frame"):      # 11-03: single JPEG (the recorder page polls it; an MJPEG <img> froze dark after a dropped connection)
                    b_ = PV["jpg"] or b""; self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Cache-Control", "no-store")
                    self.send_header("Access-Control-Allow-Origin", "*"); self.send_header("Content-Length", str(len(b_))); self.end_headers(); self.wfile.write(b_); return
                self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f"); self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
                try:
                    while True:
                        if PV["jpg"]: self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + PV["jpg"] + b"\r\n")
                        time.sleep(0.1)
                except (BrokenPipeError, ConnectionResetError): return
            def log_message(self, *x): pass
        threading.Thread(target=ThreadingHTTPServer(("0.0.0.0", a.port), Hd).serve_forever, daemon=True).start()
    T, C, drop = [], [], 0; print(f"[집게] 손목 영상 녹화 {out} ({a.hz:.0f} Hz, {W}x{H}, Ctrl+C 로 끝)", flush=True)
    # 10-29 take 190103: the right wrist camera stopped delivering at +74 s (cam.read() blocked forever, CPU 0 %) and 7 of 10 demos lost the
    # right wrist video.  Frames are now read in a per-camera-generation thread; the main thread is a watchdog: no frame for STALL_S ->
    # reopen the same camera index (only while its USB location is still listed: reopening an unplugged AVFoundation device segfaulted the
    # deploy UI), restore the exposure, continue the SAME frame numbering.  Gaps show up in t_ns.npy; reopens go to meta.json.
    STALL_S = 2.0; S = {"gen": 0, "last": time.monotonic(), "k": 0, "reopens": [], "drop": 0}; lock = threading.Lock()
    idx = a.index if a.index >= 0 else json.loads((ROOT / f"datasets/umi_wrist/cam_index_{loc}.json").read_text())["index"]
    def reader(gen, c):
        while S["gen"] == gen:
            ok, f = c.read(); t = time.monotonic_ns()
            if S["gen"] != gen: return
            if not ok: continue
            with lock:
                k = S["k"]
                if k == 0: (out / "started").write_text(str(t))
                try: q.put_nowait((k, cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)))
                except queue.Full: S["drop"] += 1; continue
                C.append(crop(f, roi)); T.append(t); S["k"] = k + 1; S["last"] = time.monotonic(); k += 1
            if a.port and k % 3 == 0:
                v = cv2.resize(f, (640, 360)); h_, w_ = v.shape[:2]; cv2.rectangle(v, (int(roi[2] * w_), int(roi[0] * h_)), (int(roi[3] * w_), int(roi[1] * h_) - 1), (0, 0, 255), 2)
                cv2.putText(v, f"{a.hand} wrist  {k} fr", (10, 28), 0, 0.8, (0, 255, 0), 2); PV["jpg"] = cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
            if k % int(a.hz * 5) == 0:
                print(f"[집게] {k} 프레임 ({k / ((t - T[0]) / 1e9 + 1e-9):.1f} fps, 버림 {S['drop']})", flush=True)
                # 10-30 take 103251: the left recorder died silently at frame 9247 and never wrote t_ns -> keep a partial copy every 5 s
                # (also the liveness signal tag_rig_record watches)
                with lock: Tp = np.array(T, np.int64)
                np.save(out / "t_ns.partial.npy", Tp)
    SH = {"cam": cam}; threading.Thread(target=reader, args=(0, cam), daemon=True).start()
    try:
        while True:
            time.sleep(0.25)
            if time.monotonic() - S["last"] < STALL_S: continue
            lst = subprocess.run([str(egopaths.UVC_UTIL), "-d"], capture_output=True, text=True).stdout
            if not any(l.split()[2].startswith(loc) for l in lst.splitlines() if "0x0c45" in l and len(l.split()) > 2):
                print(f"[집게] {a.hand} 손목 카메라 {loc} 가 USB 목록에 없음 -- 다시 꽂아 주세요 (기다리는 중)", flush=True); S["last"] = time.monotonic(); continue
            gap = time.monotonic() - S["last"]; print(f"[집게] {a.hand} 손목 카메라 {gap:.1f}초 멈춤 -> 다시 엽니다 (index {idx})", flush=True)
            S["gen"] += 1; gen = S["gen"]; old = cam; threading.Thread(target=old.release, daemon=True).start()
            S["reopens"].append(dict(t_ns=time.monotonic_ns(), after_frame=S["k"], gap_s=round(gap, 2))); S["last"] = time.monotonic() + 8.0   # next try >= 10 s later
            def reopen(gen=gen):        # 10-29 take 191007: VideoCapture() on the stalled device blocked the main thread -> SIGTERM never
                c = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)   # reached it and the parent killed the recorder before t_ns.npy was saved
                c.set(cv2.CAP_PROP_FRAME_WIDTH, 1920); c.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080); c.set(cv2.CAP_PROP_FPS, a.hz)
                try:
                    if a.exposure == "fixed": fixed_exposure(c, loc, a.exposure_abs)
                    elif a.exposure == "lock": lock_exposure(c, loc)
                except Exception as e: print(f"[집게] 노출 재설정 실패: {e}", flush=True)
                if S["gen"] != gen: c.release(); return
                SH["cam"] = c; reader(gen, c)
            threading.Thread(target=reopen, daemon=True).start()
    except KeyboardInterrupt: pass
    S["gen"] = -1; k, drop = S["k"], S["drop"]
    threading.Thread(target=SH["cam"].release, daemon=True).start(); stop.set(); [t_.join(timeout=20) for t_ in ths]
    with lock: T_, C_ = list(T), list(C)              # small files first: t_ns + meta are what the pipeline needs, crops are slow
    np.save(out / "t_ns.npy", np.array(T_, np.int64)); fps = len(T_) / ((T_[-1] - T_[0]) / 1e9) if len(T_) > 1 else 0
    (out / "meta.json").write_text(json.dumps(dict(hz=a.hz, fps_measured=fps, frames=len(T_), dropped=drop, size=[W, H], roi=roi, location=loc, hand=a.hand, exposure=expo, t_clock="host monotonic_ns at frame read", reopens=S["reopens"]), default=str))
    np.savez_compressed(out / "crops.npz", crops=np.array(C_))
    print(f"[집게] 저장 {out} ({k} 프레임, {fps:.1f} fps, 버림 {drop})", flush=True)


def preview(a):
    """10-20: live window of the camera found at LOCATION (ROI box drawn) -- which UMI is it? q / ESC to quit"""
    cam = open_cam(a.index, location=LOCATIONS[a.hand]); roi = ROIS[a.hand]
    while True:
        ok, f = cam.read()
        if not ok: continue
        h, w = f.shape[:2]; cv2.rectangle(f, (int(roi[2] * w), int(roi[0] * h)), (int(roi[3] * w) - 1, int(roi[1] * h) - 1), (0, 0, 255), 6)
        cv2.imshow("UMI wrist (q = quit)", cv2.resize(f, (960, 540)))
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27): break
    cam.release(); cv2.destroyAllWindows()


def events(t_ns, p, hi=0.7, lo=0.3, min_dwell_s=0.2):
    """hysteresis on p_open -> state (1 open / 0 closed) and the confirmed transitions [(t_ns, 'open'|'close')]"""
    st = np.zeros(len(p), int); s = int(p[0] > 0.5); cand = None; ev = []
    for i, x in enumerate(p):
        want = 1 if x > hi else 0 if x < lo else s
        if want != s:
            if cand is None: cand = i
            if (t_ns[i] - t_ns[cand]) / 1e9 >= min_dwell_s:
                s = want; ev.append((int(t_ns[cand]), "open" if s else "close")); st[cand:i] = s; cand = None   # 10-20: state from the EVENT time
        else: cand = None
        st[i] = s
    return st, ev


def grip_signal(jz, mode: str = "state", lo: float = 0.1, hi: float = 0.3) -> np.ndarray:
    """10-21: robot gripper g (0 closed .. 1 open) from umi_jaw.npz. state = hysteresis open/closed; cont = the median-filtered p_open
    mapped linearly lo -> 0, hi -> 1 (frames at p 0.3 are already visibly open, so raw p as a jaw width would under-open the robot)"""
    if mode == "state": return jz["state"].astype(float)
    if "g_is_aperture" in jz.files and int(jz["g_is_aperture"]): return np.asarray(jz["g_cont"], float)   # 10-22: keypoint aperture is already 0..1 (scripts/pose/jaw_aperture.py)
    p = jz["g_cont"] if "g_cont" in jz.files else jz["p_open"]
    return np.clip((np.asarray(p, float) - lo) / (hi - lo), 0.0, 1.0)


def grip_leads(a, cfg_path) -> tuple[float, float]:
    """10-21: (open, close) gripper lead in real robot ms. --grip-lead-ms sets both; otherwise --grip-lead-open-ms / --grip-lead-close-ms,
    then the deploy config (gripper_lead_open_ms / _close_ms, falling back to gripper_lead_ms)"""
    c = json.loads(Path(cfg_path).read_text()) if Path(cfg_path).exists() else {}
    if getattr(a, "grip_lead_ms", None) is not None: return float(a.grip_lead_ms), float(a.grip_lead_ms)
    base = float(c.get("gripper_lead_ms", 0.0))
    lo = a.grip_lead_open_ms if getattr(a, "grip_lead_open_ms", None) is not None else float(c.get("gripper_lead_open_ms", base))
    lc = a.grip_lead_close_ms if getattr(a, "grip_lead_close_ms", None) is not None else float(c.get("gripper_lead_close_ms", base))
    return float(lo), float(lc)


def sample_grip(t_ns, sig, src, lead_open_src: int, lead_close_src: int) -> np.ndarray:
    """g at source times src with opens sent lead_open early and closes lead_close early (both in source ns). An open interval [a, b]
    becomes [a - lead_open, b - lead_close]: min of the two shifted signals when lead_close >= lead_open, max otherwise. Open intervals
    shorter than lead_close - lead_open vanish (the 0.2 s min dwell keeps that rare)."""
    f = lambda L: sig[np.clip(np.searchsorted(t_ns, np.asarray(src) + L, side="right") - 1, 0, len(t_ns) - 1)]
    go, gc = f(lead_open_src), f(lead_close_src)
    return np.minimum(go, gc) if lead_close_src >= lead_open_src else np.maximum(go, gc)


def infer(a):
    import torch
    d = Path(a.dir); t = np.load(d / "t_ns.npy"); meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    hand = meta.get("hand", "right"); roi = ROIS[hand]
    try: crops = np.load(d / "crops.npz")["crops"]
    except Exception:       # 10-22: take stopped abruptly -> crops.npz truncated; re-crop from the saved frames
        crops = np.stack([crop(cv2.imread(str(d / f"wrist_{k:05d}.jpg")), ROIS[hand]) for k in range(len(t))]); print(f"[집게] crops.npz 손상 -> 저장 프레임에서 다시 자름 ({len(crops)})", flush=True)
    old_era = not np.allclose(meta.get("roi", ROI), ROIS[hand]) and np.allclose(meta.get("roi", ROI), _ROIS_BEFORE_1022[hand])   # 10-22: takes recorded before the camera re-angle
    if old_era:
        roi = _ROIS_BEFORE_1022[hand]; a.model = a.model or str(MODELS_V002[hand]); print(f"[집게] 10-22 이전 카메라 각도 녹화 -> 예전 ROI + {Path(a.model).name}", flush=True)
    if not a.model:
        a.model = str(MODELS[hand]) if MODELS[hand].exists() else str(MODEL)
        print(f"[집게] {hand} 손 모델: {Path(a.model).name}" + ("  (손별 모델 없음 -> v001, 믿지 마세요)" if a.model == str(MODEL) else ""), flush=True)
    if not np.allclose(meta.get("roi", ROI), roi):      # 10-21: recorded with an older ROI -> re-crop from the saved frames
        crops = np.stack([crop(cv2.imread(str(d / f"wrist_{k:05d}.jpg")), roi) for k in range(len(t))]); print(f"[집게] 녹화 때 ROI 가 지금과 달라 저장 프레임에서 다시 자름 ({len(crops)})", flush=True)
    net = torch.jit.load(str(a.model)).eval()
    with torch.no_grad():
        X = np.stack([to_input(c) for c in crops]); p = torch.sigmoid(torch.cat([net(torch.tensor(X[i:i + 64])).squeeze(1) for i in range(0, len(X), 64)])).numpy()
    if not old_era and a.hi == 0.3 and a.lo == 0.1: a.hi, a.lo = 0.6, 0.4   # 10-22 v003: balanced classifier -> symmetric hysteresis
    elif hand == "left" and a.hi == 0.3 and a.lo == 0.1:   # 10-21: the left model gives CLOSED jaws p 0.2-0.35 on washed-out frames -> 0.6/0.4
        a.hi, a.lo = 0.6, 0.4; print("[집게] 왼손 기준 0.6 / 0.4", flush=True)
    st, ev = events(t, p, a.hi, a.lo, a.min_dwell)
    # 10-21 user: continuous (sigmoid) gripper = p_open median-filtered over ~0.2 s. NOT a jaw-width measure: p saturates as soon as the
    # jaws part (frames at p 0.2-0.4 are already visibly open), so it is a soft open/closed with smooth edges
    kk = max(1, int(round(0.2 * (len(t) - 1) / max((t[-1] - t[0]) / 1e9, 1e-9)))) | 1
    g_cont = np.array([np.median(p[max(0, i - kk // 2):i + kk // 2 + 1]) for i in range(len(p))], np.float32)
    np.savez(d / "umi_jaw.npz", t_ns=t, p_open=p, g_cont=g_cont, state=st, hand=hand, model=Path(a.model).name, ev_t=np.array([e[0] for e in ev], np.int64), ev_kind=np.array([e[1] for e in ev]))
    print(f"[집게] {d.name}: 프레임 {len(p)}, 열림 비율 {st.mean():.2f}, 사건 {len(ev)}개: " + ", ".join(f"{(e[0] - t[0]) / 1e9:.2f}s {'열림' if e[1] == 'open' else '닫힘'}" for e in ev), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture"); c.add_argument("--cycles", type=int, default=6); c.add_argument("--phase-s", type=float, default=8.0); c.add_argument("--hz", type=float, default=8.0); c.add_argument("--index", type=int, default=-1); c.add_argument("--hand", required=True, choices=["right", "left"])
    t = sub.add_parser("train"); t.add_argument("captures", nargs="+"); t.add_argument("--hand", required=True, choices=["right", "left"]); t.add_argument("--val", nargs="*", default=[]); t.add_argument("--epochs", type=int, default=6); t.add_argument("--skip-s", type=float, default=1.5); t.add_argument("--out", default="")
    r = sub.add_parser("record"); r.add_argument("--hz", type=float, default=30.0); r.add_argument("--index", type=int, default=-1); r.add_argument("--out"); r.add_argument("--size", default="1280x720"); r.add_argument("--hand", default="right", choices=["right", "left"]); r.add_argument("--port", type=int, default=0); r.add_argument("--exposure", default="fixed", choices=["fixed", "lock", "auto"]); r.add_argument("--exposure-abs", type=int, default=2, help="fixed 모드 노출값 (10-22 sweep: 1-2 + IMAGE_CTRL 이 가장 선명)")
    v = sub.add_parser("preview"); v.add_argument("--index", type=int, default=-1); v.add_argument("--hand", default="right", choices=["right", "left"])
    i = sub.add_parser("infer"); i.add_argument("dir"); i.add_argument("--model", default=""); i.add_argument("--hi", type=float, default=0.3); i.add_argument("--lo", type=float, default=0.1); i.add_argument("--min-dwell", type=float, default=0.2)   # 10-21: hi/lo were 0.7 / 0.3; p 0.3-0.7 frames are open (jaw half hidden by cloth)
    a = ap.parse_args(); {"capture": capture, "train": train, "record": record, "infer": infer, "preview": preview}[a.cmd](a)


if __name__ == "__main__":
    main()
