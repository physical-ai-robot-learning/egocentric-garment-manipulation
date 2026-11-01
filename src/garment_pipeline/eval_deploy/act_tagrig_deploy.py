#!/usr/bin/env python3
"""[2026-10-23] Closed-loop real-robot deploy of the tag-rig laundry policies on the bimanual reBot B601-DM:
  --policy act  act_tagrig_v1_1m 500k (default)          --policy dit  dit_tagrig_v1 (checkpoints/last; Multi-Task DiT, DDIM 10)
DRY-RUN by default (reads cameras + robot, runs everything, sends NOTHING).  --go sends; only with a person at the robot.

Receding horizon (user review 10-23): every plan = fresh images + measured q -> FK(q) = T(t_obs) -> chunk (32 x 20 Hz) -> COMMIT
only --commit steps (ACT default 1, DiT default 4) -> re-plan.  Inference runs in this planner loop while the 30 Hz Streamer thread
keeps streaming the last accepted segment (it never waits for the policy).  Latency re-anchoring: when the chunk is ready (t_ready),
the latest measured TCP T_now is read, k0 = floor((t_ready - t_obs) / 50 ms) steps of the chunk are already "in the past", and the
committed targets are T_now @ inv(dT_k0) @ dT_k, k = k0+1 .. k0+commit (dT_0 = I): the plan is always anchored on the newest TCP.
Stages per committed step and arm, each logged: candidate -> limiter -> IK -> safety -> accepted -> commit (30 Hz interpolated stream).
  limiter   [temporal ensemble, ACT] -> speed scale s (--slow-start-speed for --slow-start-s, then --speed) as a fraction of the
            step from T_now -> Cartesian clamp (--max-tcp-mps, --max-rot-dps) -> finger-corner table floor (robotize_ik corner_lift)
  IK        robotize_ik.Arm.free = umi_replay_bi Pink settings (bridge joint limits 0.5 deg inside, ori cost 0.1, 7.5 deg/step bound),
            seeded with the last accepted q; then the joint-speed limiter scales dq to <= --max-deg-s
  safety    (hard: STOP + HOLD) NaN, workspace box (training pose_abs range +- --ws-margin, per arm), IK error > 10 mm / 20 deg,
            joint limits, capsule link-link collision between the arms (clearance < --arm-clear; core.Collision, capsules because
            the URDF link meshes are not on this Mac), link capsules under the table, TCP below table + 5 mm, bridge reject /
            E-STOP / stale read / stale image.  Warning only: link-origin gap < 120 mm -> logged + speed x0.5 on the next plan.
STOP = br.estop() = the MIT bridge holds the measured pose of both arms.  Nothing moves afterwards, ever, except: key r + confirm y
(--go only) -> /clear_estop + slow joint ramp to READY (umi_replay_bi.ramp_both).  Keys (this terminal): SPACE = stop + hold,
q = quit (stop + hold first); after a stop: r = return to READY (asks y/n).  Gripper: chunk row --grip-step, > 0.5 open (g 1) else closed.
Dry-run: same pipeline, nothing sent, violations are logged and the run continues ("would STOP"); state gripper = the virtual command.
Global camera = laundry_system/robot/orbbec_server.py (sudo, OrbbecCapture 848x480 = the collector's settings) at :8092.
Wrist cameras = robot Arducam fisheyes by USB location (deploy/act_tagrig_cams.json; `cams` subcommand), fixed exposure + IMAGE_CTRL
of umi_jaw record.  Output: reports/act_deploy/<run>/{log.npz, summary.json}, Desktop mp4 (Orbbec projection + top-down + wrists).
    sudo ~/ego_collector/.venv/bin/python ~/laundry_system/robot/orbbec_server.py          # terminal 1 (global camera)
    .venv-mac/bin/python deploy/act_tagrig_deploy.py cams [--cover]                         # once after any USB replug
    .venv-mac/bin/python deploy/act_tagrig_deploy.py run [--policy dit] --secs 30           # DRY-RUN (robot at READY)
    .venv-mac/bin/python deploy/act_tagrig_deploy.py run [--policy dit] --go --secs 60      # sends commands (person at the robot)"""
from __future__ import annotations
import argparse, collections, json, os, pathlib, subprocess, sys, threading, time

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
from garment_pipeline.eval_deploy import act_tagrig_core as core  # noqa: E402
from egogarment import paths as egopaths  # noqa: E402

CAMS_CFG = HERE / "act_tagrig_cams.json"
UVC = str(egopaths.UVC_UTIL)
DESKTOP = egopaths.DESKTOP
OUT = core.LDT / "reports/act_deploy"
START_FILE = HERE / "start_pose.json"
HIST_TOL_S = 0.03          # n_obs 2 (DiT): accept a history sample within 50 +- 30 ms (dit_serving HISTORY_TOL_S)


def say(t):
    print("[배포]", t, flush=True)


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


GRIP_OPEN_FRAC_MAX = {"left": 0.5, "right": 1.0}   # 10-27: the 10-23 bridge ran MIT_GRIP_OPEN_FRAC_LEFT=0.5; restarted without it -> left jaw over-opened, holder broke


def check_grip_limits(url):
    """--go refuses unless the bridge limits each jaw's opening to at most GRIP_OPEN_FRAC_MAX (bridge /status grip_open_frac)"""
    import requests
    try: got = requests.get(f"{url.rstrip('/')}/status", timeout=2).json().get("grip_open_frac") or {}
    except Exception as ex: raise SystemExit(f"bridge /status 읽기 실패 ({ex}) -- 그리퍼 열림 제한을 확인할 수 없어 GO 거부")
    bad = {s: got.get(s) for s, mx in GRIP_OPEN_FRAC_MAX.items() if got.get(s) is None or float(got[s]) > mx + 1e-6}
    if bad: raise SystemExit(f"그리퍼 열림 제한 불일치 {bad} (허용 최대 {GRIP_OPEN_FRAC_MAX}) -- bridge 를 MIT_GRIP_OPEN_FRAC_LEFT=0.5 로 다시 띄우세요. GO 거부")


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


class RobotPoller(threading.Thread):
    """/observe at ~25 Hz -> ring of (q, g_obs, estop); --assume-ready: FK(READY) forever (dry-run without a live robot)"""
    def __init__(self, br, qr, assume_ready, hz=25.0):
        super().__init__(daemon=True); self.br, self.qr, self.fake, self.hz = br, qr, assume_ready, hz; self.ring = Ring(20); self.stop = False; self.error = None; self.lat = []

    def read(self):
        if self.fake: return {s: self.qr[s].copy() for s in core.SIDES}, {s: None for s in core.SIDES}, False
        o = self.br.observe(); self.lat.append(o.latency_ms); return o.q, {s: o.g(s) for s in core.SIDES}, o.estop

    def run(self):
        while not self.stop:
            # 10-30: a bridge restart left self.error = "/observe: not connected" forever (never cleared after reads recovered) and
            # every later UI run stopped at once with "robot read: ..." -> clear it on the next good read
            try: self.ring.put(time.monotonic_ns(), self.read()); self.error = None
            except Exception as e: self.error = f"{type(e).__name__}: {e}"; self.ring.put(time.monotonic_ns(), (None, None, True))
            time.sleep(1.0 / self.hz)


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


# ---------------------------------------------------------------- keyboard (tag_rig_record /dev/tty cbreak pattern)
class Keys:
    def __init__(self):
        self.q = []; self.fd = None
        try:
            import termios, tty, atexit
            self.fd = os.open("/dev/tty", os.O_RDONLY | os.O_NONBLOCK); old = termios.tcgetattr(self.fd); tty.setcbreak(self.fd)
            atexit.register(lambda: termios.tcsetattr(self.fd, termios.TCSADRAIN, old))
            threading.Thread(target=self._loop, daemon=True).start()
        except Exception as e: print(f"[배포] 키 입력 불가 (/dev/tty: {e}) -- Ctrl+C = 정지", flush=True)

    def _loop(self):
        import select
        while True:
            r, _, _ = select.select([self.fd], [], [], 0.1)
            if not r: continue
            try: data = os.read(self.fd, 16).decode(errors="ignore")
            except BlockingIOError: continue
            self.q.extend(data)

    def pop(self):
        out, self.q = self.q, []; return out

    def wait(self, valid, timeout=None):
        t0 = time.monotonic()
        while timeout is None or time.monotonic() - t0 < timeout:
            for ch in self.pop():
                if ch.lower() in valid: return ch.lower()
            time.sleep(0.05)
        return None


# ---------------------------------------------------------------- 30 Hz command streamer
class Streamer(threading.Thread):
    """sends q interpolated along an INDEXED timeline of joint targets (tick i = t_base + i * DT) at --stream-hz (only with --go).
    [10-29] indy7 c4fba14 (bh_indy7_LeRobot inference/action_timeline.merge_chunk): a chunk row is keyed by the tick it was predicted
    for; merge() drops rows for ticks already driven (no pulling back along the covered path), blends rows for ticks still pending
    (crossfade, new weight rising) and appends the rest.  set() = restart the timeline from where the stream is now (recovery, step)."""
    def __init__(self, br, q0, g0, hz, dt=None):
        super().__init__(daemon=True); self.br, self.hz = br, hz; self.g = dict(g0); self.lock = threading.Lock(); self.stop = False; self.error = None
        self.dt = float(dt or core.DT)             # tick period (look-ahead: the policy's row period / speed)
        self.n = 0; self.last = None; self.t_base = time.monotonic(); self.tl = {0: {s: q0[s].copy() for s in core.SIDES}}; self.n_dry = 0

    def tick(self, t=None):
        return ((time.monotonic() if t is None else t) - self.t_base) / self.dt

    def executed(self, t=None):
        return int(np.floor(self.tick(t)))

    @property
    def last_idx(self):
        return max(self.tl)

    def _q_at(self, x):
        import bisect
        ks = sorted(self.tl); x = max(x, ks[0])
        if x >= ks[-1]: return self.tl[ks[-1]]
        j = bisect.bisect_right(ks, x) - 1; i0_, i1_ = ks[j], ks[j + 1]; f = (x - i0_) / (i1_ - i0_)
        return {s: self.tl[i0_][s] + f * (self.tl[i1_][s] - self.tl[i0_][s]) for s in core.SIDES}

    def anchor(self, t):
        """the tick index at time t and the COMMANDED q at that tick (the pose a chunk predicted from an observation at t starts from)"""
        with self.lock:
            i = self.executed(t); return i, {s: v.copy() for s, v in self._q_at(i).items()}

    def merge(self, i0, rows, g):
        """rows[j] = q for tick i0 + 1 + j -> (rows dropped as already driven, rows blended)"""
        with self.lock:
            ex = self.executed() + 1                # the tick being interpolated toward right now is left untouched
            new = {i0 + 1 + j: r for j, r in enumerate(rows) if i0 + 1 + j > ex}
            ov = sorted(i for i in new if i in self.tl)
            for m, i in enumerate(ov):
                w = (m + 1) / (len(ov) + 1); self.tl[i] = {s: (1 - w) * self.tl[i][s] + w * new[i][s] for s in core.SIDES}
            for i in new:
                if i not in self.tl: self.tl[i] = new[i]
            for i in [i for i in self.tl if i < ex - 1]: del self.tl[i]
            self.g = dict(g); return len(rows) - len(new), len(ov)

    def set(self, waypts, dur_each, g):
        """waypts: list of {side: q} (the committed steps); each reached after one tick (self.dt), from where the stream is now"""
        with self.lock:
            now = time.monotonic(); q = self.last if self.last is not None else self._q_at(self.tick(now))
            self.t_base = now; self.tl = {0: {s: q[s].copy() for s in core.SIDES}}
            for i, w in enumerate(waypts): self.tl[i + 1] = w
            self.g = dict(g)

    def _q(self):
        with self.lock:
            x = self.tick()
            if x > self.last_idx: self.n_dry += 1   # timeline ran dry: hold the last target (counted at stream rate)
            return self._q_at(x), self.g

    def run(self):
        nxt = time.monotonic()
        while not self.stop:
            q, g = self._q(); self.last = {s: q[s].copy() for s in core.SIDES}
            r = self.br.send(q, g); self.n += 1
            if not r.accepted: self.error = f"bridge reject: {r.error}"; return
            nxt += 1.0 / self.hz; time.sleep(max(0.0, nxt - time.monotonic()))


# ---------------------------------------------------------------- run
# ---------------------------------------------------------------- 5 Hz preview (whole chunk)
class Preview(threading.Thread):
    """~5 Hz: IK over the WHOLE latest chunk (every --preview-stride step, seeded step to step, unbounded solver robotize_ik f0) +
    workspace / capsule collision / table on those future joint poses.  Raises WARN flags; with --preview-hold a predicted future
    collision or table hit becomes a HOLD (STOP) for the control path.  Off the control path (never delays a commit)."""
    def __init__(self, arms, col, WS, a, table_z):
        super().__init__(daemon=True); self.arms, self.col, self.WS, self.a, self.table_z = arms, col, WS, a, table_z
        self.job = None; self.lock = threading.Lock(); self.stop = False; self.res = None; self.log = []

    def submit(self, t, Tc, q_seed):
        with self.lock: self.job = (t, Tc, q_seed)

    def run(self):
        while not self.stop:
            with self.lock: job, self.job = self.job, None
            if job is None: time.sleep(0.01); continue
            t0 = time.monotonic(); t, Tc, qs = job; ks = list(range(self.a.preview_stride - 1, 32, self.a.preview_stride)); flags = []
            q = {s: qs[s].copy() for s in core.SIDES}; err = {s: [] for s in core.SIDES}; clr, tab = [], []
            for k in ks:
                for s in core.SIDES:
                    q[s], pe, re = self.arms[s].f0.ik(q[s], Tc[s][k]); err[s].append(pe * 1e3)
                    if np.any(Tc[s][k][:3, 3] < self.WS[s][0]) or np.any(Tc[s][k][:3, 3] > self.WS[s][1]): flags.append(f"WARN:future_workspace_{s[0]}_k{k + 1}")
                cc = self.col.check(q, table_z=self.table_z, arm_clear=self.a.arm_clear); clr.append(cc["arm_clear_m"]); tab.append(cc["table_clear_m"])
                if not cc["ok_arm"]: flags.append(f"{'HOLD' if self.a.preview_hold else 'WARN'}:future_collision_k{k + 1}_{cc['pair']}")
                if not cc["ok_table"]: flags.append(f"{'HOLD' if self.a.preview_hold else 'WARN'}:future_table_k{k + 1}")
            ok = {s: float(np.mean(np.array(err[s]) <= self.a.max_ik_mm)) for s in core.SIDES}
            for s in core.SIDES:
                if ok[s] < 1: flags.append(f"WARN:future_ik_{s[0]}_{max(err[s]):.0f}mm")
            r = dict(t=t, ik_ok=[ok["left"], ok["right"]], ik_max_mm=[max(err["left"]), max(err["right"])], min_clear=float(min(clr)), min_table=float(min(tab)),
                     flags=flags, ms=(time.monotonic() - t0) * 1e3)
            self.res = r; self.log.append(r)
            time.sleep(max(0.0, 1.0 / self.a.preview_hz - (time.monotonic() - t0)))


# ---------------------------------------------------------------- run
class Rig:
    """persistent hardware/kinematics side shared by every run (CLI: one run; UI: many): wrist cameras, live Orbbec, 25 Hz robot
    poller (read-only HttpBridge), pinocchio arms + capsule collision, and a cache of loaded policies"""
    def __init__(self, a):
        from visual_teleop import config as vcfg
        from visual_teleop.robot.bridge import HttpBridge
        from visual_teleop.tools.robot_moves import READY_Q_DEG
        rec = recorder_running()
        if rec: raise SystemExit(f"녹화/카메라 프로세스가 실행 중 -- 끄고 다시: {rec}")
        self.c = vcfg.load(); self.url = getattr(a, "bridge_url", None) or self.c.robot["base_url"]
        self.qr = {s: np.radians(READY_Q_DEG[s]) for s in core.SIDES}
        cfg = json.loads(CAMS_CFG.read_text()); lst = uvc_list()
        if not [x for x in os.environ.get("DEPLOY_FAKE_WRIST", "").split(",") if x in core.SIDES]: cfg = auto_remap_cams(cfg, lst)
        self.cfg = cfg
        fake = [x for x in os.environ.get("DEPLOY_FAKE_WRIST", "").split(",") if x in core.SIDES]
        if fake: say(f"주의: {fake} 손목 카메라 = 검은 화면 (DEPLOY_FAKE_WRIST, 학습 분포 밖 -- 테스트 전용)")
        for s in core.SIDES:
            if s in fake: continue
            if not cfg.get(s, {}).get("location") or cfg[s]["location"].lower() not in lst.lower(): raise SystemExit(f"{s} 손목 카메라 USB 위치 {cfg.get(s, {}).get('location')} 없음 -- 연결/매핑 확인 (wave 테스트)")
        if cfg.get("cam_key") != cam_key() and not fake: raise SystemExit("USB 카메라 구성이 바뀜 -- 먼저: act_tagrig_deploy.py cams (index 다시 찾기)")
        self.wave_ok = bool(cfg.get("wave_test", {}).get("confirmed"))
        if not self.wave_ok: say("주의: 손목 카메라 좌우가 wave 테스트로 아직 확인되지 않음 (act_tagrig_cams.json wave_test)")
        umi_location_warning(cfg)
        self.arms, self.ra = core.make_arms(); self.col = core.Collision(self.arms)
        self.fake_q = None
        if getattr(a, "assume_start", False):
            if not START_FILE.exists(): raise SystemExit("start_pose.json 없음 -- 먼저 `start`")
            self.fake_q = {s: np.asarray(json.loads(START_FILE.read_text())["q"][s]) for s in core.SIDES}
        elif getattr(a, "assume_ready", False): self.fake_q = self.qr
        self.br = None if self.fake_q is not None else HttpBridge(self.url, live=False, timeout_s=max(self.c.robot["request_timeout_s"], 0.3))
        self.rob = RobotPoller(self.br, self.fake_q if self.fake_q is not None else self.qr, self.fake_q is not None)
        align = cfg.get("wrist_align") if getattr(a, "wrist_align", False) else None
        self.replay = bool(getattr(a, "replay_global", ""))
        self.gl = GlobalCam(("dir:" + a.replay_global) if self.replay else a.global_url)
        self.wr = {s: (FakeWristCam(s) if s in fake else WristCam(s, cfg[s]["location"], cfg[s]["index"], a.exposure, (align or {}).get(s))) for s in core.SIDES}
        self.runners = {}

    def start(self):
        for x in (*self.wr.values(), self.gl, self.rob): x.start()
        t_w = time.monotonic()
        while any(x.ring.latest()[1] is None for x in (*self.wr.values(), self.gl)):
            if time.monotonic() - t_w > 8: raise SystemExit(f"카메라 프레임 없음: { {s: w.n for s, w in self.wr.items()} } global {self.gl.n}")
            time.sleep(0.05)
        return self

    def runner(self, ckpt, device, n_samples):
        key = (str(ckpt), n_samples)
        if key not in self.runners:
            self.runners = {}                       # keep ONE model in memory (DiT is 1.1 GB)
            is_xvla = '"xvla"' in (pathlib.Path(ckpt) / "pretrained_model" / "config.json").read_text() if (pathlib.Path(ckpt) / "pretrained_model" / "config.json").exists() else False
            self.runners[key] = core.XvlaRunner(ckpt) if is_xvla else core.PolicyRunner(ckpt, device, n_samples=n_samples)   # 10-28 X-VLA = server
        return self.runners[key]

    def close(self):
        for x in (*self.wr.values(), self.gl, self.rob): x.stop = True
        if self.br is not None: self.br.close()


class KeysCtl:
    """CLI control: SPACE / q from this terminal stop the run"""
    def __init__(self): self.keys = Keys()

    def poll(self):
        ks = self.keys.pop()
        if " " in ks: return "SPACE"
        if "q" in ks or "Q" in ks: return "q"
        return None

    def publish(self, snap): pass


def check_run_args(a):
    dit = a.policy == "dit"
    if a.commit is None: a.commit = 4 if (dit or a.policy == "label") else 1
    if a.policy == "label" and not getattr(a, "label", ""): raise SystemExit("--policy label 에는 --label <ik_v1/demoNN.npz> 필요")
    if not 1 <= a.commit <= 32: raise SystemExit(f"--commit {a.commit} 불가 (1..32 = the whole chunk)")     # 10-23 user: up to the full 32-step chunk
    # 11-01 user "지금 너무 느린데": the S03/S05 sets are time-stretched, so speed > 1 is allowed; retime() then reaches the chunk end
    # after 32 / speed rows and the rest would hold the last pose -> commit at most that many rows
    if dit and a.speed > 1.0 and a.commit > int(32 / a.speed):
        say(f"속도 x{a.speed}: commit {a.commit} -> {int(32 / a.speed)} (청크 끝을 넘지 않게)"); a.commit = int(32 / a.speed)
    # 10-23 user: commit 8 allowed with --go after the first GO at commit 1 drifted (ACT is trained to execute whole chunks)
    if a.go and (a.assume_ready or a.assume_start): raise SystemExit("--assume-ready / --assume-start 는 dry-run 전용")
    if a.go and a.replay_global: raise SystemExit("--replay-global (녹화 프레임) 은 테스트 전용 -- --go 금지")
    # 10-23 user: DiT plans disagree by ~16 mm (k8, p50) from plan to plan -> back-and-forth; the ensemble now applies to DiT too


def run_cmd(a):
    check_run_args(a)
    rig = Rig(a).start(); ctl = KeysCtl()
    try:
        summ = run_session(a, rig, ctl)
        if a.go and not getattr(a, "no_prompt", False):
            print("다음 동작은 사람만: r = READY 로 천천히 복귀 (확인 필요), q = 종료 (자세 유지)", flush=True)
            k = ctl.keys.wait("rq")
            if k == "r":
                print("정말 READY 로 복귀? 경로 주변을 비우고 y (다른 키 = 취소)", flush=True)
                if ctl.keys.wait("yn", timeout=30) == "y":
                    from visual_teleop.robot.bridge import HttpBridge
                    brl = HttpBridge(rig.url, live=True, timeout_s=0.5)
                    try: go_ready(brl, rig.arms, rig.qr)
                    finally: brl.close()
    finally: rig.close()
    return summ


def run_session(a, rig, ctl):
    """ONE run (dry or --go) on a started Rig.  ctl.poll() -> stop reason or None (keys / UI button / heartbeat watchdog);
    ctl.publish(snapshot) every cycle (UI).  --go: a live HttpBridge (single writer) exists only inside this function; any stop
    -> streamer stopped + br.estop() (bridge holds the measured pose)."""
    check_run_args(a); dit = a.policy == "dit"
    # 10-28 user: step mode (GO only) -- the policy keeps planning and the UI shows each plan, but a plan is sent ONLY when the UI
    # [한 스텝 실행] button set a step token (ctl.take_step); one token = one committed segment (a.commit steps), then hold again.
    # A plan with STOP/HOLD flags is never sent; in step mode it does not end the run (the token is consumed and the step refused).
    step = bool(getattr(a, "step_mode", False)) and a.go
    # 10-28 user: one-arm execution -- the other arm's plan is computed and shown (dry) but its joints/jaw stay at the last command;
    # its own safety flags become DRY:... (logged, never block the active arm); arm-arm collision is checked with the held pose.
    active = [x for x in str(getattr(a, "active_arms", "both")).split(",") if x in core.SIDES] or list(core.SIDES)
    if len(active) < 2: say(f"한 팔 실행: {active} 만 움직임, 나머지는 계획만 표시 (DRY)")
    take_step = getattr(ctl, "take_step", lambda: False)
    from visual_teleop.robot.bridge import HttpBridge
    from visual_teleop.tools.robot_moves import unpark
    arms, ra, col, rob, gl, wr, qr = rig.arms, rig.ra, rig.col, rig.rob, rig.gl, rig.wr, rig.qr; cfg = rig.cfg
    lo = {s: arms[s].lo for s in core.SIDES}; hi = {s: arms[s].hi for s in core.SIDES}
    for s in core.SIDES:            # 10-23 user: let the fingers reach / scrape the table (umi replay 10-21 used --tip-clear -0.01)
        arms[s].a.tip_clear = a.tip_clear; arms[s].a.tip_clear_left_extra = 0.0
    if a.go:
        if not rig.wave_ok: raise SystemExit("--go 전에 wave 테스트로 손목 카메라 좌우를 확인하세요")
        if rig.replay or rig.fake_q is not None: raise SystemExit("--go 는 라이브 Orbbec + 실제 로봇 상태에서만")
    mock = rig.url.rstrip("/") != rig.c.robot["base_url"].rstrip("/")         # e.g. the MIT_MOCK bridge on :8031
    run_dir = OUT / (("mock_" if mock else "") + f"{a.policy}_" + ("go_" if a.go else "dry_") + time.strftime("%Y%m%d_%H%M%S")); run_dir.mkdir(parents=True, exist_ok=True)
    ckpt = a.label if a.policy == "label" else (a.ckpt or str(core.DIT_CKPT if dit else core.CKPT))
    say(f"{a.policy.upper()} {ckpt} | {'실행 (--go)' if a.go else 'DRY-RUN (명령 안 보냄)'}  commit {a.commit}, speed {a.speed} (처음 {a.slow_start_s}s 는 {a.slow_start_speed}), 최대 {a.secs}s -> {run_dir}")
    if a.policy == "label":                          # 10-27 label replay (A' vs Cg): the label as an oracle policy, same loop
        runner = core.LabelRunner(a.label, a.label_rate)
        a.secs = runner.dur / a.label_rate * 2.0 + 10.0     # cursor mode: ends at the label end; secs = only a safety cap
        say(f"라벨 replay: {a.label} ({runner.dur:.1f}s 라벨, 속도 x{a.label_rate} -> {a.secs:.0f}s)")
    else: runner = rig.runner(ckpt, a.device, a.dit_samples)
    if hasattr(runner, "grip_fake_open"):            # 10-29 diagnostic: jaw state fed as OPEN (copycat checkpoints never open on their own)
        runner.grip_fake_open = bool(getattr(a, "grip_fake_open", False))
        if runner.grip_fake_open: say("진단: 상태 입력의 집게를 항상 '열림'으로 넣음 (집게 명령은 청크 중간 행)")
        if getattr(runner, "grip_mask", False): say("집게 상태 가림 체크포인트: 상태 입력의 집게 = 0")
    xvla = getattr(runner, "kind", "") == "xvla"; retimed = xvla or dit     # 10-29: X-VLA, then DiT (user "dit에도 적용해줘")
    if xvla:                                         # 10-29: X-VLA speed = time scale along the predicted path (amplitude scale stays 1)
        runner.time_scale = float(a.speed); say(f"X-VLA: 15 Hz 청크를 20 Hz 실행 행으로 재샘플, 속도 x{a.speed} = 시간 배율 (경로는 그대로)")
    elif dit: say(f"DiT: 속도 x{a.speed} = 시간 배율 (청크 경로를 그대로 천천히)")
    # 10-29 indy7 c4fba14 look-ahead (continuous GO of X-VLA / DiT): the next forward starts while the current chunk still runs, the
    # chunk is anchored on the COMMANDED pose at the observation tick and merged into the streamer's indexed timeline (Streamer.merge)
    # 10-30 user (backstep at every chunk switch, 37/40 cycles -40 mm): --no-lookahead = run the whole commit, THEN observe + infer
    # 10-30 user "z offset": height correction of the WHOLE trajectory -- added once to the first committed chunk (ramped over 16 rows);
    # later chunks chain on the commanded pose (q_acc), so it neither accumulates nor fades.  Not a limit: any value, 0 = off.
    z_off_m = float(getattr(a, "z_offset_mm", 0.0)) / 1000.0; z_off_done = z_off_m == 0.0
    # 11-01 user "로봇에 맞추어서": --height-calib adds the per-arm table-height error measured by deploy/height_calib.py (robot jaw tip on the
    # table tags vs the training B frame), on top of any manual z offset.  The measured error is a PLANE (left 5 touches: tilt ~4.9 deg,
    # +3 mm near the base .. -24 mm far out), so it follows the TCP: every committed row gets dz(row xy) minus what the commanded pose
    # already carries (z_base); the first chunk ramps in over 16 rows like the manual offset
    hcal = None; z_off = {s: z_off_m for s in core.SIDES}; z_base = {s: 0.0 for s in core.SIDES}
    if getattr(a, "height_calib", False):
        import height_calib as HC; hcal = HC.load(); z_off_done = False
        if hcal is None: raise SystemExit("deploy/height_calib.json 없음 -- 먼저 `python deploy/height_calib.py measure`")
        say("높이 캘리브레이션 " + hcal["created"] + ": " + ", ".join(f"{s} {f['dz_mean_mm']:+.1f} mm" for s, f in hcal["fit"].items()))
    look = retimed and a.go and not step and not getattr(a, "no_lookahead", False); lat_hist = collections.deque(maxlen=5); n_dropped = n_blend = 0
    # 10-29 user "Real time chunking도 같이 구현해줘": look-ahead only (rows keyed by tick).  The previous plan = the COMMANDED timeline
    # (after limiter / IK) re-anchored at this observation's tick; jaws from the committed rows.  guided = RTC guidance (DiT, one VJP per
    # DDIM step), blend = gradient-free clean-chunk inpainting (X-VLA always: a VJP through the 0.9B model would triple its ~950 ms).
    rtc_mode = getattr(a, "rtc", "off") if look and getattr(runner, "kind", "") in ("multi_task_dit", "xvla") else "off"
    if rtc_mode != "off" and getattr(runner, "kind", "") == "xvla": rtc_mode = "blend"
    rtc_g, rtc_log = {}, []
    if rtc_mode != "off": say(f"Real-Time Chunking: {rtc_mode} (beta {getattr(a, 'rtc_beta', 5.0)})")
    dt_row = core.DT                                 # robot seconds per executed chunk row (limiter caps, pacing)
    if look:
        # the model's own rows are executed (no resampling): one tick = row period / speed (X-VLA 1/15 s, DiT 1/20 s at speed 1) -> the
        # 32-row chunk spans 2.13 s / 1.6 s at speed 1, half of it covers one forward (X-VLA ~0.9 s on MPS, DiT 0.2-0.45 s)
        dt_row = (core.XvlaRunner.ROW_DT if xvla else core.DT) / max(float(a.speed), 0.05)
        if xvla: runner.time_scale = None            # XvlaRunner.predict: raw 15 Hz rows
        say(f"look-ahead (indy7 / LeRobot async: 남은 행이 청크 절반 이하면 다음 추론, 겹치는 행 섞음): 행 간격 {dt_row * 1e3:.0f} ms (속도 x{a.speed})")
    WS = runner.workspace(a.ws_margin)
    if dit and getattr(a, "dit_steps", None): runner.pol.objective.num_inference_steps = int(a.dit_steps)
    q0, g_obs0, est = rob.read()
    if est:
        if a.go: raise SystemExit("로봇 상태 E-STOP/미연결/읽기 지연 -- --go 불가 (브리지 /status 의 read_age_ms 확인)")
        raise SystemExit("로봇 상태 E-STOP/미연결/읽기 지연 (읽기 지연 = 로봇 꺼짐/USB). 로봇 없이 확인만: --assume-ready")
    off = {s: float(np.degrees(np.abs(q0[s] - qr[s])).max()) for s in core.SIDES}
    sp = json.loads(START_FILE.read_text()) if START_FILE.exists() else None
    off_start = {s: float(np.degrees(np.abs(q0[s] - np.asarray(sp["q"][s]))).max()) for s in core.SIDES} if sp else None
    at_start = bool(sp and max(off_start.values()) <= 5)
    if max(off.values()) > a.ready_tol_deg and not (a.allow_start_pose and at_start):
        if a.go: raise SystemExit(f"두 팔이 READY 가 아님 (최대 관절 차이 {off}){' / 시작 자세 차이 ' + str(off_start) if sp else ''} -- 먼저 READY (또는 `start --go` 후 --allow-start-pose)")
        say(f"주의: READY 가 아님 (관절 차이 {off} deg) -- dry-run 은 계속")
    if at_start: say(f"시작 자세 (start_pose.json, 학습 에피소드 {sp['episode']}) 에서 시작")
    prev = Preview(arms, col, WS, a, ra.table_z); prev.start()
    st_w = core.build_state(*(core.fk(arms, s, q0[s]) for s in core.SIDES), *(core.fk(arms, s, q0[s]) for s in core.SIDES), 0.0, 0.0)
    im_w = {"global": gl.ring.latest()[1][0], "left_wrist": wr["left"].ring.latest()[1], "right_wrist": wr["right"].ring.latest()[1]}
    for _ in range(3): runner.predict(runner.item(im_w, st_w, None))         # MPS warm-up (first call ~1.4 s DiT / 0.2 s ACT); nothing is used
    # ---- start state
    T0 = {s: core.fk(arms, s, q0[s]) for s in core.SIDES}; anchor = {s: T0[s].copy() for s in core.SIDES}
    g_cmd = {s: float(a.init_grip if a.init_grip >= 0 else (round(g_obs0[s]) if g_obs0[s] is not None else 0.0)) for s in core.SIDES}
    q_acc = {s: q0[s].copy() for s in core.SIDES}; streamer = None; brl = None
    L = collections.defaultdict(list); ens = []
    stop_reason = None; n_viol = 0; cyc = 0; lat_ema = None; prev_Tc = None; warn_slow = False; n_resync = 0; n_recover = 0
    try:
        if a.go:
            check_grip_limits(rig.url)
            brl = HttpBridge(rig.url, live=True, timeout_s=max(rig.c.robot["request_timeout_s"], 0.3))
            say("두 팔 토크 켬 (unpark), 3초 뒤 시작. 멈춤 = 스페이스")
            for s in core.SIDES: unpark(brl, s)
            t_u = time.monotonic()
            while time.monotonic() - t_u < 3.0:
                r_ = ctl.poll()
                if r_: raise SystemExit(f"시작 전 정지: {r_}")
                time.sleep(0.05)
            q0, _, est = rob.read()
            if est: raise SystemExit("unpark 후 E-STOP/읽기 지연")
            q_acc = {s: q0[s].copy() for s in core.SIDES}; streamer = Streamer(brl, q0, g_cmd, a.stream_hz, dt=dt_row); streamer.start()
        t_start = time.monotonic()
        if a.policy == "label": runner.cursor = 0.0; runner.t0 = t_start; (run_dir / "t_start_monotonic_ns.txt").write_text(str(int(t_start * 1e9)))
        say("시작" + ("" if a.go else " (DRY-RUN: 아무것도 안 보냄)"))
        while True:
            tc0 = time.monotonic(); el = tc0 - t_start
            if a.secs > 0 and el > a.secs: stop_reason = f"max time {a.secs:.0f}s"; break     # 10-23 user: secs 0 = no time limit
            r_ = ctl.poll()
            if r_: stop_reason = r_; break
            if streamer is not None and streamer.error: stop_reason = streamer.error; break
            if rob.error and not a.assume_ready: stop_reason = f"robot read: {rob.error}"; break
            # ---- observe (latest of every ring; DiT: + the samples nearest t_obs - 50 ms)
            t_obs = time.monotonic_ns(); tq, (qm, gobs, est) = rob.ring.latest()
            if (est or qm is None) and a.go and a.keep_going:
                # 10-23 user: an arm drive dropping out (left wrist) must not end the run -> wait for the bridge to re-enable it,
                # clear the latch, re-seed both arms on their measured pose and continue
                n_recover += 1; say(f"[{el:5.1f}s] 브리지 E-STOP/읽기 지연 -> 복구 대기 ({n_recover})")
                time.sleep(1.0)
                try:
                    if brl.status().get("estop"): brl.s.post(brl.base_url + "/clear_estop", timeout=2.0)
                    for s in core.SIDES: unpark(brl, s)
                except Exception as e: say(f"복구 실패, 재시도: {type(e).__name__}")
                _, (qr_, _, est_) = rob.ring.latest()
                if qr_ is not None and not est_:
                    q_acc = {s: qr_[s].copy() for s in core.SIDES}; streamer.set([dict(q_acc)], core.DT, g_cmd); say("복구됨 -> 측정 자세에서 계속")
                cyc += 1; continue
            if est or qm is None: stop_reason = "bridge E-STOP / stale read"; break
            tg_, (g_img, g_jpg) = gl.ring.latest(); w_ = {s: wr[s].ring.latest() for s in core.SIDES}
            img_age = [(t_obs - tg_) / 1e6] + [(t_obs - w_[s][0]) / 1e6 for s in core.SIDES]
            stale_img = max(img_age) > a.max_img_age_ms and not a.replay_global and a.policy != "label"     # 10-27: label replay uses no images
            if stale_img and a.go and not a.keep_going: stop_reason = f"stale image {np.round(img_age)} ms"; break
            if stale_img and a.go: time.sleep(0.05); cyc += 1; continue          # keep_going: skip this plan, hold, re-observe
            imgs = {"global": g_img, "left_wrist": w_["left"][1], "right_wrist": w_["right"][1]}
            Tm = {s: core.fk(arms, s, qm[s]) for s in core.SIDES}
            st = core.build_state(Tm["left"], Tm["right"], anchor["left"], anchor["right"], g_cmd["left"], g_cmd["right"])
            prev_obs, h_used = None, False
            if dit:
                th = t_obs - int(core.DT * 1e9); hq = rob.ring.near(th, HIST_TOL_S); hg = gl.ring.near(th, HIST_TOL_S + 0.04)
                hs = None
                if hq is not None and hq[1][0] is not None:       # state at t - 50 ms from the 25 Hz robot ring (same anchor / jaw command)
                    Th = {s: core.fk(arms, s, hq[1][0][s]) for s in core.SIDES}
                    hs = (hq[0], core.build_state(Th["left"], Th["right"], anchor["left"], anchor["right"], g_cmd["left"], g_cmd["right"]))
                hw = {s: wr[s].ring.near(th, HIST_TOL_S) for s in core.SIDES}
                if hs is not None and hg is not None and all(v is not None for v in hw.values()):
                    prev_obs = ({"global": hg[1][0], "left_wrist": hw["left"][1], "right_wrist": hw["right"][1]}, hs[1]); h_used = True
            t1 = time.monotonic()
            # ---- policy
            rtc = None
            if rtc_mode != "off":
                i_r, q_r = streamer.anchor(t_obs / 1e9)
                with streamer.lock: Qp = [streamer._q_at(i_r + k) for k in range(1, 33) if i_r + k <= streamer.last_idx]
                if Qp:
                    Ta = {s: core.inv_T(core.fk(arms, s, q_r[s])) for s in core.SIDES}; Yr = np.zeros((32, 20))
                    for k in range(32):
                        q_ = Qp[min(k, len(Qp) - 1)]; gk = rtc_g.get(i_r + 1 + k, g_cmd)
                        for s in core.SIDES: Yr[k, core.SL[s]] = core.T_to_pose9(Ta[s] @ core.fk(arms, s, q_[s])); Yr[k, core.GI[s]] = gk[s]
                    d_r = int(np.ceil(max(lat_hist) / dt_row)) + 1 if lat_hist else 2
                    rtc = dict(Y=Yr, W=core.rtc_weights(32, d_r, len(Qp)), g0=np.array([g_cmd["left"], g_cmd["right"]], np.float64), mode=rtc_mode, beta=getattr(a, "rtc_beta", 5.0))
            pkw = dict(rtc=rtc) if rtc is not None else {}
            cn, ch = runner.predict(runner.item(imgs, st, prev_obs), seed=(0 if a.dit_fixed_noise else cyc), **pkw); t2 = time.monotonic()
            if rtc is not None:      # how well the new chunk continues the previous plan (W = 1 rows) and the overlap (0 < W < 1), mm
                hard, soft = rtc["W"] >= 1.0, (rtc["W"] > 0) & (rtc["W"] < 1.0)
                e_ = np.array([[np.linalg.norm(ch[k, core.SL[s]][:3] - rtc["Y"][k, core.SL[s]][:3]) * 1e3 for s in core.SIDES] for k in range(32)]).mean(1)
                rtc_log.append(dict(t=round(el, 2), d=int(hard.sum()), n_prev=int((rtc["W"] > 0).sum()), hard_mm=round(float(e_[hard].mean()), 1) if hard.any() else None,
                                    soft_mm=round(float(e_[soft].mean()), 1) if soft.any() else None, pol_ms=round((t2 - t1) * 1e3)))   # 10-23: fixed noise = less DiT jitter
            if dit and a.speed != 1.0 and not look: ch = core.retime(ch, core.DT, a.speed)   # 10-29: DiT 20 Hz rows slowed along the path (X-VLA: in the runner; look-ahead: tick = row / speed)
            flags = [f"STOP:stale_image_{max(img_age):.0f}ms"] if stale_img else []
            if not np.isfinite(ch).all(): flags.append("STOP:nan_chunk")
            Tc = core.decode(ch, Tm["left"], Tm["right"])          # absolute targets anchored at T(t_obs)
            if look: i_anc, q_anc = streamer.anchor(t_obs / 1e9)     # tick of the observation + the commanded q there
            prev.submit(el, Tc, qm)
            ws_out = {s: float(np.mean(np.any((Tc[s][:, :3, 3] < WS[s][0]) | (Tc[s][:, :3, 3] > WS[s][1]), axis=1))) for s in core.SIDES}
            if a.temporal_ensemble > 0: ens.append((el, Tc)); ens = [e for e in ens if el - e[0] < 32 * core.DT]
            # ---- re-anchor on the newest measured TCP (latency compensation)
            tq2, (qn, _, est2) = rob.ring.latest()
            if (est2 or qn is None) and a.go and a.keep_going: cyc += 1; continue     # recovered at the top of the next cycle
            if est2 or qn is None: stop_reason = "bridge E-STOP / stale read"; break
            Tn = {s: core.fk(arms, s, qn[s]) for s in core.SIDES}
            if a.policy == "label": Tn = {s: core.fk(arms, s, q_acc[s]) for s in core.SIDES}   # 10-27 label replay: chain on the COMMANDED pose (no tracking-lag loss)
            # 10-29 X-VLA: the arm trails the commanded path by 50-80 mm at segment end -> a plan anchored on the MEASURED TCP starts behind
            # the streamer and every new segment first went backward, then forward again.  Chain on the commanded pose (resync handles drift).
            if retimed and a.go: Tn = {s: core.fk(arms, s, q_acc[s]) for s in core.SIDES}
            if look: Tn = {s: core.fk(arms, s, q_anc[s]) for s in core.SIDES}
            # 10-23 user: the left wrist drives (j4/j5) drop out and the bridge re-enables them in place -> the commanded path (q_acc)
            # no longer matches the arm and the next IK fails.  Re-seed that arm on its measured q, hold it there this cycle, continue.
            if a.go:
                q_ref = streamer.last if (look and streamer.last is not None) else q_acc   # look-ahead: the q being streamed NOW
                rs = [s for s in core.SIDES if float(np.degrees(np.abs(q_ref[s] - qn[s])).max()) > a.resync_deg]
                if rs:
                    n_resync += 1
                    for s in rs: q_acc[s] = qn[s].copy()
                    say(f"[{el:5.1f}s] 측정 자세와 명령이 {a.resync_deg}도 넘게 어긋남 ({', '.join(rs)}) -> 측정 자세로 다시 맞추고 계속 ({n_resync}/{a.max_resync})")
                    if n_resync > a.max_resync and not a.keep_going: stop_reason = f"resync {n_resync} 회 초과"; break
                    streamer.set([dict(q_acc)], core.DT, g_cmd); time.sleep(0.3); cyc += 1; continue
            age_s = (time.monotonic_ns() - t_obs) / 1e9; k0 = int(np.clip(np.floor(age_s / core.DT), 0, 32 - a.commit))
            if a.policy == "label" or look: k0 = 0  # label cursor / look-ahead: rows are keyed by tick, the timeline drops the late ones
            if age_s > a.max_plan_age_s: flags.append(f"HOLD:stale_plan_{age_s * 1e3:.0f}ms")     # never commit a plan this old
            spd = (a.slow_start_speed if el < a.slow_start_s else a.speed) * (0.5 if warn_slow else 1.0)
            if retimed: spd = 0.5 if warn_slow else 1.0   # slowness is already in the retimed chunk (X-VLA runner / DiT retime)
            # 10-23: with long commits the jaw must follow the committed SEGMENT, not chunk row 1 (a 16-step GO never opened: k1 0.2-0.3,
            # later rows 0.7) -> --grip-at mid (default) = the row in the middle of the committed steps; max = open if any committed row opens
            seg = ch[k0:k0 + a.commit]
            gp = {s: float({"first": ch[a.grip_step - 1, core.GI[s]], "mid": seg[len(seg) // 2, core.GI[s]], "last": seg[-1, core.GI[s]],
                            "max": seg[:, core.GI[s]].max()}[a.grip_at]) for s in core.SIDES}
            # 10-30 user "그냥 continuous로 해보자 gripper": --grip-continuous sends the predicted open probability (0..1) as the jaw opening
            # instead of the 0/1 threshold (the left jaw kept stopping half-way while the binary command flipped every cycle)
            # 10-30 user: also a SIGMOID jaw -- 1 / (1 + exp(-gain (p - 0.5))): continuous like the linear mode but sharper around 0.5
            if getattr(a, "grip_mode", "") == "sigmoid":
                kg = float(getattr(a, "grip_gain", 10.0))
                g_new = {s: float(1.0 / (1.0 + np.exp(-kg * (float(gp[s]) - 0.5)))) if s in active else g_cmd[s] for s in core.SIDES}
            elif getattr(a, "grip_mode", "") == "relu":       # 10-30 user: ReLU jaw = clip(gain (p - 0.5), 0, 1): closed below 0.5, ramps open above
                kg = float(getattr(a, "grip_gain", 2.0))
                g_new = {s: float(np.clip(kg * (float(gp[s]) - 0.5), 0.0, 1.0)) if s in active else g_cmd[s] for s in core.SIDES}
            elif getattr(a, "grip_continuous", False):
                g_new = {s: float(np.clip(gp[s], 0.0, 1.0)) if s in active else g_cmd[s] for s in core.SIDES}
            else:
                g_new = {s: (1.0 if gp[s] > a.grip_thr else 0.0) if s in active else g_cmd[s] for s in core.SIDES}
            pr = prev.res
            if pr is not None: flags += [f_ for f_ in pr["flags"] if f_.startswith("HOLD")]
            # ---- committed steps: candidate -> limiter -> IK -> safety
            q_prev = {s: (q_acc[s] if (a.go or a.policy == "label") else qn[s]).copy() for s in core.SIDES}; T_cand, T_lim, Q_ik, Q_acc, ikmm, ikdeg = [], [], [], [], [], []
            if look: q_prev = {s: q_anc[s].copy() for s in core.SIDES}
            for j in range(1, a.commit + 1):
                k = k0 + j
                cand = {s: Tn[s] @ core.inv_T(Tc[s][k0 - 1] if k0 > 0 else Tm[s]) @ Tc[s][k - 1] for s in core.SIDES}   # T_now @ inv(dT_k0) @ dT_k
                if a.temporal_ensemble > 0 and not look:     # look-ahead blends in the timeline instead
                    cand = {s: ensemble(ens, el + k * core.DT, s, a.temporal_ensemble) for s in core.SIDES}
                if hcal is not None:                        # plane follows the row xy; only the not-yet-carried part is added
                    ramp = 1.0 if z_off_done else min(1.0, j / 16.0)
                    for s_ in core.SIDES:
                        z_off[s_] = z_off_m + HC.dz_at(hcal, s_, cand[s_][:2, 3]); cand[s_] = cand[s_].copy(); cand[s_][2, 3] += (z_off[s_] - z_base[s_]) * ramp
                elif not z_off_done:
                    for s_ in core.SIDES: cand[s_] = cand[s_].copy(); cand[s_][2, 3] += z_off[s_] * min(1.0, j / 16.0)
                lim, qj, qa, em, ed = {}, {}, {}, {}, {}
                for s in core.SIDES:
                    if s not in active:                     # dry arm: hold the last commanded joints
                        lim[s] = Tn[s]; qj[s] = q_prev[s]; qa[s] = q_prev[s]; em[s] = ed[s] = 0.0; continue
                    T = Tn[s].copy(); Tt = cand[s]
                    T[:3, 3] = Tn[s][:3, 3] + spd * (Tt[:3, 3] - Tn[s][:3, 3]); T[:3, :3] = core.rot_interp(Tn[s][:3, :3], Tt[:3, :3], spd)
                    dp = T[:3, 3] - Tn[s][:3, 3]; mx = a.max_tcp_mps * dt_row * j
                    if np.linalg.norm(dp) > mx: T[:3, 3] = Tn[s][:3, 3] + dp * mx / np.linalg.norm(dp); flags.append(f"LIM:tcp_speed_{s[0]}")
                    ang = core.rot_angle_deg(Tn[s][:3, :3], T[:3, :3]); mr = a.max_rot_dps * dt_row * j
                    if ang > mr: T[:3, :3] = core.rot_interp(Tn[s][:3, :3], T[:3, :3], mr / ang); flags.append(f"LIM:rot_speed_{s[0]}")
                    lift = 0.0 if getattr(a, "no_floor", False) else arms[s].corner_lift(T, g_new[s])   # 10-29 user "바닥 제한을 없애줘": UI GO sets no_floor
                    if lift > 0: T[2, 3] += lift; flags.append(f"LIM:corner_lift_{s[0]}_{lift * 1e3:.0f}mm")
                    if np.any(T[:3, 3] < WS[s][0]) or np.any(T[:3, 3] > WS[s][1]): flags.append(f"STOP:workspace_{s[0]}")
                    lim[s] = T
                    q, pe, re = arms[s].free.ik(q_prev[s], T); qj[s] = q; em[s], ed[s] = pe * 1e3, re
                    if not np.isfinite(q).all(): flags.append(f"STOP:ik_nan_{s[0]}")
                    # 10-23 user: like the robotized labels (FK of the feasible IK path), go to the closest reachable pose; STOP only if far off
                    if pe > a.max_ik_stop_mm / 1e3: flags.append(f"STOP:ik_{s[0]}_{pe * 1e3:.0f}mm_{re:.0f}deg")
                    elif pe > a.max_ik_mm / 1e3 or re > a.max_ik_deg: flags.append(f"LIM:ik_closest_{s[0]}_{pe * 1e3:.0f}mm_{re:.0f}deg")
                    dq = q - q_prev[s]; lim_r = np.radians(a.max_deg_s) * dt_row; m_ = float(np.abs(dq).max())
                    if m_ > lim_r: dq = dq * lim_r / m_; flags.append(f"LIM:joint_speed_{s[0]}")
                    qa[s] = q_prev[s] + dq
                    if np.any(qa[s] < lo[s]) or np.any(qa[s] > hi[s]): flags.append(f"STOP:joint_limit_{s[0]}")
                    Tq_ = core.fk(arms, s, qa[s]); tz_ = ra.table_z + (HC.dz_at(hcal, s, Tq_[:2, 3]) if hcal is not None else 0.0)   # 11-01 measured table
                    if Tq_[2, 3] < tz_ + a.tcp_table_margin: flags.append(f"STOP:tcp_table_{s[0]}")
                cc = col.check(qa, table_z=ra.table_z, arm_clear=a.arm_clear); gap = core.link_gap(arms, qa)
                if not cc["ok_arm"]: flags.append(f"STOP:arm_collision_{cc['pair']}_{cc['arm_clear_m'] * 1e3:.0f}mm")
                if not cc["ok_table"]: flags.append(f"STOP:link_table_{cc['table_clear_m'] * 1e3:.0f}mm")
                if gap < a.warn_gap: flags.append(f"WARN:link_gap_{gap * 1e3:.0f}mm")
                if j == a.commit and (not z_off_done or hcal is not None):
                    z_off_done = True; z_base = dict(z_off); flags.append(f"INFO:z_offset_L{z_off['left'] * 1e3:+.0f}_R{z_off['right'] * 1e3:+.0f}mm")
                T_cand.append(cand); T_lim.append(lim); Q_ik.append(qj); Q_acc.append(qa); ikmm.append(em); ikdeg.append(ed); q_prev = qa
            warn_slow = any(f_.startswith("WARN:link_gap") for f_ in flags)
            t3 = time.monotonic()
            if getattr(a, "no_safety", False):        # 10-27 user (label replay): collision / table / joint-limit / workspace -> logged only
                flags = [("WARN:" + f_[5:]) if f_.startswith(("STOP:arm_collision", "STOP:link_table", "STOP:joint_limit", "STOP:workspace", "STOP:tcp_table")) else f_ for f_ in flags]
            if len(active) < 2:                             # dry arm's own flags never block the active arm
                dry_s = [x[0] for x in core.SIDES if x not in active]
                flags = [("DRY:" + f_.split(":", 1)[1]) if any(f_.endswith(f"_{c}") or f"_{c}_" in f_ for c in dry_s) and f_.startswith(("STOP", "HOLD")) and not f_.startswith("STOP:arm_collision") else f_ for f_ in flags]   # arm-arm collision always blocks
            stop = [f_ for f_ in flags if f_.startswith(("STOP", "HOLD"))]
            # ---- commit (go) / virtual commit (dry-run)
            if stop:
                n_viol += 1          # keep_going: this plan is NOT sent (the streamer holds the last accepted pose), the run continues
                if step:
                    if take_step(): say(f"[{el:5.1f}s] 스텝 거부 (안전 플래그): {', '.join(stop)}")
                elif a.go and not a.keep_going: stop_reason = "SAFETY " + ", ".join(stop)
            elif a.go and step and not take_step():
                flags.append("WAIT:step")      # step mode: plan shown, nothing sent, the streamer holds the last accepted pose
            elif look:
                nd, nb = streamer.merge(i_anc, [Q_acc[j] for j in range(a.commit)], g_new); n_dropped += nd; n_blend += nb; q_acc = Q_acc[-1]; g_cmd = g_new
                if rtc_mode != "off":       # jaw rows of the committed segment, keyed by tick (the RTC target for the next plan)
                    rtc_g.update({i_anc + 1 + j: {s: float(ch[j, core.GI[s]]) for s in core.SIDES} for j in range(a.commit)})
                    for i_ in [i_ for i_ in rtc_g if i_ < i_anc - 64]: del rtc_g[i_]
            elif a.go:
                streamer.set([Q_acc[j] for j in range(a.commit)], core.DT, g_new); q_acc = Q_acc[-1]; g_cmd = g_new
                if step: say(f"[{el:5.1f}s] 한 스텝 실행: {a.commit} 스텝 ({a.commit * core.DT * 1e3:.0f} ms 분량), 집게 L {g_new['left']:.0f} R {g_new['right']:.0f}")
            elif a.policy == "label": q_acc = Q_acc[-1]      # label dry-run: virtual chaining on the commanded path
            if not a.go and not stop: g_cmd = g_new        # dry-run: the state gripper follows the virtual command
            if a.policy == "label" and not stop:
                runner.cursor = min(runner.dur, runner.cursor + a.commit * core.DT * a.label_rate)
                if runner.done(): stop_reason = "label end"
            t4 = time.monotonic()
            # ---- log
            sm = {f"dchunk_k{k}_mm": (float(np.mean([np.linalg.norm(Tc[s][k - 1][:3, 3] - prev_Tc[s][k - 1][:3, 3]) for s in core.SIDES])) * 1e3 if prev_Tc is not None else np.nan) for k in (1, 8, 16, 32)}
            prev_Tc = Tc
            for k_, v_ in dict(t=el, lat_obs_ms=(t1 - tc0) * 1e3, lat_pol_ms=(t2 - t1) * 1e3, lat_ctrl_ms=(t3 - t2) * 1e3, lat_total_ms=(t4 - tc0) * 1e3,
                               obs_to_commit_ms=(time.monotonic_ns() - t_obs) / 1e6, k0=k0, hist_used=h_used,
                               q_meas=np.r_[qm["left"], qm["right"]], q_now=np.r_[qn["left"], qn["right"]], T_meas=np.stack([Tm["left"], Tm["right"]]), T_now=np.stack([Tn["left"], Tn["right"]]),
                               state=st, chunk_norm=cn, chunk=ch, T_chunk=np.stack([Tc["left"], Tc["right"]]), grip_raw=ch[:, 18:20].copy(),
                               T_cand=np.stack([[x["left"], x["right"]] for x in T_cand]), T_lim=np.stack([[x["left"], x["right"]] for x in T_lim]),
                               q_ik=np.stack([np.r_[x["left"], x["right"]] for x in Q_ik]), q_acc=np.stack([np.r_[x["left"], x["right"]] for x in Q_acc]),
                               ik_mm=[[x["left"], x["right"]] for x in ikmm], ik_deg=[[x["left"], x["right"]] for x in ikdeg],
                               pv_ik_ok=(pr["ik_ok"] if pr else [np.nan, np.nan]), pv_ik_max_mm=(pr["ik_max_mm"] if pr else [np.nan, np.nan]), pv_min_clear=(pr["min_clear"] if pr else np.nan),
                               pv_age_s=(el - pr["t"] if pr else np.nan), g_pred=[gp["left"], gp["right"]], g_cmd=[g_new["left"], g_new["right"]], speed=spd, flags=flags + ([f_ for f_ in pr["flags"] if f_.startswith("WARN")] if pr else []),
                               arm_clear=cc["arm_clear_m"], table_clear=cc["table_clear_m"], link_gap=gap, ws_chunk_out=[ws_out["left"], ws_out["right"]], img_age_ms=img_age,
                               stop=bool(stop), label_tau=(runner.cursor if a.policy == "label" else np.nan), gimg_jpg=g_jpg, wl=imgs["left_wrist"], wr=imgs["right_wrist"], **sm).items(): L[k_].append(v_)
            d1 = {s: np.linalg.norm(Tc[s][0][:3, 3] - Tm[s][:3, 3]) * 1e3 for s in core.SIDES}; d32 = {s: (Tc[s][31][:3, 3] - Tm[s][:3, 3]) * 1e3 for s in core.SIDES}
            if cyc % a.print_every == 0 or stop:
                print(f"[{el:5.1f}s] loop {L['lat_total_ms'][-1]:5.1f} ms (obs {L['lat_obs_ms'][-1]:4.1f} pol {L['lat_pol_ms'][-1]:5.1f} ctrl {L['lat_ctrl_ms'][-1]:4.1f}) k0 {k0} | k1 L {d1['left']:5.1f} R {d1['right']:5.1f} mm | "
                      f"k32 L {np.round(d32['left']).astype(int)} R {np.round(d32['right']).astype(int)} mm | g L {gp['left']:.2f} R {gp['right']:.2f} | "
                      f"preview IK {pr['ik_ok'] if pr else '-'} clr {cc['arm_clear_m'] * 1e3:.0f} mm | {' '.join(f_ for f_ in flags if not f_.startswith('LIM'))}", flush=True)
            ctl.publish(dict(t=el, cyc=cyc, Tm=Tm, Tc=Tc, grip=ch[:, 18:20].copy(), flags=L["flags"][-1], lat_pol_ms=L["lat_pol_ms"][-1],
                             lat_total_ms=L["lat_total_ms"][-1], hz=cyc / max(el, 1e-3), g_cmd=g_new, speed=spd, stop=stop, go=a.go, k0=k0))
            cyc += 1
            if stop and a.go and not a.keep_going and not step: break
            if stop_reason == "label end": break
            # ---- pacing: next plan so that it is ready when the committed segment ends (ACT: 20 Hz)
            lat_ema = (t4 - tc0) if lat_ema is None else 0.8 * lat_ema + 0.2 * (t4 - tc0)
            t_next = tc0 + a.commit * core.DT - (lat_ema if dit else 0.0)
            if look:      # next forward when the queued ticks only cover the slowest recent plan (+1 tick), like indy7 ChunkLookahead
                # re-plan when the queued ticks only cover the slowest recent plan + 1 tick.  10-29: a fixed half-chunk floor (LeRobot async
                # chunk_size_threshold 0.5) kept only the first half of every chunk: DiT never got past its dip, X-VLA at speed 0.2 (333 ms
                # rows, plan ~3 ticks) hovered while the preview's chunk END went to the T-shirt.  At speed 1 X-VLA still re-plans at ~16/32.
                lat_hist.append(t4 - tc0); need = int(np.ceil(max(lat_hist) / dt_row)) + 1
                t_next = streamer.t_base + (streamer.last_idx - need) * dt_row
            time.sleep(max(0.0, t_next - time.monotonic()))
    except KeyboardInterrupt: stop_reason = "Ctrl+C"
    except SystemExit as e: stop_reason = str(e)
    except Exception as e: stop_reason = f"exception {type(e).__name__}: {e}"; import traceback; traceback.print_exc()
    finally:
        if a.go:
            # 11-01 user "estop 으로 계속 영점이 틀어져, 보통 스톱은 없어?": every end used to latch an E-STOP (instant hold at the measured
            # pose = a velocity step).  An operator stop / time limit / label end now freezes the stream at the pose it is commanding
            # NOW, the bridge's acceleration-limited profile brakes onto it, and nothing is latched.  Safety stops, bridge errors and
            # exceptions still E-STOP.
            sr = str(stop_reason or "")
            soft = sr.startswith(("정지", "max time", "label end", "Ctrl+C")) and streamer is not None and streamer.error is None and streamer.is_alive()
            if soft:
                streamer.set([], streamer.dt, streamer.g); time.sleep(0.8); streamer.stop = True
                if brl is not None: brl.close()
                say(f"부드럽게 정지 (감속 후 현재 명령 자세 유지, E-STOP 아님): {stop_reason}")
            else:
                if streamer is not None: streamer.stop = True
                if brl is not None: brl.estop(); brl.close()
                say(f"정지 + 현재 자세 유지 (E-STOP 홀드): {stop_reason}")
        prev.stop = True
    say(f"끝: {stop_reason}")
    summ = summarize(L, a, stop_reason, n_viol, run_dir, WS, off, cfg, prev, ckpt, runner)
    if look:                                         # 10-29 look-ahead counters (summarize() has no access to the loop state)
        lk = dict(row_dt_ms=round(dt_row * 1e3, 1), rows_dropped=n_dropped, rows_blended=n_blend, dry_stream_s=round(streamer.n_dry / a.stream_hz, 2) if streamer is not None else None,
                  rtc=rtc_mode, rtc_hard_mm_mean=(round(float(np.mean([r["hard_mm"] for r in rtc_log if r["hard_mm"] is not None])), 1) if any(r["hard_mm"] is not None for r in rtc_log) else None),
                  rtc_pol_ms_mean=(round(float(np.mean([r["pol_ms"] for r in rtc_log])), 0) if rtc_log else None), rtc_cycles=rtc_log)
        (run_dir / "lookahead.json").write_text(json.dumps(lk, indent=1)); say(f"look-ahead: {lk}")
        if isinstance(summ, dict): summ["lookahead"] = lk
    if summ is not None: summ["stop_reason"] = stop_reason
    return summ


def start_cmd(a):
    """READY -> training start pose.  Target = the start TCPs (pose_abs at frame 0) of ONE training episode (default: the medoid
    episode, so both arms and T_LR come from the same real start), + --start-lift in z.  Path: Cartesian line READY -> target + 10 cm
    (IK seeded step to step), then straight down; every waypoint checked (IK error, joint limits, capsule arm-arm, link-table, TCP
    above table).  Dry-run prints the plan and writes deploy/start_pose.json; --go streams it at --deg-s (umi_replay_bi.stream_path)
    and then simply stops streaming (the bridge holds the last goal)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from visual_teleop.tools.robot_moves import READY_Q_DEG, unpark
    from visual_teleop import config as vcfg
    from visual_teleop.robot.bridge import HttpBridge
    if getattr(a, "label", ""):                    # 10-27 label replay: the label's own first pose (A' vs Cg test)
        lr = core.LabelRunner(a.label); e = a.label; i0 = 0; P0 = {s: lr.T_at(s, 0.0) for s in core.SIDES}
    else:
        root = core.LDT / "data/lerobot" / (getattr(a, "root", "") or "tagrig_laundry_v1_fused_train"); hf = LeRobotDataset("local/x", root=root, video_backend="pyav").hf_dataset.with_format("numpy")   # 10-28 --root (GOLD-281)
        P = np.asarray(hf["pose_abs"]).astype(float); fi = np.asarray(hf["frame_index"]); ep = np.asarray(hf["episode_index"]); st = np.flatnonzero(fi == 0)
        med = np.median(P[st][:, [0, 1, 2, 9, 10, 11]], 0)
        e = int(ep[st[np.argmin(np.linalg.norm(P[st][:, [0, 1, 2, 9, 10, 11]] - med, axis=1))]]) if a.episode < 0 else a.episode
        i0 = int(np.flatnonzero((ep == e) & (fi == 0))[0]); P0 = {s: core.pose9_to_T(P[i0, core.SL[s]]) for s in core.SIDES}
    arms, ra = core.make_arms(); col = core.Collision(arms)
    qr = {s: np.radians(READY_Q_DEG[s]) for s in core.SIDES}
    if a.go:
        c = vcfg.load(); br = HttpBridge(c.robot["base_url"], live=True, timeout_s=0.5); o = br.observe()
        if o.estop: raise SystemExit("로봇 E-STOP/미연결/읽기 지연")
        if max(np.degrees(np.abs(o.q[s] - qr[s])).max() for s in core.SIDES) > 5: raise SystemExit("두 팔이 READY 가 아님")
    tgt = {}
    hc = None
    if getattr(a, "height_calib", False):          # 11-01 user "xy 도 맞추어줘": land the start TCP where the label frame has it (robot FK = B - shift)
        import height_calib as HC; hc = HC.load()
    for s in core.SIDES:
        T = P0[s].copy(); T[2, 3] += a.start_lift
        if hc is not None and "xy_B_minus_robot" in hc["fit"].get(s, {}): T[:2, 3] -= np.asarray(hc["fit"][s]["xy_B_minus_robot"])
        tgt[s] = T
    if hc is not None: say("시작 자세 xy 보정: " + ", ".join(f"{s} {np.round(np.asarray(hc['fit'][s].get('xy_B_minus_robot', [0, 0])) * -1e3, 1).tolist()} mm" for s in core.SIDES))
    def leg(qa, Ta, Tb, n=40):
        from scipy.spatial.transform import Rotation as Rot, Slerp
        sl = Slerp([0, 1], Rot.from_matrix([Ta[:3, :3], Tb[:3, :3]])); out, q = [], qa.copy()
        for u in np.linspace(0, 1, n):
            T = np.eye(4); T[:3, :3] = sl(u).as_matrix(); T[:3, 3] = (1 - u) * Ta[:3, 3] + u * Tb[:3, 3]; q, pe, re = arms[s_].free.ik(q, T); out.append((q.copy(), pe, re))
        return out
    path = {}; rep = {}
    for s_ in core.SIDES:
        Tr = core.fk(arms, s_, qr[s_]); up = tgt[s_].copy(); up[2, 3] += 0.10
        L1 = leg(qr[s_], Tr, up); L2 = leg(L1[-1][0], up, tgt[s_], 20); path[s_] = L1 + L2
        rep[s_] = dict(target_p=np.round(tgt[s_][:3, 3], 3).tolist(), ik_max_mm=round(max(x[1] for x in path[s_]) * 1e3, 1), rot_max_deg=round(max(x[2] for x in path[s_]), 1),
                       limits_ok=bool(all(np.all(x[0] >= arms[s_].lo) and np.all(x[0] <= arms[s_].hi) for x in path[s_])),
                       q_start_deg=np.round(np.degrees(path[s_][-1][0]), 1).tolist())
    n = max(len(v) for v in path.values()); W = [{s: path[s][min(i, len(path[s]) - 1)][0] for s in core.SIDES} for i in range(n)]
    chk = [col.check(w, table_z=ra.table_z) for w in W]; tcpz = min(core.fk(arms, s, w[s])[2, 3] for w in W for s in core.SIDES)
    jdist = max(float(np.degrees(np.abs(W[i + 1][s] - W[i][s])).max()) for i in range(n - 1) for s in core.SIDES)
    ok = all(r["ik_max_mm"] <= 10 and r["limits_ok"] for r in rep.values()) and min(c_["arm_clear_m"] for c_ in chk) >= 0.01 and min(c_["table_clear_m"] for c_ in chk) >= 0 and tcpz >= ra.table_z + 0.005
    lengths = sum(max(float(np.degrees(np.abs(W[i + 1][s] - W[i][s])).max()) for s in core.SIDES) for i in range(n - 1))
    out = dict(episode=e, frame0_index=i0, start_lift_m=a.start_lift, arms=rep, min_arm_clear_mm=round(min(c_["arm_clear_m"] for c_ in chk) * 1e3), min_table_clear_mm=round(min(c_["table_clear_m"] for c_ in chk) * 1e3),
               min_tcp_z_mm=round(tcpz * 1e3), max_joint_step_deg=round(jdist, 1), duration_s=round(lengths / a.deg_s, 1), ok=bool(ok),
               q={s: path[s][-1][0].tolist() for s in core.SIDES}, T={s: tgt[s].tolist() for s in core.SIDES}, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(json.dumps({k: v for k, v in out.items() if k not in ("q", "T")}, indent=1, ensure_ascii=False))
    if not ok: raise SystemExit("시작 자세 경로 불가 -- 이동 안 함")
    START_FILE.write_text(json.dumps(out, indent=1)); print("저장", START_FILE)
    if not a.go: print("DRY: 이동 안 함. 움직이려면 사람이 옆에서: start --go"); return
    from visual_teleop.tools.umi_replay_bi import stream_path
    for s in core.SIDES: unpark(br, s)
    g = {s: o.g(s) for s in core.SIDES}; say("학습 시작 자세로 천천히 이동"); stream_path(br, [{s: br.observe().q[s] for s in core.SIDES}] + W, g, deg_s=a.deg_s)
    say("시작 자세 도착 (브리지가 마지막 목표를 유지)"); br.close()


def ensemble(ens, t_target, s, m):
    """ACT temporal ensemble on ABSOLUTE base-frame targets: every buffered chunk's step nearest t_target, weights exp(-m * i), i = 0 oldest"""
    from scipy.spatial.transform import Rotation as Rot
    P, R, W = [], [], []
    for i, (tp, Tc) in enumerate(ens):
        k = int(round((t_target - tp) / core.DT)) - 1
        if 0 <= k < 32: P.append(Tc[s][k][:3, 3]); R.append(Tc[s][k][:3, :3]); W.append(np.exp(-m * len(W)))
    W = np.array(W) / np.sum(W); T = np.eye(4); T[:3, 3] = np.sum(np.array(P) * W[:, None], 0); T[:3, :3] = Rot.from_matrix(np.array(R)).mean(weights=W).as_matrix(); return T


def go_ready(br, arms, qr):
    """explicit user request only: /clear_estop, link-gap check along the joint line, umi_replay_bi.ramp_both at 10 deg/s"""
    from visual_teleop.tools.umi_replay_bi import ramp_both
    o = br.observe(); q = o.q
    for u in np.linspace(0, 1, 30):
        qq = {s: q[s] + u * (qr[s] - q[s]) for s in core.SIDES}
        if core.link_gap(arms, qq) < 0.10: print("READY 경로에서 두 팔이 가까움 (< 100 mm) -- 손으로 옮기세요. 이동 안 함", flush=True); return
    r = br.s.post(br.base_url + "/clear_estop", timeout=2.0).json()
    if not r.get("ok", True): print("clear_estop 실패:", r); return
    g = {s: o.g(s) for s in core.SIDES}; say("READY 로 천천히 복귀"); ramp_both(br, q, qr, g, deg_s=10.0); br.estop(); say("READY 도착, 홀드")


# ---------------------------------------------------------------- visualisation
COL = {"left": (255, 140, 0), "right": (0, 140, 255)}      # BGR: left = blue, right = orange (tagrig_qa convention)


def topdown(Tm, Tc, WS, train_pts, size=480):
    """base frame from above: x forward = up, +y (robot left) = left"""
    img = np.full((size, size, 3), 255, np.uint8); x0, x1, y0, y1 = -0.1, 0.8, -0.65, 0.65
    px = lambda p: (int((y1 - p[1]) / (y1 - y0) * size), int((x1 - p[0]) / (x1 - x0) * size))
    for p in train_pts: cv2.circle(img, px(p), 1, (215, 215, 215), -1)
    for s in core.SIDES:
        a_, b_ = WS[s]; cv2.rectangle(img, px((b_[0], b_[1])), px((a_[0], a_[1])), COL[s], 1)
        pts = [px(T[:3, 3]) for T in Tc[s]]
        for i in range(len(pts) - 1): cv2.line(img, pts[i], pts[i + 1], COL[s], 2)
        cv2.circle(img, px(Tm[s][:3, 3]), 7, (0, 0, 0), 2); cv2.circle(img, pts[0], 4, COL[s], -1); cv2.circle(img, pts[-1], 6, COL[s], 2)
    cv2.circle(img, px((0, 0.28)), 5, (0, 0, 0), -1); cv2.circle(img, px((0, -0.28)), 5, (0, 0, 0), -1)
    cv2.putText(img, "top view: up = +x fwd, left = +y (robot L)", (8, 18), 0, 0.45, (0, 0, 0), 1); cv2.putText(img, "L", px((0.02, 0.33)), 0, 0.6, COL["left"], 2); cv2.putText(img, "R", px((0.02, -0.25)), 0, 0.6, COL["right"], 2)
    return img


def summarize(L, a, stop_reason, n_viol, run_dir, WS, off, cfg, prev, ckpt, runner):
    n = len(L["t"])
    if n == 0: print("기록 없음"); return {}
    big = ("flags", "gimg_jpg", "wl", "wr")
    arr = {k: np.asarray(v) for k, v in L.items() if k not in big}
    np.savez_compressed(run_dir / "log.npz", **arr, flags=np.array([";".join(f) for f in L["flags"]]), wl=np.stack(L["wl"]), wr=np.stack(L["wr"]),
                        gimg_jpg=np.array(L["gimg_jpg"], dtype=object), preview=np.array([json.dumps(r) for r in prev.log]))
    Tm, Tc = arr["T_meas"], arr["T_chunk"]
    d1 = np.linalg.norm(Tc[:, :, 0, :3, 3] - Tm[:, :, :3, 3], axis=2) * 1e3; d32 = (Tc[:, :, 31, :3, 3] - Tm[:, :, :3, 3]) * 1e3
    cnt = {}
    for fs in L["flags"]:
        for f_ in fs:
            if ":" not in f_: cnt[f_] = cnt.get(f_, 0) + 1; continue      # 10-30: a colon-less flag crashed the summary
            h, b = f_.split(":", 1); k = h + ":" + "_".join(b.split("_")[:2]); cnt[k] = cnt.get(k, 0) + 1
    q = lambda x: dict(p50=round(float(np.nanmedian(x)), 1), p90=round(float(np.nanpercentile(x, 90)), 1), max=round(float(np.nanmax(x)), 1)) if np.isfinite(np.asarray(x, float)).any() else None
    G = arr["grip_raw"]                       # (n, 32, 2) continuous gripper outputs before the threshold
    thirds = np.array_split(np.arange(n), 3)
    grip = {s: dict(all_steps=dict(zip(("p01", "p10", "p50", "p90", "p99"), np.round(np.percentile(G[:, :, j], [1, 10, 50, 90, 99]), 3).tolist())),
                    k1_by_third=[round(float(np.mean(G[ix, 0, j])), 3) for ix in thirds], k32_by_third=[round(float(np.mean(G[ix, 31, j])), 3) for ix in thirds],
                    open_frac_k1=round(float(np.mean(G[:, 0, j] > 0.5)), 3), open_frac_any_step=round(float(np.mean((G[:, :, j] > 0.5).any(1))), 3),
                    out_of_0_1=round(float(np.mean((G[:, :, j] < -0.05) | (G[:, :, j] > 1.05))), 3)) for j, s in enumerate(core.SIDES)}
    pv = prev.log
    S = dict(run=str(run_dir), policy=a.policy, ckpt=ckpt, dit_sampler=("DDIM 10 x%d" % a.dit_samples if a.policy == "dit" else None), go=a.go, commit=a.commit,
             cycles=n, secs=round(float(L["t"][-1]), 1), plan_hz=round(n / max(L["t"][-1], 1e-6), 1), stop_reason=stop_reason, violation_cycles=n_viol,
             assume_ready=a.assume_ready, global_src=("REPLAY " + a.replay_global) if a.replay_global else a.global_url, ready_offset_deg=off,
             cams={s: cfg[s] for s in core.SIDES}, wave_test=cfg.get("wave_test"),
             latency_ms=dict(loop=q(arr["lat_total_ms"]), obs=q(arr["lat_obs_ms"]), policy=q(arr["lat_pol_ms"]), ctrl_ik_safety=q(arr["lat_ctrl_ms"]),
                             obs_to_commit=q(arr["obs_to_commit_ms"]), k0_steps_skipped=q(arr["k0"]), preview=q([r["ms"] for r in pv]) if pv else None),
             dit_history_used_frac=round(float(np.mean(arr["hist_used"])), 3) if a.policy == "dit" else None,
             k1_disp_mm=dict(left=q(d1[:, 0]), right=q(d1[:, 1])),
             k32_disp_mm_median=dict(left=np.round(np.median(d32[:, 0], 0)).tolist(), right=np.round(np.median(d32[:, 1], 0)).tolist()),
             k32_dz_mm=dict(left=q(d32[:, 0, 2]), right=q(d32[:, 1, 2])), k32_norm_mm=dict(left=q(np.linalg.norm(d32[:, 0], axis=1)), right=q(np.linalg.norm(d32[:, 1], axis=1))),
             chunk_y_median_m=dict(left=round(float(np.median(Tc[:, 0, :, 1, 3])), 3), right=round(float(np.median(Tc[:, 1, :, 1, 3])), 3)),
             left_on_plus_y=bool(np.median(Tc[:, 0, :, 1, 3]) > 0 and np.median(Tc[:, 1, :, 1, 3]) < 0),
             grip_raw=grip, chunk_smoothness_mm={k: q(arr[f"dchunk_{k}_mm"]) for k in ("k1", "k8", "k16", "k32")},
             preview=dict(n=len(pv), ik_ok=np.round(np.nanmean([r["ik_ok"] for r in pv], 0), 3).tolist() if pv else None,
                          ik_max_mm=q([max(r["ik_max_mm"]) for r in pv]) if pv else None, min_clear_mm=q([r["min_clear"] * 1e3 for r in pv]) if pv else None,
                          min_table_mm=q([r["min_table"] * 1e3 for r in pv]) if pv else None),
             ik_commit_mm=q(arr["ik_mm"].ravel()), ws_chunk_out_frac=np.round(arr["ws_chunk_out"].mean(0), 3).tolist(),
             arm_clear_mm=q(arr["arm_clear"] * 1e3), link_gap_mm=q(arr["link_gap"] * 1e3), flag_counts=cnt, img_age_ms=q(arr["img_age_ms"].ravel()),
             workspace={s: [np.round(WS[s][0], 3).tolist(), np.round(WS[s][1], 3).tolist()] for s in core.SIDES})
    (run_dir / "summary.json").write_text(json.dumps(S, indent=1, ensure_ascii=False)); print(json.dumps(S, indent=1, ensure_ascii=False))
    if not a.no_video:
        S["video"] = str(render_video(L, arr, run_dir, WS)); S["grip_plot"] = str(grip_plot(arr, run_dir))
        (run_dir / "summary.json").write_text(json.dumps(S, indent=1, ensure_ascii=False))
    return S


def grip_plot(arr, run_dir):
    """raw gripper outputs over time (pre-threshold, 1 = open): k = 1 / 8 / 32 per arm -> Desktop png"""
    t = arr["t"]; G = arr["grip_raw"]; C = {1: (0, 0, 0), 8: (200, 120, 0), 32: (0, 120, 230)}
    im = [core.cv_plot([(t, G[:, k - 1, j], C[k], f"k={k}") for k in (1, 8, 32)], (-0.2, 1.2), f"{run_dir.name} {s} gripper raw (threshold 0.5)", hline=0.5, xlabel="s")
          for j, s in enumerate(core.SIDES)]
    p = DESKTOP / f"act_{run_dir.name}_grip.png"; cv2.imwrite(str(p), np.vstack(im)); return p


def render_video(L, arr, run_dir, WS):
    T_C_B, K = core.robot_cam()
    tp = core.LDT / "deploy/train_tcp_sample.npy"; train = np.load(tp) if tp.exists() else np.zeros((0, 3))
    path = DESKTOP / f"act_{run_dir.name}.mp4"; tmp = run_dir / "video_raw.mp4"
    W, H = 848 + 480, 480 + 216; vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 10.0 if run_dir.name.startswith("dit") else 20.0, (W, H))
    for i in range(len(L["t"])):
        g = cv2.resize(cv2.imdecode(np.frombuffer(L["gimg_jpg"][i], np.uint8), cv2.IMREAD_COLOR), (848, 480))
        Tm = {s: arr["T_meas"][i][j] for j, s in enumerate(core.SIDES)}; Tc = {s: arr["T_chunk"][i][j] for j, s in enumerate(core.SIDES)}
        for s in core.SIDES:
            uv = core.project(T_C_B, K, np.array([T[:3, 3] for T in Tc[s]]))
            for k in range(31):
                if np.isfinite(uv[k]).all() and np.isfinite(uv[k + 1]).all(): cv2.line(g, tuple(uv[k].astype(int)), tuple(uv[k + 1].astype(int)), COL[s], 2)
            u0 = core.project(T_C_B, K, Tm[s][None, :3, 3])[0]
            if np.isfinite(u0).all(): cv2.circle(g, tuple(u0.astype(int)), 8, (0, 0, 0), 2)
            if np.isfinite(uv[-1]).all(): cv2.circle(g, tuple(uv[-1].astype(int)), 6, COL[s], -1)
        txt = f"t {L['t'][i]:5.1f}s loop {L['lat_total_ms'][i]:4.0f}ms pol {L['lat_pol_ms'][i]:4.0f}ms  gL {arr['g_pred'][i][0]:.2f} gR {arr['g_pred'][i][1]:.2f}  spd {L['speed'][i]:.2f}"
        cv2.rectangle(g, (0, 0), (848, 30), (0, 0, 0), -1); cv2.putText(g, txt, (8, 21), 0, 0.55, (0, 255, 0), 1)
        st = [x for x in L["flags"][i] if not x.startswith("LIM")]
        if st: cv2.putText(g, " ".join(st)[:90], (8, 470), 0, 0.5, (0, 0, 255), 2)
        top = np.hstack([g, topdown(Tm, Tc, WS, train)])
        wl = cv2.cvtColor(L["wl"][i], cv2.COLOR_RGB2BGR); wr_ = cv2.cvtColor(L["wr"][i], cv2.COLOR_RGB2BGR)
        cv2.putText(wl, "L wrist (policy input)", (5, 15), 0, 0.45, (0, 0, 255), 1); cv2.putText(wr_, "R wrist (policy input)", (5, 15), 0, 0.45, (0, 0, 255), 1)
        bot = np.hstack([wl, wr_, np.full((216, W - 768, 3), 40, np.uint8)])
        k1 = [np.linalg.norm(Tc[s][0][:3, 3] - Tm[s][:3, 3]) * 1e3 for s in core.SIDES]; k32 = [(Tc[s][31][:3, 3] - Tm[s][:3, 3]) * 1e3 for s in core.SIDES]
        for r_, t_ in enumerate([f"k1 step  L {k1[0]:.1f} mm  R {k1[1]:.1f} mm", f"k32 L dxyz {np.round(k32[0]).astype(int)} mm", f"k32 R dxyz {np.round(k32[1]).astype(int)} mm",
                                 f"preview IK ok L/R {arr['pv_ik_ok'][i][0]:.2f}/{arr['pv_ik_ok'][i][1]:.2f}", f"arm clear {arr['arm_clear'][i] * 1e3:.0f} mm  gap {arr['link_gap'][i] * 1e3:.0f} mm"]):
            cv2.putText(bot, t_, (778, 30 + 36 * r_), 0, 0.6, (255, 255, 255), 1)
        vw.write(np.vstack([top, bot]))
    vw.release()
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(path)], capture_output=True)
    if r.returncode: import shutil; shutil.copy(tmp, path)
    print("영상", path, flush=True); return path


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    c_ = sub.add_parser("cams", help="USB location -> OpenCV index (brightness probe), Desktop snapshot"); c_.add_argument("--cover", action="store_true"); c_.add_argument("--candidates", default="0,2,3,4,5")
    w_ = sub.add_parser("wave", help="side-by-side wrist streams labelled by USB location; --confirm left=LOC right=LOC pins the mapping")
    w_.add_argument("--port", type=int, default=8761); w_.add_argument("--secs", type=float, default=0); w_.add_argument("--exposure", type=int, default=2)
    w_.add_argument("--confirm", nargs=2, metavar="SIDE=LOC"); w_.add_argument("--by", default="user")
    s_ = sub.add_parser("start", help="plan (dry) / move (--go) READY -> a TRAINING episode start pose (READY is outside the training TCP distribution)")
    s_.add_argument("--go", action="store_true"); s_.add_argument("--episode", type=int, default=-1, help="train episode index (default: medoid start)")
    s_.add_argument("--start-lift", type=float, default=0.03, help="m above the training start TCP (table safety)")
    s_.add_argument("--height-calib", action="store_true", help="11-01: shift the start TCP by the per-arm xy of deploy/height_calib.json"); s_.add_argument("--deg-s", type=float, default=10.0)
    s_.add_argument("--root", default="", help="10-28: training root under data/lerobot for the start pose (default tagrig_laundry_v1_fused_train)"); s_.add_argument("--assume-ready", action="store_true"); s_.add_argument("--label", default="", help="10-27: start at this label's first pose (ik_v1/demoNN.npz) instead of a training episode")
    r = sub.add_parser("run")
    r.add_argument("--policy", choices=["act", "dit", "label"], default="act"); r.add_argument("--label", default="", help="--policy label: robotize_ik ik_v1/demoNN.npz"); r.add_argument("--label-rate", type=float, default=0.2, help="--policy label: label time per wall second"); r.add_argument("--no-safety", action="store_true", help="10-27 user: collision/table/joint-limit/workspace STOPs only logged (label replay)"); r.add_argument("--no-prompt", action="store_true", help="10-27: no r/q prompt after --go (driven from a script; stop = bridge /estop)"); r.add_argument("--ckpt", default=None, help="default: ACT 500k / DiT checkpoints/last")
    r.add_argument("--go", action="store_true", help="명령을 보냄 (사람이 옆에 있을 때만)"); r.add_argument("--secs", type=float, default=60.0)
    r.add_argument("--commit", type=int, default=None, help="ACT 1 (1/2/4), DiT 4 (2/4)"); r.add_argument("--speed", type=float, default=0.5)
    r.add_argument("--slow-start-s", type=float, default=2.0); r.add_argument("--slow-start-speed", type=float, default=0.25)
    r.add_argument("--max-tcp-mps", type=float, default=0.25); r.add_argument("--max-rot-dps", type=float, default=90.0); r.add_argument("--max-deg-s", type=float, default=30.0)
    r.add_argument("--max-ik-mm", type=float, default=10.0); r.add_argument("--max-ik-deg", type=float, default=20.0); r.add_argument("--ws-margin", type=float, default=0.05)
    r.add_argument("--arm-clear", type=float, default=0.010, help="capsule link-link clearance (m) -> STOP"); r.add_argument("--warn-gap", type=float, default=0.12, help="link-origin gap (m) -> WARN + speed x0.5")
    r.add_argument("--preview-hz", type=float, default=5.0); r.add_argument("--preview-stride", type=int, default=4); r.add_argument("--preview-hold", action="store_true", help="future collision/table in the preview -> HOLD")
    r.add_argument("--grip-step", type=int, default=1, help="chunk row (1..32) whose gripper decides open/close"); r.add_argument("--init-grip", type=float, default=0.0, help="state gripper at start (1 open, 0 closed, -1 = observed)")
    r.add_argument("--temporal-ensemble", type=float, default=0.01, help="temporal ensembling of overlapping plans (ACT and DiT; 0 = off; 0.01 = ACT paper)")
    r.add_argument("--dit-samples", type=int, default=1, help="DiT: mean of K batched DDIM samples (1 = offline-eval sampler)")
    r.add_argument("--rtc", default="off", choices=["off", "guided", "blend"], help="10-29 Real-Time Chunking (look-ahead GO only): guided = RTC guidance (DiT), blend = gradient-free (X-VLA always)")
    r.add_argument("--no-lookahead", action="store_true", help="10-30: GO without look-ahead (run the whole commit, then observe + infer)")
    r.add_argument("--z-offset-mm", type=float, default=0.0, help="10-30: raise (+) / lower (-) the whole commanded trajectory once, at the first chunk")
    r.add_argument("--height-calib", action="store_true", help="11-01: add the per-arm table-height error from deploy/height_calib.json (robot-measured tags)")
    r.add_argument("--rtc-beta", type=float, default=5.0, help="RTC guidance weight clip (paper: 5)")
    r.add_argument("--dit-fixed-noise", type=int, default=1, help="DiT: 1 = same initial noise every plan (less jitter), 0 = seed per cycle")
    r.add_argument("--grip-at", choices=["first", "mid", "last", "max"], default="mid", help="which committed chunk row sets the jaw")
    r.add_argument("--grip-thr", type=float, default=0.5, help="jaw opens above this")
    r.add_argument("--grip-continuous", action="store_true", help="10-30: jaw opening = predicted P(open) 0..1 (no threshold)")
    r.add_argument("--grip-mode", default="", choices=["", "sigmoid", "relu"], help="10-30: sigmoid = 1/(1+exp(-gain (p-0.5))); relu = clip(gain (p-0.5), 0, 1)")
    r.add_argument("--grip-gain", type=float, default=10.0, help="sigmoid gain (larger = closer to the 0/1 threshold)")
    r.add_argument("--keep-going", type=int, default=1, help="--go: 1 = safety hits skip that plan (hold) instead of ending the run; drive dropouts are recovered (re-enable, clear latch, re-seed) and the run continues. Only Space / q / UI stop / heartbeat end it")
    r.add_argument("--max-ik-stop-mm", type=float, default=80.0, help="IK error above --max-ik-mm only limits (closest reachable pose); above this it STOPs")
    r.add_argument("--tip-clear", type=float, default=0.0, help="finger-corner floor vs the table (m); < 0 = fingers may press into / scrape the table")   # 10-23 user: -0.01 scraped too much
    r.add_argument("--tcp-table-margin", type=float, default=-0.005, help="hard STOP if the TCP goes below table_z + this (m)")
    r.add_argument("--resync-deg", type=float, default=6.0, help="--go: measured vs commanded joint gap that triggers a re-seed + hold (motor dropout)")
    r.add_argument("--max-resync", type=int, default=5, help="--go: stop after this many re-syncs in one run")
    r.add_argument("--stream-hz", type=float, default=30.0); r.add_argument("--exposure", type=int, default=2); r.add_argument("--wrist-align", action="store_true", help="apply act_tagrig_cams.json wrist_align {left: {...}, right: {...}} (copy from wrist_align_proposed, act_tagrig_domain_gap.py)")
    r.add_argument("--global-url", default="http://127.0.0.1:8092", help="orbbec_server (live Orbbec, sudo)")
    r.add_argument("--replay-global", default="", metavar="TAKE_DIR", help="TEST ONLY: recorded color_*.jpg instead of the live Orbbec (refused with --go)")
    r.add_argument("--assume-ready", action="store_true", help="dry-run without a live robot: state = FK(READY)")
    r.add_argument("--assume-start", action="store_true", help="dry-run without a live robot: state = FK(start_pose.json)")
    r.add_argument("--allow-start-pose", action="store_true", help="--go may start from start_pose.json (after `start --go`) instead of READY")
    r.add_argument("--ready-tol-deg", type=float, default=5.0); r.add_argument("--dit-steps", type=int, default=10, help="DiT DDIM steps")
    r.add_argument("--bridge-url", default=None, help="default: visual_teleop config robot.base_url (:8021); mock bridge e.g. http://127.0.0.1:8031")
    r.add_argument("--max-plan-age-s", type=float, default=0.4, help="observation -> commit older than this -> HOLD"); r.add_argument("--max-img-age-ms", type=float, default=250.0); r.add_argument("--print-every", type=int, default=10)
    r.add_argument("--device", default="mps"); r.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    {"cams": cams_cmd, "wave": wave_cmd, "run": run_cmd, "start": start_cmd}[a.cmd](a)


if __name__ == "__main__":
    main()
