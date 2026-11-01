"""Orbbec RGB-D recording for the UMI AprilTag rig (tag36h11 IDs 200-202) -- 2026-10-16, global-anchor plan (user: depth cam + 3 tags,
slow x/y/z + yaw correction of OpenVINS). Needs sudo on macOS (pyorbbecsdk). Spoken Korean protocol, live tag counts every second:
    still   UMI held still with >= 2 tags visible (tag-to-tag agreement / noise floor)
    rotate  slow turns so every PAIR of tags is seen together at several angles (rig bundle: T_rig_tag_i)
    still   end
Saves datasets/tag_rig/<ts>/: color_%05d.jpg (q95), depth.npz (uint16 aligned to colour, metres = value * depth_scale), t_ns.npy (host
monotonic at receive), meta.json (intrinsics fx fy cx cy, depth_scale, protocol with t0). Files are chowned back to $SUDO_USER.
    sudo .venv/bin/python -m visual_teleop.tools.tag_rig_record [--proto rig|pivot] [--hz 15]
10-17: --proto rig = both UMIs (still, then each UMI turned slowly); --proto pivot = jaw tip held on one desk point while the UMI is tilted
in every direction (T_rig_TCP by pivot least squares), right UMI then left UMI. Both XIAO IMUs are logged too (imu_{left,right}.npz:
device_us, host_ns, gyro rad/s, acc m/s^2) for the IMU<->rig rotation and time offset.
10-20: --proto robotcal = free recording of --secs (Ctrl+C ends and still saves) while visual_teleop.tools.robot_cam_poses moves the robot
through its calibration poses; the preview also serves /tags (latest tag ids + pixel sides, JSON) so the mover can check tag 220 per pose.
--proto axes = 6-direction translation strokes per UMI (replay parity, GPT plan step 6): forward/back -> left/right -> up/down relative to
the ROBOT base, orientation and jaw fixed, --reps times, hold 3 s between strokes (scripts/pose/umi_tcp_traj.py + scripts/evaluation/replay_parity.py)."""
from __future__ import annotations
import argparse, json, os, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TAG_IDS = (200, 201, 202, 210, 211, 212, 220, 221, 222, 230, 231, 232, 400, 401)   # 10-21: 400/401 = left/right UMI start anchors on the table   # 10-17: UMI R 200-202, UMI L 210-212, robot R 220-222, robot L 230-232   # 10-23: 220/221 = 30 mm camera-back tags on the UMIs (220 with 210-212, 221 with 200-202)


PREV = {"jpg": None, "tags": {}}
# 10-29 (user: "모터를 사용해서 gripper 정보를 모을 거야"): the HandUMI jaw servos (Feetech STS3215, id 1, torque OFF = passive encoder) are
# logged at ~100 Hz on the host monotonic clock -> gripper_motor_{left,right}.npz (host_ns, raw register, signed position ticks).
# Sides follow configs/handumi/hardware_handumi_v1.yaml and the USB hubs they share with the XIAO IMUs (hand assignment of the XIAO IMUs, data owner 11-01: left hand 63ACF722.., right hand 2073676409..; an earlier
# version of this comment listed them by rig label, not by hand).
# 10-29 19:45: the processing chain (scripts/tag_rig_fuse.py) pairs a tag rig with imu_<rig>.npz (rig "right" = tags 200-202 = the unit that
# carries IMU 63ACF722).  That unit is in the user's LEFT hand, so the IMUs (opened by HAND, config/teleop_rebot.yaml) are saved under
# their RIG name; the live meter keeps hand sides.  Takes 185121..192644 were renamed afterwards (imu_byhand_*.npz = as recorded).
RIG_OF_HAND = {"left": "right", "right": "left"}
MOTOR_CAL = {"left": (-850.0, 200.0), "right": (1830.0, 1340.0)}   # (closed, open) ticks = scripts/jaw_from_motor.py CAL (10-29)
MOTOR_PORTS = {"left": "/dev/cu.usbmodem5B3D0497471", "right": "/dev/cu.usbmodem5B3D0480821"}   # 10-29 15:57 physical test (user moved only the LEFT
# jaw: 5B3D049747 moved, 5B3D048082 did not; the IMU 63ACF722 moved with it) -> swapped vs hardware_handumi_v1.yaml


class MotorLog(threading.Thread):
    def __init__(self, side, port, hz=100.0):
        super().__init__(daemon=True); self.side, self.port, self.hz = side, port, hz; self.rows = []; self.stop_ = False; self.err = None; self.last = None
        from handumi_collector.devices.feetech_bus import FeetechBus
        self.bus = FeetechBus(port); self.bus.open(); pk, pt = self.bus._packet, self.bus._port
        pk.write1ByteTxRx(pt, 1, 40, 0)                      # torque OFF: never drive the jaw while a human holds it (10-29: a torque-on with an
        if pk.read1ByteTxRx(pt, 1, 40)[0] != 0: raise RuntimeError("torque did not switch off")   # out-of-range goal swung one jaw ~48 deg)

    def run(self):
        pk, pt = self.bus._packet, self.bus._port; nxt = time.monotonic()
        while not self.stop_:
            try:
                v, comm, _ = pk.read2ByteTxRx(pt, 1, 56); t = time.monotonic_ns()
                if comm == 0:
                    sv = -(v & 0x7FFF) if v & 0x8000 else v       # bit 15 = sign (sign-magnitude), e.g. 33313 -> -545
                    self.rows.append((t, v, sv)); self.last = sv
            except Exception as e: self.err = str(e)
            nxt += 1.0 / self.hz; time.sleep(max(0.0, nxt - time.monotonic()))
        self.bus.close()

    def save(self, out):
        r = np.array(self.rows, np.int64).reshape(-1, 3)
        np.savez(out / f"gripper_motor_{self.side}.npz", host_ns=r[:, 0], raw=r[:, 1], pos=r[:, 2], port=self.port, servo_id=1,
                 note="Feetech STS3215 present position (addr 56), torque off; pos = sign-magnitude decoded ticks (4096/rev)")
        return len(r)
COL = {200: (0, 220, 0), 210: (255, 0, 255), 220: (0, 220, 255), 230: (0, 220, 255), 400: (255, 128, 0)}
# 10-29 (user: "너무 빨리 움직여서 학습이 잘 안되는 거 같아"): live hand-speed meter. GOLD-281 demos measured median 0.16 m/s, p95 0.93 m/s,
# rotation p95 127 deg/s (scratchpad speed/demo_speed.py) -- far beyond what the reBot tracks. Linear speed = Orbbec depth at the rig tag
# centres (same tag over a 0.2 s window), rotation = XIAO gyro |w| over the last 0.2 s. Green / yellow edges below; above yellow = red.
SIDE_TAGS = {"right": (210, 211, 212, 220), "left": (200, 201, 202, 221)}   # 10-29 take 185255: tag speed vs the (user-verified) IMU sides
# correlated crossed (tags 210-212 x right IMU 0.83, 200-202 x left IMU 0.58; same-side 0.34 / 0.17) -- the 10-22 rig swap; the
# speed_live.npz / [속도] summaries of takes 185121 and 185255 have the LINEAR speeds left<->right swapped (rotation sides are right)
SPEED_LIM = {"lin": (0.25, 0.40), "rot": (60.0, 90.0)}


class SpeedMeter:
    WIN = 0.2

    def __init__(self, intr: dict, depth_scale: float):
        self.K = intr; self.mm = float(depth_scale or 1.0)       # Orbbec depth units are mm * depth_scale
        self.trk = {}; self.val = {sd: {"lin": None, "rot": None, "t": 0.0} for sd in SIDE_TAGS}; self.log = []

    def _xyz(self, c, depth):
        u, v = (int(round(x)) for x in c.reshape(4, 2).mean(0))
        p = depth[max(v - 3, 0):v + 4, max(u - 3, 0):u + 4].astype(np.float64); p = p[p > 0]
        if p.size < 5: return None
        z = float(np.median(p)) * self.mm / 1000.0
        return np.array([(u - self.K["cx"]) * z / self.K["fx"], (v - self.K["cy"]) * z / self.K["fy"], z])

    def update(self, t: float, cs, ids, depth, imu: dict) -> None:
        got = {}
        for c, i in zip(cs if ids is not None else [], ids.ravel() if ids is not None else []):
            x = self._xyz(c, depth); i = int(i)
            if x is None: continue
            h = self.trk.setdefault(i, []); h.append((t, x)); del h[:-12]
            old = [(tt, xx) for tt, xx in h if t - tt >= self.WIN]
            if old and t - old[-1][0] < 0.5:
                sp = float(np.linalg.norm(x - old[-1][1]) / (t - old[-1][0]))
                if sp < 3.0: got.setdefault(i, sp)            # > 3 m/s = depth edge / mis-detection
        for sd, tags in SIDE_TAGS.items():
            v = self.val[sd]; s = [got[i] for i in tags if i in got]
            if s: v["lin"] = float(np.median(s)) if v["lin"] is None else 0.5 * v["lin"] + 0.5 * float(np.median(s)); v["t"] = t
            elif t - v["t"] > 0.5: v["lin"] = None
            smp = [x for x in imu.get(sd, [])[-400:] if x.host_receive_ns >= (t - self.WIN) * 1e9]
            v["rot"] = float(np.degrees(np.mean([np.linalg.norm([x.gx, x.gy, x.gz]) for x in smp]))) if smp else None
            self.log.append((t, 0 if sd == "left" else 1, np.nan if v["lin"] is None else v["lin"], np.nan if v["rot"] is None else v["rot"]))

    @staticmethod
    def level(x, key):
        return -1 if x is None else 0 if x <= SPEED_LIM[key][0] else 1 if x <= SPEED_LIM[key][1] else 2

    def red(self) -> bool:
        return any(max(self.level(v["lin"], "lin"), self.level(v["rot"], "rot")) == 2 for v in self.val.values())

    def draw(self, img):
        h = img.shape[0]; cv2.rectangle(img, (0, h - 76), (img.shape[1], h), (0, 0, 0), -1)
        for j, sd in enumerate(("left", "right")):
            v = self.val[sd]; x0 = 10 + j * img.shape[1] // 2
            for r, (key, txt) in enumerate((("lin", "--" if v["lin"] is None else f"{v['lin']:.2f} m/s"), ("rot", "--" if v["rot"] is None else f"{v['rot']:.0f} deg/s"))):
                col = {-1: (160, 160, 160), 0: (0, 220, 0), 1: (0, 220, 255), 2: (0, 0, 255)}[self.level(v[key], key)]
                cv2.putText(img, f"{sd.upper()[0]} {'move' if key == 'lin' else 'turn'} {txt}", (x0, h - 46 + r * 34), 0, 0.9, col, 2)
        if self.red(): cv2.putText(img, "SLOW DOWN", (img.shape[1] // 2 - 120, h - 90), 0, 1.4, (0, 0, 255), 4)
        return img

    def summary(self, t0: float, t1: float) -> str:
        L = np.array([r for r in self.log if t0 <= r[0] <= t1]).reshape(-1, 4); out = []
        for k, sd in ((0, "왼손"), (1, "오른손")):
            s = L[L[:, 1] == k]; lin = s[:, 2][np.isfinite(s[:, 2])]; rot = s[:, 3][np.isfinite(s[:, 3])]
            if lin.size or rot.size:
                out.append(f"{sd} 이동 상위5% {np.percentile(lin, 95):.2f} m/s, 회전 상위5% {np.percentile(rot, 95):.0f}도/초, 빨강 비율 "
                           f"{np.mean((lin > SPEED_LIM['lin'][1])) if lin.size else 0:.0%}/{np.mean((rot > SPEED_LIM['rot'][1])) if rot.size else 0:.0%}"
                           if lin.size and rot.size else f"{sd} 데이터 부족")
        return " | ".join(out)

    def save(self, out: Path) -> None:
        L = np.array(self.log).reshape(-1, 4)
        np.savez(out / "speed_live.npz", t_mono=L[:, 0], side=L[:, 1].astype(np.int8), lin_mps=L[:, 2], rot_dps=L[:, 3], side_names=np.array(["left", "right"]),
                 lim_lin=np.array(SPEED_LIM["lin"]), lim_rot=np.array(SPEED_LIM["rot"]), note="live meter: Orbbec depth at rig tags (0.2 s window, EMA) + gyro |w|")


def start_preview(port: int) -> None:
    """10-17 (user: "녹화과정도 보고 싶어"): MJPEG of the Orbbec colour with tag overlays + phase/timer, http://localhost:<port>/"""
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/tags"):           # 10-20: for robot_cam_poses (tag 220 visible at this pose?)
                b = json.dumps(PREV["tags"]).encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b); return
            if self.path.startswith("/stream"):
                self.send_response(200); self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f"); self.end_headers()
                try:
                    while True:
                        if PREV["jpg"]: self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + PREV["jpg"] + b"\r\n")
                        time.sleep(0.07)
                except (BrokenPipeError, ConnectionResetError): return
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers()
            ws = "".join(f"<div style='display:inline-block;width:49.5%;color:#fff;font:bold 18px sans-serif;text-align:center'>{'왼손 LEFT' if h == 'left' else '오른손 RIGHT'}"
                         f"<img src='http://localhost:{p}/stream' style='width:100%'></div>" for h, p in PREV.get("wrist_ports", []))      # 10-20: wrist previews; 10-29 user: left hand on the LEFT, labelled
            self.wfile.write(("<html><body style='margin:0;background:#111'><img src='/stream' style='width:100%'>" + ws + "</body></html>").encode())
        def log_message(self, *a): pass
    threading.Thread(target=ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever, daemon=True).start()
    print(f"[태그] 미리보기 http://localhost:{port}/", flush=True)


def annotate(img, cs, ids, title: str):
    v = img.copy()
    for c, i in zip(cs if ids is not None else [], ids.ravel() if ids is not None else []):
        i = int(i); px = c.reshape(4, 2); side = float(np.mean(np.linalg.norm(px - np.roll(px, 1, 0), axis=1)))
        if i not in TAG_IDS: continue
        cv2.polylines(v, [px.astype(np.int32)], True, COL.get(i // 10 * 10, (255, 255, 255)), 3)
        org = tuple(int(x) for x in px[0] + [0, -8]); cv2.putText(v, f"{i} {side:.0f}px", org, 0, 0.8, (0, 0, 0), 5)
        cv2.putText(v, f"{i} {side:.0f}px", org, 0, 0.8, (0, 0, 255) if side < 30 else (255, 255, 255), 2)
    cv2.rectangle(v, (0, 0), (v.shape[1], 44), (0, 0, 0), -1); cv2.putText(v, title, (10, 31), 0, 0.85, (0, 255, 0), 2)
    return v


def say(t: str) -> None:
    print("[태그]", t, flush=True)
    try: subprocess.Popen(["say", "-v", "Yuna", t], stdin=subprocess.DEVNULL)
    except Exception: pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--still-s", type=float, default=20.0); ap.add_argument("--rotate-s", type=float, default=40.0); ap.add_argument("--hz", type=float, default=15.0)
    ap.add_argument("--proto", default="rig", choices=["rig", "pivot", "view", "corners", "robotcal", "axes", "rot", "free", "grip", "align", "task"]); ap.add_argument("--hand", default="both", choices=["right", "left", "both"]); ap.add_argument("--port", type=int, default=8747)
    ap.add_argument("--wrist", action="store_true", help="UMI 손목 어안 영상(30 Hz)도 같이 녹화 (umi_jaw record 를 사용자 권한으로 띄움)"); ap.add_argument("--secs", type=float, default=900.0); ap.add_argument("--reps", type=int, default=3); ap.add_argument("--stroke-cm", type=int, default=15); ap.add_argument("--axes", default="XYZ", help="axes 프로토콜에서 할 축만 (예: X)")
    ap.add_argument("--task-s", type=float, default=0.0, help="시연 길이 (초). 0 = 제한 없음: 스페이스를 눌러야 다음으로 넘어감 (10-29 기본)"); ap.add_argument("--task-name", default="수건 반 접기")
    ap.add_argument("--rot-axes", default="YAW,PITCH,ROLL", help="rot 프로토콜에서 할 회전만"); ap.add_argument("--rot-deg", type=int, default=20)
    ap.add_argument("--no-motor", action="store_true", help="집게 모터(Feetech) 기록 안 함"); ap.add_argument("--no-speed-voice", action="store_true", help="너무 빠를 때 '천천히' 음성 안내 끄기 (화면 표시는 유지)"); ap.add_argument("--width", type=int, default=1280); ap.add_argument("--height", type=int, default=720); a = ap.parse_args()
    from ego_collector.camera.capture import CaptureConfig
    from ego_collector.camera.orbbec_capture import OrbbecCapture
    cap = OrbbecCapture(CaptureConfig(index=0, width=a.width, height=a.height, fps=30)); cap.open()
    out = ROOT / "datasets/tag_rig" / time.strftime("%Y%m%d_%H%M%S"); out.mkdir(parents=True, exist_ok=True)
    meter = SpeedMeter(cap.intrinsics, cap.depth_scale); slow_t, slow_said = None, 0.0
    start_preview(a.port)
    if a.proto == "view":                      # 10-17: preview only (aim the Orbbec / tags), nothing saved, Ctrl+C to stop
        out.rmdir(); det_v = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)); t0v = time.monotonic()
        print("[태그] 미리보기만 (저장 안 함). 끄려면 Ctrl+C", flush=True)
        try:
            while True:
                f = cap.get(timeout=0.5)
                if f is None: continue
                cs, ids, _ = det_v.detectMarkers(cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY)); vis = sorted(int(i) for i in ids.ravel() if int(i) in TAG_IDS) if ids is not None else []
                PREV["jpg"] = cv2.imencode(".jpg", cv2.resize(annotate(f.image, cs, ids, f"VIEW {time.monotonic() - t0v:5.0f}s  tags {vis}"), (1280, 720)), [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
        except KeyboardInterrupt: pass
        cap.close(); return
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
    if a.proto == "rig":
        proto = [(5.0, "양손 유미를 오르벡 카메라 쪽으로 들고, 태그가 두 개 이상 보이게 맞추세요"), (a.still_s, "그대로 완전히 가만히 계세요"),
                 (a.rotate_s, "이제 오른손 유미를 천천히 돌리면서, 태그 두 개가 함께 보이는 여러 각도를 보여 주세요"),
                 (a.rotate_s, "이제 왼손 유미를 천천히 돌리면서, 태그 두 개가 함께 보이는 여러 각도를 보여 주세요"), (5.0, "멈추고 가만히 계세요")]
    elif a.proto == "corners":           # 10-19 baseline accuracy (GPT review): every board corner 4x with different UMI tilts, 3 s hold each
        proto = [(6.0, "모서리 시험입니다. 안내되는 모서리에 집게 끝을 대고 멈추세요")]
        for hand in (["오른손", "왼손"] if a.hand == "both" else ["오른손" if a.hand == "right" else "왼손"]):
            for corner in ("왼쪽 아래", "오른쪽 아래", "오른쪽 위", "왼쪽 위"):
                for tilt in ("태그가 카메라를 보게", "다시 대고", "살짝 다른 각도로", "다시 대고"):   # 10-20: the Orbbec sees the tags only near this pose (side tilts hid them)
                    proto += [(6.0, f"{hand}, {corner}, {tilt}, 멈추세요"), (1.5, "떼세요")]      # analysis uses the last 1.5 s of each hold
        proto += [(3.0, "끝났습니다")]
    elif a.proto == "robotcal":         # 10-20: robot base <-> Orbbec; the robot moves itself (robot_cam_poses), this only records
        proto = [(a.secs, "로봇 캘리브레이션 녹화를 시작합니다. 끝나면 컨트롤 C")]
    elif a.proto == "axes":             # 10-20 replay parity: 6 directions relative to the ROBOT base, slow (~4 s per stroke), orientation + jaw fixed
        proto = [(6.0, "방향 시험입니다. 집게를 닫은 채, 손목 각도를 고정하고, 로봇 기준 방향으로 움직이세요")]
        for hand in (["오른손", "왼손"] if a.hand == "both" else ["오른손" if a.hand == "right" else "왼손"]):
            for ax, (d1, d2) in (("X", ("앞으로", "뒤로")), ("Y", ("왼쪽으로", "오른쪽으로")), ("Z", ("위로", "아래로"))):
                if ax not in a.axes.upper(): continue
                for r in range(a.reps):
                    for dname in (d1, d2):
                        proto += [(3.0, f"{hand}, {ax}축, {r + 1}회, 멈추세요"), (4.0, f"{hand} {dname} {a.stroke_cm}센티 천천히"), (3.0, "멈추세요"), (4.0, "제자리로 천천히")]
        proto += [(3.0, "멈추세요"), (2.0, "끝났습니다")]
    elif a.proto == "free":             # 10-20 combined translation + rotation (GPT: 20-40 s free motion, end hold 2-3 s), one segment per trial
        proto = [(6.0, "자유 동작 시험입니다. 집게를 닫고, 안내에 따라 움직이면서 손목도 같이 돌리세요")]
        CUES = ("자유: 앞으로 가면서 끝을 아래로", "자유: 옆으로 가면서 비틀기", "자유: 위로 들면서 왼쪽으로 돌리기", "자유: 짧게 다가갔다가 방향 바꿔 물러나기")
        for hand in (["오른손", "왼손"] if a.hand == "both" else ["오른손" if a.hand == "right" else "왼손"]):
            for r in range(a.reps):
                proto += [(3.0, f"{hand}, 자유동작, {r + 1}회, 멈추세요")] + [(7.0, c_) for c_ in CUES] + [(3.0, "멈추세요")]
        proto += [(2.0, "끝났습니다")]
    elif a.proto == "task":
        if a.task_s <= 0:                # 10-29 user: "30초 제한 없애줘, 내가 끝내야지만 다음으로": the demo phase lasts until SPACE (1 h cap)
            a.task_s = 3600.0; print("[태그] 시연 시간 제한 없음: 시연이 끝나면 스페이스바를 누르세요", flush=True)             # 10-20 real-task pilot (GPT: one simple fold, 3 demos): hold -> task (--task-s) -> hold, one segment per demo
        proto = [(6.0, f"작업 시연입니다. {a.task_name}. 시작 자세에서 기다리세요")]
        for r in range(a.reps):
            proto += [(3.0, f"{'양손' if a.hand == 'both' else ('오른손' if a.hand == 'right' else '왼손')}, 자유동작, {r + 1}회, 멈추세요"), (a.task_s, f"자유: {a.task_name} 시작"), (3.0, "멈추세요"), (6.0, "다음 시연을 준비하세요")]
        proto += [(2.0, "끝났습니다")]
    elif a.proto == "align":            # 10-20 UMI <-> robot gripper orientation (absolute replay): robot at READY, UMI held parallel to its gripper
        proto = [(8.0, "자세 맞추기입니다. 로봇이 준비 자세에 있는지 확인하세요")]
        for r in range(a.reps + 1):
            proto += [(6.0, f"유미를 로봇 그리퍼 옆에 같은 방향으로 맞춰 대고 멈추세요, {r + 1}회"), (3.0, "떼세요")]
        proto += [(2.0, "끝났습니다")]
    elif a.proto == "grip":             # 10-20 gripper timing (GPT): pick-and-place-like motion; run umi_jaw record beside it for the jaw video
        proto = [(6.0, "집게 시험입니다. 집게를 닫은 채 시작하고, 안내에 맞춰 집게를 열고 닫으세요")]
        CUES = ("자유: 앞으로 다가가면서 집게 열기", "자유: 집게 닫기, 잡으세요", "자유: 들어서 옆으로 옮기기", "자유: 내려놓으며 집게 열기", "자유: 물러나며 집게 닫기")
        for hand in (["오른손", "왼손"] if a.hand == "both" else ["오른손" if a.hand == "right" else "왼손"]):
            for r in range(a.reps):
                proto += [(3.0, f"{hand}, 자유동작, {r + 1}회, 멈추세요")] + [(5.0, c_) for c_ in CUES] + [(3.0, "멈추세요")]
        proto += [(2.0, "끝났습니다")]
    elif a.proto == "rot":              # 10-20 (user: "yaw roll pitch도 맞추기는 해야"): rotations about the jaw TIP, robot-base axes, tip kept in place
        proto = [(6.0, "회전 시험입니다. 집게를 닫고, 집게 끝을 제자리에 둔 채 손목만 돌리세요")]
        R_DIRS = (("YAW", ("왼쪽돌리기", "오른쪽돌리기")), ("PITCH", ("끝위로", "끝아래로")), ("ROLL", ("시계방향비틀기", "반시계방향비틀기")))
        for hand in (["오른손", "왼손"] if a.hand == "both" else ["오른손" if a.hand == "right" else "왼손"]):
            for ax, (d1, d2) in R_DIRS:
                if ax not in a.rot_axes.upper().split(","): continue
                for r in range(a.reps):
                    for dname in (d1, d2):
                        proto += [(3.0, f"{hand}, {ax}축, {r + 1}회, 멈추세요"), (4.0, f"{hand} {dname} {a.rot_deg}도 천천히"), (3.0, "멈추세요"), (4.0, "제자리로 천천히")]
        proto += [(3.0, "멈추세요"), (2.0, "끝났습니다")]
    else:
        proto = [(6.0, "오른손 유미의 집게 끝을 책상의 표시한 점에 대세요"), (30.0, "집게 끝을 떼지 말고, 유미를 앞뒤 좌우로 천천히 기울이세요"),
                 (6.0, "이제 왼손 유미의 집게 끝을 같은 점에 대세요"), (30.0, "집게 끝을 떼지 말고, 유미를 앞뒤 좌우로 천천히 기울이세요"), (4.0, "멈추세요")]
    wrist_ps = []
    if a.wrist:        # 10-20 GPT plan: wrist RGB + IMU + Orbbec on one host monotonic clock; the cameras run as the normal user (camera permission)
        u_, g_ = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
        if u_ and g_: os.chown(out, int(u_), int(g_))
        print("[태그] 손목 카메라 준비 중 (번호를 처음 찾을 때만 ~30초씩, 한 대씩 차례로)", flush=True)
        for hd in (["right", "left"] if a.hand == "both" else [a.hand]):      # 10-20: one at a time -- two simultaneous brightness probes collided
            wd = out / ("wrist" if hd == "right" else "wrist_left"); wd.mkdir()
            if u_ and g_: os.chown(wd, int(u_), int(g_))
            cmd = (["sudo", "-u", os.environ["SUDO_USER"]] if os.environ.get("SUDO_USER") else []) + [str(ROOT / ".venv/bin/python"), "-X", "faulthandler", "-m", "visual_teleop.tools.umi_jaw", "record", "--hand", hd, "--out", str(wd), "--port", str(a.port + (1 if hd == "right" else 2))]
            # 10-30: the child's stderr (tracebacks, faulthandler crash dumps) goes to <wrist dir>/stderr.log -- the left recorder of take
            # 103251 died silently mid-take and nothing was left to explain it
            wp = subprocess.Popen(cmd, cwd=str(ROOT / "src"), stdin=subprocess.DEVNULL, stderr=open(wd / "stderr.log", "w"), start_new_session=True); wrist_ps.append((hd, wd, wp)); t_w = time.monotonic()
            while not (wd / "started").exists():
                if wp.poll() is not None or time.monotonic() - t_w > 90:
                    for _, _, x in wrist_ps: x.terminate()
                    cap.close(); raise SystemExit(f"[태그] {hd} 손목 카메라 녹화가 시작되지 않음")
                time.sleep(0.2)
            print(f"[태그] {hd} 손목 카메라 시작", flush=True)
        print(f"[태그] 손목 카메라 녹화 시작됨: {[h for h, _, _ in wrist_ps]}", flush=True)
        PREV["wrist_ports"] = sorted([(h, a.port + (1 if h == "right" else 2)) for h, _, _ in wrist_ps], key=lambda x: x[0] != "left")   # left first (screen left)
    try: subprocess.Popen((["sudo", "-u", os.environ["SUDO_USER"]] if os.environ.get("SUDO_USER") else []) + ["open", f"http://localhost:{a.port}/"], stdin=subprocess.DEVNULL, start_new_session=True)
    except Exception: pass
    imus = {}
    try:
        from visual_teleop import config as vcfg
        from visual_teleop.tools.calib_right import open_imu
        for side in ("right", "left"):
            try: imus[side] = open_imu(vcfg.load(), side); imus[side].buffer.drain()
            except Exception as e: print(f"[태그] {side} IMU 없음: {e}", flush=True)
    except Exception as e: print(f"[태그] IMU 열기 실패: {e}", flush=True)
    IMU = {sd: [] for sd in imus}
    motors = {}
    if not a.no_motor:
        for sd, prt in MOTOR_PORTS.items():
            try: motors[sd] = MotorLog(sd, prt); motors[sd].start(); print(f"[태그] {sd} 집게 모터 기록 시작 ({prt}, 토크 꺼짐)", flush=True)
            except Exception as e: print(f"[태그] {sd} 집게 모터 없음: {e}", flush=True)
    t_start = time.monotonic(); sched, acc = [], 0.0
    for d, txt in proto: sched.append(dict(t0=acc, dur=d, text=txt)); acc += d
    T, D, k, nxt, last_print, seen = [], [], 0, 0, 0.0, []
    WRIST_WARNED, BAD = {}, {}
    phase = -1; long_take = a.proto in ("robotcal", "axes", "rot", "free", "grip", "task")
    # 10-22 user: SPACE ends the current free/task phase early (child processes get stdin=DEVNULL: a nested `sudo -u` relays the
    # terminal through its own pty and swallowed the keys in the first try) (demo done sooner). stdin in cbreak mode on a TTY; Ctrl+C still works.
    SKIP = {"req": False}; _tty_old = None
    try:       # 10-22: read keys from the controlling terminal (/dev/tty), unbuffered; stdin under sudo's pty relay did not deliver them
        import termios, tty, select, atexit
        _fd = os.open("/dev/tty", os.O_RDONLY | os.O_NONBLOCK); _tty_old = termios.tcgetattr(_fd); tty.setcbreak(_fd)
        atexit.register(lambda: termios.tcsetattr(_fd, termios.TCSADRAIN, _tty_old))
    except Exception as e:
        _fd = None; print(f"[태그] 키 입력을 못 받음 (/dev/tty: {e}) -- 스페이스 기능 없이 진행", flush=True)
    if _fd is not None:
        def _keys():
            while True:
                try:       # a nested sudo restores the terminal to line mode when it exits -> re-assert cbreak every poll
                    if termios.tcgetattr(_fd)[3] & termios.ICANON: tty.setcbreak(_fd)
                except Exception: pass
                r, _, _ = select.select([_fd], [], [], 0.2)
                if not r: continue
                try: data = os.read(_fd, 16)
                except BlockingIOError: continue
                for ch in data.decode(errors="ignore"):
                    if ch == " ": SKIP["req"] = True; print("[태그] 스페이스 입력 받음", flush=True)
                    elif ch.isprintable(): print(f"[태그] 키 입력 {ch!r} (시연 종료는 스페이스)", flush=True)
        threading.Thread(target=_keys, daemon=True).start()
        if a.proto in ("free", "grip", "task"): print("[태그] 시연이 일찍 끝나면 스페이스바 = 이번 시연 종료", flush=True)
    while True:
      try:
        el = time.monotonic() - t_start
        if el >= acc: break
        p = max(i for i, s in enumerate(sched) if el >= s["t0"])
        if SKIP["req"]:
            SKIP["req"] = False
            if sched[p]["text"].startswith("자유:"):     # shorten this phase to now and pull every later phase forward
                cut = sched[p]["t0"] + sched[p]["dur"] - el; sched[p]["dur"] = el - sched[p]["t0"]; sched[p]["ended_early"] = True
                for s_ in sched[p + 1:]: s_["t0"] -= cut
                acc -= cut; print(f"[태그] 스페이스: 시연 {sched[p]['dur']:.1f}s 에서 종료", flush=True)
                continue
        if p != phase:
            if phase >= 0 and sched[phase]["text"].startswith("자유:"):     # 10-29: per-demo speed report when a demo phase ends
                sm = meter.summary(t_start + sched[phase]["t0"], time.monotonic()); print(f"[속도] 이번 시연: {sm}", flush=True); sched[phase]["speed"] = sm
            phase = p; say(sched[p]["text"])
        f = cap.get(timeout=0.5)
        if f is None or getattr(f, "depth", None) is None:
            if k == 0 and el > 6.0:      # 10-16 first run: the Orbbec opened but never delivered a frame for 85 s -> fail fast
                cap.close(); raise SystemExit(f"[태그] 6초 동안 Orbbec 프레임 0장 (읽기 실패 {getattr(cap, 'read_failures', '?')}회). "
                                              "USB 재연결(허브 말고 Mac 본체 포트) 후 다시, 계속되면 Mac 재부팅 (예전 Orbbec 멈춤 증상)")
            continue
        for sd, dv in imus.items(): IMU[sd] += dv.buffer.drain()
        if time.monotonic() < nxt: continue
        nxt = time.monotonic() + 1.0 / a.hz
        img = f.image; cv2.imwrite(str(out / f"color_{k:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if long_take: cv2.imwrite(str(out / f"depth_{k:05d}.png"), np.asarray(f.depth, np.uint16))   # 10-20: 15 min x 15 Hz in RAM = 25 GB
        else: D.append(np.asarray(f.depth, np.uint16))
        T.append(time.monotonic_ns()); k += 1
        cs, ids, _ = det.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)); ids_raw = ids
        meter.update(time.monotonic(), cs, ids_raw, np.asarray(f.depth), IMU)
        if meter.red() and sched[phase]["text"].startswith("자유:"):        # red for > 1 s inside a demo -> spoken cue, at most every 6 s
            slow_t = slow_t or time.monotonic()
            if not a.no_speed_voice and time.monotonic() - slow_t > 1.0 and time.monotonic() - slow_said > 6.0:
                slow_said = time.monotonic(); subprocess.Popen(["say", "-v", "Yuna", "천천히"], stdin=subprocess.DEVNULL)
        else: slow_t = None
        sides = [float(np.mean(np.linalg.norm(c.reshape(4, 2) - np.roll(c.reshape(4, 2), 1, 0), axis=1))) for c, i in zip(cs, ids.ravel()) if int(i) in TAG_IDS] if ids is not None else []
        ids = sorted(int(i) for i in ids.ravel() if int(i) in TAG_IDS) if ids is not None else []
        seen.append(len(ids))
        PREV["tags"] = {"t_ns": T[-1], "k": k - 1, "ids": ids, "side_px": {str(int(i)): round(float(np.mean(np.linalg.norm(c.reshape(4, 2) - np.roll(c.reshape(4, 2), 1, 0), axis=1))), 1)
                                                                        for c, i in zip(cs, ids_raw.ravel()) if int(i) in TAG_IDS} if ids_raw is not None else {}}
        ph_n = (["READY", "STILL", "ROTATE RIGHT UMI", "ROTATE LEFT UMI", "STOP"] if a.proto == "rig" else ["RIGHT TIP ON DOT", "RIGHT: TILT, KEEP TIP", "LEFT TIP ON DOT", "LEFT: TILT, KEEP TIP", "STOP"]
                if a.proto == "pivot" else ["ROBOT CAL (Ctrl+C = stop)"] if a.proto == "robotcal" else [("HOLD " + x["text"].split(",")[1]) if "멈추세요" in x["text"] and "," in x["text"]
              else "MOVE" if "천천히" in x["text"] else "HOLD" for x in sched] if a.proto in ("axes", "rot") else [("FREE " + x["text"][4:]) if x["text"].startswith("자유:") else "HOLD" for x in sched] if a.proto in ("free", "grip", "task") else ["CORNER TOUCH / HOLD" if "멈추세요" in x["text"] else "LIFT" for x in sched])
        title = f"{a.proto.upper()} {phase + 1}/{len(sched)} {ph_n[min(phase, len(ph_n) - 1)]}  {el - sched[phase]['t0']:4.1f}/{sched[phase]['dur']:.0f}s  tags {ids}" + \
                (("  jaw " + " ".join(f"{sd[0].upper()}{m.last}" for sd, m in motors.items())) if motors else "")
        PREV["jpg"] = cv2.imencode(".jpg", cv2.resize(meter.draw(annotate(img, cs, ids_raw, title)), (1280, 720)), [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
        if time.monotonic() - last_print > 1.0:
            # 10-30 take 103251: the left wrist recorder died at 308 s and 11 more demos were recorded without its video -> warn at once
            for hd_, wd_, wp_ in wrist_ps:
                pf = wd_ / "t_ns.partial.npy"; dead = wp_.poll() is not None
                stale = pf.exists() and time.time() - pf.stat().st_mtime > 12
                if dead or stale: BAD[hd_] = "카메라"
                # 10-30 take 121751: the one-shot warning was not heard -> repeat every 15 s while the fault lasts (+ status-line flag)
                if (dead or stale) and time.monotonic() - WRIST_WARNED.get(hd_, -1e9) > 15:
                    WRIST_WARNED[hd_] = time.monotonic(); nm = "왼손" if hd_ == "left" else "오른손"
                    print(f"[태그] !!! {nm} 손목 카메라 {'프로세스 종료 (rc ' + str(wp_.returncode) + ')' if dead else '12초 넘게 프레임 없음'} -- Ctrl+C 로 멈추고 다시 시작하세요. 이후 시연은 {nm} 영상이 없습니다 ({wd_ / 'stderr.log'})", flush=True)
                    say(f"{nm} 카메라 멈춤. 녹화를 멈추세요")
            # 10-30 take 172433: the right servo was re-plugged and its reading moved +2150 ticks -> the jaw label saw no events.
            # scripts/jaw_from_motor.py now re-centres per take, but warn while recording so the user knows (same 15 s repeat).
            for sd_, m_ in motors.items():
                lo_, hi_ = sorted(MOTOR_CAL[sd_]); v_ = m_.last
                if v_ is not None and not (lo_ - 300 <= v_ <= hi_ + 300):
                    BAD["motor_" + sd_] = "모터"
                    if time.monotonic() - WRIST_WARNED.get("motor_" + sd_, -1e9) > 60:
                        WRIST_WARNED["motor_" + sd_] = time.monotonic(); nm = "왼손" if sd_ == "left" else "오른손"
                        print(f"[태그] !!! {nm} 집게 모터 값 {v_} 이 보정 범위 {lo_:.0f}..{hi_:.0f} 밖 (다시 꽂아서 기준점이 바뀜?) -- 녹화는 계속 가능, 레이블은 세션별로 자동 보정", flush=True)
                        say(f"{nm} 집게 모터 기준이 바뀌었어요")
            # 10-30 take 105015: the LEFT-hand XIAO IMU (63ACF722, hub 1) sent 0 bytes for the whole take (it stopped at 307 s of take 103251
            # together with the left wrist camera) -> warn when an opened IMU has no sample for 3 s
            for sd_ in imus:
                last_ = IMU[sd_][-1].host_receive_ns / 1e9 if IMU[sd_] else None
                imu_bad = last_ is None and el > 5 or last_ is not None and time.monotonic() - last_ > 3
                if imu_bad: BAD["imu_" + sd_] = "IMU"
                if imu_bad and time.monotonic() - WRIST_WARNED.get("imu_" + sd_, -1e9) > 15:
                    WRIST_WARNED["imu_" + sd_] = time.monotonic(); nm = "왼손" if sd_ == "left" else "오른손"
                    print(f"[태그] !!! {nm} IMU 샘플 없음 -- Ctrl+C 로 멈추고 {nm} IMU(XIAO) USB 를 뽑았다 꽂은 뒤 다시 시작하세요", flush=True)
                    say(f"{nm} 아이엠유 멈춤. 녹화를 멈추세요")
            last_print = time.monotonic(); print((f"[태그] !!! 장치 이상 {sorted(BAD)} !!! " if BAD else "") + f"[태그] {el:5.1f}s 프레임 {k} | 보이는 태그 {ids} 크기 {[round(x) for x in sides]}px (30px 이상 권장)  (2개 이상 비율 {np.mean(np.array(seen) >= 2):.2f})"
                                                 + (" | 집게 모터 " + " ".join(f"{sd} {m.last} ({len(m.rows)})" for sd, m in motors.items()) if motors else "")
                                                 + " | 속도 " + " ".join(f"{sd} " + ("--" if v["lin"] is None else f"{v['lin']:.2f}m/s") + "/" + ("--" if v["rot"] is None else f"{v['rot']:.0f}도/s") for sd, v in meter.val.items()), flush=True)
      except KeyboardInterrupt:
        print("[태그] Ctrl+C -- 지금까지 저장합니다", flush=True); break
    cap.close(); say("끝났습니다")
    try: meter.save(out)
    except Exception as e: print(f"[속도] 저장 실패: {e}", flush=True)
    for _, _, wp in wrist_ps:
        wp.terminate()
    for _, _, wp in wrist_ps:
        try: wp.wait(timeout=90)   # 10-29: crops.npz of a long take + a slow release; t_ns/meta are written first
        except Exception: wp.kill()
    # 10-17: the pivot take lost meta/t_ns/depth (crash after the frames) -> write the small, essential files FIRST, each step guarded
    np.save(out / "t_ns.npy", np.array(T, np.int64))
    (out / "meta.json").write_text(json.dumps(dict(intrinsics=cap.intrinsics, depth_scale=cap.depth_scale, size=[a.width, a.height], hz=a.hz, tag_ids=TAG_IDS,
                                                    protocol=sched, proto=a.proto, frames=k, t_start_monotonic_ns=int(t_start * 1e9)), ensure_ascii=False, indent=1))
    for sd, m in motors.items():
        try: m.stop_ = True; m.join(timeout=2.0); print(f"[태그] {sd} 집게 모터 샘플 {m.save(out)}" + (f" (오류 {m.err})" if m.err else ""), flush=True)
        except Exception as e: print(f"[태그] {sd} 집게 모터 저장 실패: {e}", flush=True)
    for sd, dv in imus.items():
        try:
            IMU[sd] += dv.buffer.drain(); dv.stop(); dv.close(); S_ = IMU[sd]
            np.savez(out / f"imu_{RIG_OF_HAND[sd]}.npz", device_us=np.array([x.device_timestamp_us for x in S_]), host_ns=np.array([x.host_receive_ns for x in S_]),
                     gyro=np.array([[x.gx, x.gy, x.gz] for x in S_]), acc=np.array([[x.ax, x.ay, x.az] for x in S_]))
            print(f"[태그] {sd} IMU 샘플 {len(S_)}", flush=True)
        except Exception as e: print(f"[태그] {sd} IMU 저장 실패: {e}", flush=True)
    try:
        if D: np.savez_compressed(out / "depth.npz", depth=np.stack(D))
    except Exception as e: print(f"[태그] depth 저장 실패: {e}", flush=True)
    (out / "meta.json").write_text(json.dumps({**json.loads((out / "meta.json").read_text()), "depth_files": "png" if long_take else "npz"}, ensure_ascii=False, indent=1))
    u = os.environ.get("SUDO_UID"); g = os.environ.get("SUDO_GID")
    if u and g:
        for pth in [out, *out.iterdir()]: os.chown(pth, int(u), int(g))
        os.chown(out.parent, int(u), int(g))
    print(f"[태그] 저장 {out} | 프레임 {k} | 태그 2개 이상 보인 비율 {np.mean(np.array(seen) >= 2):.2f}, 1개 이상 {np.mean(np.array(seen) >= 1):.2f}", flush=True)


if __name__ == "__main__":
    main()
