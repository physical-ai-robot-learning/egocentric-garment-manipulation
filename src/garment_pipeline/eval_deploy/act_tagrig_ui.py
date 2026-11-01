#!/usr/bin/env python3
"""[2026-10-23] Inference web UI for the tag-rig laundry policies on the bimanual reBot (http://localhost:8770, Korean UI).
Built on deploy/act_tagrig_deploy.py: the SAME Rig (cameras, live Orbbec, robot poller, kinematics) and run_session (receding
horizon, limiter -> IK -> capsule link collision / table / joint limits / workspace -> HOLD-only stop, 120 mm warn-only, preview).
  live views   Orbbec with both arms' predicted 32-step TCP chunks projected (L blue / R orange, k1 big dot, measured TCP black ring),
               robot wrist views (+ optional UMI training reference frame and live jaw-area / axis numbers = domain-gap check)
  status       plan Hz, policy latency, bridge connected / estop / parked, per-arm read age, motor status (/motor_states), TCP, flags
  gripper      raw continuous outputs: 32-step strip per arm + k1 history, threshold 0.5
  controls     policy ACT / DiT + checkpoint dropdown, commit, speed, slow-start speed, max seconds, DiT DDIM steps -- persisted to
               deploy/act_tagrig_ui_settings.json (a UI restart never resets them)
  modes        [DRY-RUN 시작] | [실행 시작 (GO)] | [스텝 모드로 시작 (GO)] + [한 스텝 실행 ▶ (N)] (10-28: one click = one committed segment) (confirm; wave-confirmed cams, arms at READY, live Orbbec) | [정지 / HOLD] (Space; GO:
               bridge /estop = hold measured pose, nothing moves afterwards) | [READY로 이동] / [휴식(park)] (confirm, idle only)
  watchdog     during GO the page heartbeats every 250 ms; > 1 s without one -> HOLD (server side, also when the tab dies)
  runs         reports/act_deploy/<run>/summary.json + Desktop mp4 per run, listed with links
    .venv-mac/bin/python deploy/act_tagrig_ui.py [--port 8770] [--bridge-url http://127.0.0.1:8031 (mock)] [--dry-only]"""
from __future__ import annotations
import argparse, json, pathlib, sys, threading, time, traceback

import cv2
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
from garment_pipeline.eval_deploy import act_tagrig_core as core  # noqa: E402
from garment_pipeline.eval_deploy import act_tagrig_deploy as dep  # noqa: E402
from egogarment import paths as egopaths  # noqa: E402

SETTINGS = HERE / "act_tagrig_ui_settings.json"
XVLA_FT = (egopaths.CKPT_ROOT / "ckpts/runpod/pod2/XVLA-FT-GOLD281-D21-FROM-MIX3-100K-RELBIN32-B8-300K-FP32/checkpoints")
DEFAULTS = dict(policy="act", ckpt_act=str(core.CKPT), ckpt_dit=str(core.DIT_CKPT), ckpt_xvla=str(XVLA_FT / "100000"), commit_act=1, commit_dit=4, commit_xvla=8,
                speed=0.5, slow_start_speed=0.25, active_arms="both", grip_fake_open=False, rtc="blend", z_offset_mm=0.0, height_calib=False, tens=False, grip_cont=True, grip_mode="linear", grip_gain=10.0,
                secs=60.0, dit_steps=10, show_umi_ref=False)
CK_DIRS = {"act": [core.LDT / "checkpoints_pod/act_tagrig_v1_1m/checkpoints"],
           "dit": [core.LDT / "checkpoints_pod/dit_tagrig_v1/checkpoints",
                   # 10-28: Exp 1 = DiT Soft-Fold pretrain 300k -> GOLD-281 300k (best offline val/test at 300k), SSD archive
                   (egopaths.CKPT_ROOT / "ckpts/runpod/pod1/DITFT-SFPRE300K-FLAT300-A-GOLD281-BS16-300K/checkpoints"),
                   # 10-30: UMI-style S03 (0.3x) GOLD-281 all-281 B64 BCE GSM, H100 (stopped at epoch 26) + UMI50 0.5x 150 epochs, 4090
                   (egopaths.CKPT_ROOT / "ckpts/runpod/h100/DITFT-MIX3PT100K-S03ALL281-B64-BCE-GSM-E150-NOCLIP/checkpoints"),
                   (egopaths.CKPT_ROOT / "ckpts/4090/DITFT-S03PT50K-UMI50-S05-B64ACC4-BCE-GSM-E150-NOCLIP/checkpoints"),
                   # 10-30: today's pinching (A' 53, 0.5x) 1000-epoch run from E150, every 25 epochs; P10 test: 50 ep 18.5 mm vs E150 28.5
                   (egopaths.CKPT_ROOT / "ckpts/4090/DITFT-UMI50E150-U1030AP-S05-B64ACC4-BCE-GSM-E1000-NOCLIP/checkpoints")],
           "xvla": [XVLA_FT]}         # 10-28: Exp 3 X-VLA through deploy/xvla_server.py (:8775, separate venv)


# 10-30: 1.0 s tripped two GO runs (heartbeat 1.04 s old) -- the DiT forward in this process holds the GIL while /api/hb waits, and a
# background browser tab throttles setInterval to 1 s.  Space / [정지] still stop at once.
HB_TIMEOUT_S = 3.0


def load_settings():
    s = dict(DEFAULTS)
    try: s.update(json.loads(SETTINGS.read_text()))
    except Exception: pass
    return s


def save_settings(s): SETTINGS.write_text(json.dumps(s, indent=1, ensure_ascii=False))


def ckpts(kind):
    return [str(p) for d in CK_DIRS[kind] if d.exists() for p in sorted(d.iterdir()) if (p / "pretrained_model").is_dir()]


class UICtl:
    """run_session control: stop request (button / Space / watchdog) + heartbeat; latest snapshot for the views"""
    def __init__(self, go):
        self.go = go; self.stop_req = None; self.hb = time.monotonic(); self.snap = None; self.k1 = []; self.lock = threading.Lock()
        self.step_mode = False; self.step_n = 0; self.steps_done = 0       # 10-28 step mode: [한 스텝 실행] sets ONE token

    def take_step(self):
        with self.lock:
            if self.step_n <= 0: return False
            self.step_n -= 1; self.steps_done += 1; return True

    def poll(self):
        if self.stop_req: return self.stop_req
        if self.go and time.monotonic() - self.hb > HB_TIMEOUT_S: self.stop_req = f"UI heartbeat lost (> {HB_TIMEOUT_S:.0f} s)"; return self.stop_req
        return None

    def publish(self, snap):
        with self.lock:
            self.snap = snap; self.k1.append((snap["t"], float(snap["grip"][0, 0]), float(snap["grip"][0, 1]))); self.k1 = self.k1[-400:]


class App:
    def __init__(self, a):
        self.a = a; self.s = load_settings(); save_settings(self.s)
        self.run_args = dep_parser().parse_args(["run"] + (["--replay-global", a.replay_global] if a.replay_global else []) + (["--global-url", a.global_url]) +
                                                 (["--bridge-url", a.bridge_url] if a.bridge_url else []))
        self.rig = dep.Rig(self.run_args).start(); self.lock = threading.Lock()
        self.ctl = None; self.thread = None; self.mode = "idle"; self.msg = "준비됨"; self.last = None; self.busy = None
        self.status = {}; self.motors = {}
        from visual_teleop.robot.bridge import HttpBridge
        self.br_ro = HttpBridge(self.rig.url, live=False, timeout_s=0.5)
        threading.Thread(target=self._poll_bridge, daemon=True).start(); threading.Thread(target=self._watchdog, daemon=True).start()
        self.umi_ref = self._umi_refs(); self.jaw = {}
        threading.Thread(target=self._jaw_loop, daemon=True).start()

    # ---- background
    def _poll_bridge(self):
        while True:
            try: self.status = self.br_ro.s.get(self.rig.url + "/status", timeout=0.5).json()
            except Exception as e: self.status = {"error": f"{type(e).__name__}"}
            try: self.motors = self.br_ro.s.get(self.rig.url + "/motor_states", timeout=0.5).json()
            except Exception: self.motors = {}
            time.sleep(0.5)

    def _watchdog(self):
        """server-side: GO running and no heartbeat for > HB_TIMEOUT_S -> /estop at once (hold), even if run_session is inside an inference"""
        while True:
            c = self.ctl
            if c is not None and c.go and self.mode == "go" and time.monotonic() - c.hb > HB_TIMEOUT_S:
                c.stop_req = c.stop_req or f"UI heartbeat lost (> {HB_TIMEOUT_S:.0f} s)"; self.br_ro.estop(); self.mode = "hold"; self.msg = "하트비트 끊김 -> HOLD (E-STOP 홀드)"
                print(f"[UI] watchdog HOLD {time.strftime('%H:%M:%S')} (heartbeat {time.monotonic() - c.hb:.2f} s old)", flush=True)
                time.sleep(0.5)
            time.sleep(0.05)

    def _umi_refs(self):
        try:
            from garment_pipeline.eval_deploy.act_tagrig_domain_gap import umi_frames
            U = umi_frames(1, start_only=True); return {s: cv2.imread(str(U[s][len(U[s]) // 2])) for s in core.SIDES}
        except Exception: return {}

    def _jaw_loop(self):
        from garment_pipeline.eval_deploy.act_tagrig_domain_gap import measure
        while True:
            if self.s.get("show_umi_ref"):
                for s in core.SIDES:
                    f = self.rig.wr[s].raw
                    if f is not None:
                        m, _, _ = measure(cv2.resize(f, (1280, 720), interpolation=cv2.INTER_AREA))
                        self.jaw[s] = dict(area=round(m["jaw_area_pct"], 1), top=m.get("jaw_top_y"), axis=round(m.get("jaw_axis_deg_from_vertical", float("nan")), 0))
                if not self.jaw.get("umi") and self.umi_ref:
                    self.jaw["umi"] = {s: (lambda m: dict(area=round(m["jaw_area_pct"], 1), top=m.get("jaw_top_y"), axis=round(m.get("jaw_axis_deg_from_vertical", float("nan")), 0)))(measure(self.umi_ref[s])[0]) for s in core.SIDES}
            time.sleep(0.5)

    # ---- runs
    def start_run(self, go, step=False):
        with self.lock:
            if self.thread is not None and self.thread.is_alive(): return False, "이미 실행 중"
            if self.busy: return False, f"{self.busy} 중"
            if go and self.a.dry_only: return False, "이 UI 는 --dry-only 로 실행됨 (GO 비활성)"
            if go and self.rig.replay: return False, "녹화 프레임(replay) 모드에서는 GO 불가"
            if go and not self.rig.wave_ok: return False, "손목 카메라 wave 테스트 미확인"
            s = self.s; ra = self.run_args; k = s["policy"]
            ra.policy = k; ra.ckpt = s[f"ckpt_{k}"]; ra.commit = int(s[f"commit_{k}"]); ra.speed = float(s["speed"]); ra.slow_start_speed = float(s["slow_start_speed"])
            ra.secs = float(s["secs"]); ra.dit_steps = int(s["dit_steps"]); ra.go = bool(go); ra.allow_start_pose = True   # READY or start_pose.json
            ra.step_mode = bool(go and step); ra.secs = 0.0 if ra.step_mode else ra.secs        # step mode: no time limit (the user drives it)
            ra.active_arms = s.get("active_arms", "both")                          # 10-28: one-arm execution (other arm dry)
            ra.grip_fake_open = bool(s.get("grip_fake_open", False))                 # 10-29 diagnostic: jaw state fed as OPEN
            gm = s.get("grip_mode", "linear" if s.get("grip_cont", True) else "binary")   # 10-30: binary | linear | sigmoid
            ra.grip_continuous = gm == "linear"; ra.grip_mode = gm if gm in ("sigmoid", "relu") else ""; ra.grip_gain = float(s.get("grip_gain", 10.0))
            ra.z_offset_mm = float(s.get("z_offset_mm", 0.0))                       # 10-30 height correction (not a limit)
            ra.height_calib = bool(s.get("height_calib", False))                     # 11-01 robot-measured per-arm table height (height_calib.py)
            # 10-30: temporal ensembling averaged old plans anchored on the MEASURED (lagging) TCP -> every new no-look-ahead chunk was
            # pulled ~45 mm back ("백포지션").  Off unless the user ticks it.
            if not s.get("tens", False): ra.temporal_ensemble = 0.0
            ra.no_lookahead = s.get("rtc") == "nolook"                             # 10-30: no look-ahead (whole commit, then infer)
            ra.rtc = "off" if ra.no_lookahead else s.get("rtc", "blend"); ra.rtc_beta = 5.0                     # 10-29 Real-Time Chunking (GO look-ahead only). Offline (scratchpad rtc_offline_test.py): blend = prefix 0 mm, overlap -30..35 %, no extra latency; guided ~no effect on the DDIM eps model
            if go:                              # 10-29 user: "스텝모드 말고 그냥" -- the same relaxations for the continuous GO too
                ra.max_img_age_ms, ra.max_plan_age_s = 600.0, 1.5   # 10-28: robot is still between steps (Mac DiT plan 0.2-0.45 s); 10-29 X-VLA plan ~0.9 s on MPS
                # 10-28 user: limits relaxed (joint 30 -> 60 deg/s, TCP 0.25 -> 0.5 m/s); continue from the CURRENT pose (no READY /
                # start-pose check) -- the user approves every chunk on screen; collision / table / joint-limit / workspace / IK STOPs stay
                ra.max_deg_s, ra.max_tcp_mps, ra.ready_tol_deg = 60.0, 0.5, 180.0
                ra.max_ik_stop_mm = 1e9         # 10-28 user: "리스트릭션 다 없애줘" -- IK misses never stop (closest reachable pose)
                ra.resync_deg = 20.0            # 10-29: normal tracking lag (50-80 mm) crossed 6 deg -> re-seed on the measured pose = a jump back
                                                #        ("뒤로 돌아가는 움직임"); 20 deg still catches a real drive dropout
                ra.no_safety = True             # collision / table / joint-limit / workspace STOPs -> WARN (as the 10-27 label replay)
                ra.no_floor = True              # 10-29 user "바닥 제한을 없애줘": no finger-corner table floor (corner_lift) -- targets may reach the table
                ra.max_deg_s, ra.max_tcp_mps, ra.max_rot_dps = 90.0, 1.0, 360.0     # speed caps at the bridge maximum
                try:                                                    # resume after a HOLD: the bridge keeps holding the measured pose
                    if (self.br_ro.s.get(self.rig.url + "/status", timeout=1.0).json() or {}).get("estop"):
                        self.br_ro.s.post(self.rig.url + "/clear_estop", timeout=2.0); time.sleep(0.5)
                except Exception as e: return False, f"E-STOP 해제 실패: {type(e).__name__}"
            self.ctl = UICtl(go); self.ctl.step_mode = ra.step_mode; self.mode = "go" if go else "dry"
            self.msg = ("실행 (GO, 스텝 모드: 버튼 한 번 = 한 구간)" if ra.step_mode else "실행 (GO)" if go else "DRY-RUN") + " 시작 중..."
            self.thread = threading.Thread(target=self._run, args=(ra, self.ctl), daemon=True); self.thread.start(); return True, "시작"

    def _run(self, ra, ctl):
        try:
            S = dep.run_session(ra, self.rig, ctl); self.last = S or {}
            self.msg = f"끝: {(S or {}).get('stop_reason')}" + (" -- 로봇은 현재 자세 유지 (HOLD). 다음 동작은 버튼으로만" if ra.go else "")
        except SystemExit as e: self.msg = f"시작 안 됨: {e}"
        except Exception as e: self.msg = f"오류: {type(e).__name__}: {e}"; traceback.print_exc()
        finally: self.mode = "hold" if ra.go else "idle"

    def step(self):
        c = self.ctl
        if c is None or self.mode != "go" or not c.step_mode: return False, "스텝 모드 실행 중이 아님"
        with c.lock: c.step_n = 1                                       # at most one pending step
        return True, f"한 스텝 요청 (지금까지 {c.steps_done} 스텝)"

    def hold(self):
        c = self.ctl; go_running = self.mode == "go"
        if c is not None: c.stop_req = c.stop_req or "정지 / HOLD (UI)"
        # 11-01: soft stop -- a running GO brakes onto its current command (run_session), idle arms get the bridge /stop (hold, no latch)
        if not self.a.dry_only and not go_running and self.mode in ("hold", "idle"):
            try: self.br_ro.s.post(self.rig.url + "/stop", timeout=2.0)
            except Exception: pass
        if go_running: self.mode = "hold"
        self.msg = ("정지 (DRY 전용 UI: 브리지 명령 없음)" if self.a.dry_only else "정지 -- 감속 후 현재 자세 유지 (E-STOP 아님)"); return True, self.msg

    def estop(self):
        """11-01: the hard stop kept as its own button: latch the bridge E-STOP (instant hold at the measured pose)"""
        c = self.ctl
        if c is not None: c.stop_req = c.stop_req or "E-STOP (UI)"
        if not self.a.dry_only: self.br_ro.estop()
        if self.mode == "go": self.mode = "hold"
        self.msg = "E-STOP -- 즉시 정지 (현재 자세 유지, 해제 필요)"; return True, self.msg

    def _motion(self, name, fn):
        with self.lock:
            if (self.thread is not None and self.thread.is_alive()) or self.busy: return False, "실행 중에는 불가"
            if self.a.dry_only: return False, "이 UI 는 --dry-only 로 실행됨 (로봇 이동 비활성)"
            self.busy = name
        def w():
            from visual_teleop.robot.bridge import HttpBridge
            brl = None
            try:
                brl = HttpBridge(self.rig.url, live=True, timeout_s=0.5); fn(brl); self.msg = f"{name} 끝"
            except SystemExit as e: self.msg = f"{name} 중단: {e}"
            except Exception as e: self.msg = f"{name} 오류: {type(e).__name__}: {e}"; traceback.print_exc()
            finally:
                if brl is not None: brl.close()
                self.busy = None; self.mode = "idle"
        threading.Thread(target=w, daemon=True).start(); return True, f"{name} 시작"

    def go_ready(self): return self._motion("READY로 이동", lambda br: move_ready(br, self.rig))

    def go_start(self):
        """10-23 user: READY -> training start pose from the UI = the deploy `start --go` (same plan + checks, its own bridge client)"""
        import argparse
        def f(br):
            br.close()                                    # start_cmd opens its own live client; keep a single writer
            dep.start_cmd(argparse.Namespace(go=True, episode=-1, start_lift=0.0, deg_s=10.0, assume_ready=False, root="flat300_A_gold281_train", label="", height_calib=bool(self.s.get("height_calib", False))))   # 10-29: GOLD-281 medoid start (ep 104), no lift: the training starts sit at the tag edge (L 291/148/33, R 305/-216/28 mm; plan: IK <= 3 mm, link-table 24 mm) -- was the OLD dataset ep 82 + 3 cm (5-6 cm off)
        return self._motion("시작 자세로 이동", f)

    def clear_estop(self):
        """10-30 user "e stop 푸는 버튼도 넣어줘": idle only.  /clear_estop; if the bridge refuses because an arm's torque is off (the guard
        stays latched), try /reenable once and clear again.  Nothing moves: the arms keep holding the measured pose."""
        if self.mode in ("go", "dry") or self.busy: return False, "실행/이동 중에는 해제할 수 없습니다"
        try:
            u = self.rig.url; r = self.br_ro.s.post(u + "/clear_estop", timeout=3.0).json()
            if not r.get("ok", True) or r.get("error"):
                # 10-30: after a left-arm power cycle /reenable succeeded but the guard stayed latched; /clear_collision clears the latch,
                # re-enables both arms at the measured pose and releases E-STOP in one call
                re_ = self.br_ro.s.post(u + "/clear_collision", timeout=20.0).json(); r = {"ok": True} if re_.get("ok") else re_
                if not r.get("ok", True) or r.get("error"):
                    return False, f"E-STOP 해제 실패: {r.get('error', r)} | clear_collision: {re_.get('arms', re_)} -- 팔 토크/모터 커넥터 확인"
            self.mode = "idle"; self.msg = "E-STOP 해제됨 (팔은 현재 자세 유지)"; return True, self.msg
        except Exception as e:
            return False, f"E-STOP 해제 실패: {type(e).__name__}: {e}"

    def park(self):
        def f(br):
            st = br.status()
            if st.get("estop"): br.s.post(br.base_url + "/clear_estop", timeout=2.0)
            r = br.s.post(br.base_url + "/rest", timeout=3.0).json()
            if not r.get("ok"): raise SystemExit(f"/rest 거부: {r}")
        return self._motion("휴식(park)", f)

    # ---- views
    def tcp(self):
        _, (q, _, _) = self.rig.rob.ring.latest()
        return {s: np.round(core.fk(self.rig.arms, s, q[s])[:3, 3], 3).tolist() for s in core.SIDES} if q is not None else None

    def global_jpg(self):
        _, x = self.rig.gl.ring.latest()
        if x is None: return None
        g = cv2.imdecode(np.frombuffer(x[1], np.uint8), cv2.IMREAD_COLOR); T_C_B, K = core.robot_cam(); c = self.ctl
        # 11-01 user "UI 에서 TCP 가 벗어나": the Orbbec extrinsic is in the label B frame, FK is the robot's -> map FK points to B with
        # deploy/height_calib.json (per-arm xy shift from the tag camera ray + table-height plane) before projecting
        from garment_pipeline.eval_deploy import height_calib as HC; hc = HC.load(); toB = lambda s_, P_: HC.to_B(hc, s_, P_)
        snap = c.snap if (c is not None and self.mode in ("dry", "go")) else None
        _, (q, _, _) = self.rig.rob.ring.latest()
        if q is not None:
            for s in core.SIDES:
                u = core.project(T_C_B, K, toB(s, core.fk(self.rig.arms, s, q[s])[None, :3, 3]))[0]
                if np.isfinite(u).all(): cv2.circle(g, tuple(u.astype(int)), 9, (0, 0, 0), 2)
        if snap is not None:
            for s in core.SIDES:
                uv = core.project(T_C_B, K, toB(s, np.array([T[:3, 3] for T in snap["Tc"][s]])))
                for k in range(31):
                    if np.isfinite(uv[k]).all() and np.isfinite(uv[k + 1]).all(): cv2.line(g, tuple(uv[k].astype(int)), tuple(uv[k + 1].astype(int)), dep.COL[s], 2)
                if np.isfinite(uv[0]).all(): cv2.circle(g, tuple(uv[0].astype(int)), 7, dep.COL[s], -1); cv2.circle(g, tuple(uv[0].astype(int)), 8, (255, 255, 255), 1)
                if np.isfinite(uv[-1]).all(): cv2.circle(g, tuple(uv[-1].astype(int)), 5, dep.COL[s], 2)
        return cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()

    def wrist_jpg(self, s):
        f = self.rig.wr[s].raw
        if f is None: return None
        v = cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA)
        if self.s.get("show_umi_ref") and s in self.umi_ref:
            r = cv2.resize(self.umi_ref[s], (640, 360)); v = np.hstack([v, r])
            cv2.putText(v, "UMI train ref", (650, 24), 0, 0.7, (0, 255, 255), 2)
            j = self.jaw.get(s); ju = (self.jaw.get("umi") or {}).get(s)
            if j: cv2.putText(v, f"robot jaw area {j['area']}% top {j['top']} axis {j['axis']}", (8, 350), 0, 0.55, (0, 255, 0), 2)
            if ju: cv2.putText(v, f"UMI jaw area {ju['area']}% top {ju['top']} axis {ju['axis']}", (650, 350), 0, 0.55, (0, 255, 0), 2)
        cv2.putText(v, f"{s.upper()} {self.rig.cfg[s]['location']}", (8, 24), 0, 0.7, (0, 255, 255), 2)
        return cv2.imencode(".jpg", v, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()

    def api_status(self):
        c = self.ctl; snap = c.snap if c is not None else None; st = self.status or {}
        if self.s.get("policy") == "xvla":
            try: xs = self.br_ro.s.get("http://127.0.0.1:8775/status", timeout=0.3).json(); self.xvla = dict(up=True, ckpt=xs.get("ckpt"))
            except Exception: self.xvla = dict(up=False)
        mot = {f"{k}": all(m.get("status") == 1 for m in v.values()) for k, v in (self.motors or {}).items() if isinstance(v, dict)}
        out = dict(mode=self.mode, msg=self.msg, busy=self.busy, settings=self.s, dry_only=self.a.dry_only, replay=self.rig.replay, wave_ok=self.rig.wave_ok,
                   xvla=getattr(self, "xvla", None), step_mode=bool(c is not None and c.step_mode), step_pending=bool(c is not None and c.step_n > 0), steps_done=(c.steps_done if c is not None else 0),
                   bridge=dict(url=self.rig.url, connected=st.get("connected"), estop=st.get("estop"), parked=st.get("parked"), status=st.get("status"), error=st.get("error") or st.get("error"),
                               read_age_ms={k: v.get("read_age_ms") for k, v in (st.get("arms") or {}).items()}, motors_ok=mot),
                   tcp=self.tcp(), ckpts={k: ckpts(k) for k in CK_DIRS}, orbbec_frames=self.rig.gl.n,
                   wrist_frames={s: self.rig.wr[s].n for s in core.SIDES})
        if snap is not None:
            out.update(t=round(snap["t"], 1), hz=round(snap["hz"], 1), lat_pol_ms=round(snap["lat_pol_ms"], 1), lat_total_ms=round(snap["lat_total_ms"], 1), k0=snap["k0"],
                       flags=[f for f in snap["flags"] if not f.startswith("LIM")], lims=[f for f in snap["flags"] if f.startswith("LIM")], speed=snap["speed"],
                       grip=np.round(snap["grip"], 3).tolist(), g_cmd=snap["g_cmd"], k1=[[round(t, 2), round(l, 3), round(r, 3)] for t, l, r in c.k1[-300:]])
        if self.last: out["last"] = {k: self.last.get(k) for k in ("run", "stop_reason", "plan_hz", "latency_ms", "video", "grip_plot", "violation_cycles", "flag_counts")}
        return out


def move_ready(br, rig):
    """READY = visual_teleop ready_pose's fixed READY_Q_DEG, with ready_pose's checks (TCP >= 0.10 m along the joint line, bridge
    joint limits) and its smoothstep joint ramp at 15 deg/s through HttpBridge.send; both arms, one at a time (right, then left), each
    path also checked against the other arm (capsule clearance).  /clear_estop first (the user confirmed in the UI)."""
    from visual_teleop.tools.robot_moves import unpark
    st = br.status()
    if st.get("estop"): br.s.post(br.base_url + "/clear_estop", timeout=2.0)
    # 11-01: a PARKED arm is re-enabled by the first /execute_step (~3 s) -> the stream's 0.5 s timeout E-STOPPED the READY move.
    # Unpark first with a long timeout; a refused write (Damiao latch: reads ok, writes ignored) stops here with the reason.
    for s in (br.status().get("parked") or []):
        r = br.s.post(br.base_url + "/torque", json={"arm": s, "on": True}, timeout=15.0).json()
        msg = str((r.get("arms") or {}).get(s, ""))
        if msg.startswith("ERR"): raise SystemExit(f"{s} 팔 토크 켜기 실패 ({msg[:80]}) -- 모터 쓰기 무응답: 그 팔 전원을 껐다 켜기")
    for s in ("right", "left"):
        o = br.observe()
        if o.estop: raise SystemExit("E-STOP/읽기 지연")
        q0 = o.q[s].copy(); qr = rig.qr[s]; other = "left" if s == "right" else "right"
        zmin = min(core.fk(rig.arms, s, q0 + u * (qr - q0))[2, 3] for u in np.linspace(0, 1, 41))
        if zmin < 0.10 and np.degrees(np.abs(qr - q0)).max() > 2: raise SystemExit(f"{s}: 경로 중 TCP 최저 {zmin:.3f} m < 0.10 -- 이동 안 함")
        if np.any(qr < rig.arms[s].lo) or np.any(qr > rig.arms[s].hi): raise SystemExit("READY 가 관절 한계 밖")
        for u in np.linspace(0, 1, 21):
            cc = rig.col.check({s: q0 + u * (qr - q0), other: o.q[other]})
            if not cc["ok_arm"]: raise SystemExit(f"{s}: READY 경로에서 두 팔 충돌 위험 ({cc['pair']}) -- 이동 안 함")
        unpark(br, s); o = br.observe(); q0 = o.q[s].copy(); d = qr - q0; g = {x: o.g(x) for x in core.SIDES}
        n = max(2, int(np.ceil(np.degrees(np.abs(d)).max() / 15.0 * 30.0))); t0 = time.monotonic()
        for k in range(1, n + 1):
            u = k / n; u = u * u * (3 - 2 * u)
            r = br.send({s: q0 + u * d, other: o.q[other]}, g)
            if not r.accepted: br.estop(); raise SystemExit(f"브리지 거부: {r.error} -- E-STOP")
            if k % 30 == 0 and br.observe().estop: br.estop(); raise SystemExit("이동 중 E-STOP/읽기 지연")
            time.sleep(max(0.0, t0 + k / 30.0 - time.monotonic()))


def dep_parser():
    """the deploy CLI's `run` arguments (single source of defaults)"""
    import argparse as ap_
    orig = ap_.ArgumentParser.parse_args
    holder = {}
    def grab(self, args=None, namespace=None):
        holder["p"] = self; raise SystemExit(0)
    ap_.ArgumentParser.parse_args = grab
    try: dep.main()
    except SystemExit: pass
    finally: ap_.ArgumentParser.parse_args = orig
    return holder["p"]


PAGE = r"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>세탁 정책 추론</title>
<style>
body{margin:0;background:#15171a;color:#e6e6e6;font:14px -apple-system,'Apple SD Gothic Neo',sans-serif}
.top{display:flex;gap:8px;padding:8px;align-items:flex-start}.col{display:flex;flex-direction:column;gap:8px}
img{background:#000;border-radius:6px;display:block}
.card{background:#202328;border-radius:8px;padding:10px}
.bar{display:flex;flex-wrap:wrap;gap:14px;padding:6px 10px;background:#202328;border-radius:8px}
.k{color:#9aa}.ok{color:#5d5}.bad{color:#f66;font-weight:600}.warn{color:#fb4}
button{font-size:15px;padding:9px 14px;border-radius:7px;border:0;cursor:pointer;margin:3px;background:#3a4048;color:#fff}
button:disabled{opacity:.35;cursor:default}
#hold{background:#c21d1d;font-size:22px;padding:16px 30px;font-weight:700}
#go{background:#1f6f3a}#dry{background:#2b5aa6}#gostep{background:#5a3fa0}#step{background:#7a52d0;font-size:20px;padding:14px 26px;font-weight:700}
select,input{background:#2b2f35;color:#eee;border:1px solid #444;border-radius:5px;padding:4px;margin:2px}
canvas{background:#111;border-radius:6px}
#flags span{display:inline-block;margin:2px 4px;padding:1px 6px;border-radius:4px;background:#3a2222}
table{border-collapse:collapse}td{padding:2px 8px;border-bottom:1px solid #333}
a{color:#8cf}
</style></head><body>
<div class="bar" id="status"></div>
<div class="top">
 <div class="col"><img id="g" src="/stream/global" width="848" height="480"><div class="card" id="flagsbox"><b>안전 플래그</b> <span class="k">(최근 사이클)</span><div id="flags"></div><div id="lims" class="k"></div></div>
  <div class="card"><b>집게 출력 (원시값, 1 = 열림, 기준선 0.5)</b><br><canvas id="strip" width="820" height="120"></canvas><canvas id="hist" width="820" height="160"></canvas></div></div>
 <div class="col"><img id="wl" src="/stream/left" width="560"><img id="wr" src="/stream/right" width="560">
  <div class="card"><b>설정</b> <span class="k">(파일에 저장됨)</span><br>
   정책 <select id="policy"><option value="act">ACT</option><option value="dit">DiT</option><option value="xvla">X-VLA (서버 :8775)</option></select>
   체크포인트 <select id="ckpt" style="max-width:320px"></select><br>
   커밋 스텝 <select id="commit"></select> 속도 <input id="speed" type="number" step="0.05" min="0.05" max="2" style="width:60px">
   처음 2초 속도 <input id="slow" type="number" step="0.05" min="0.05" max="1" style="width:60px"><br>
   최대 시간(초, 0 = 무제한) <input id="secs" type="number" step="5" min="0" max="100000" style="width:70px">
   DiT 샘플러 스텝 <input id="dsteps" type="number" step="1" min="2" max="100" style="width:50px">
   <label><input id="umi" type="checkbox"> UMI 학습 기준 영상 + 집게 수치</label><br>
   움직일 팔 <select id="arms"><option value="both">두 팔</option><option value="left">왼팔만 (오른팔은 계획만)</option><option value="right">오른팔만 (왼팔은 계획만)</option></select>
   <label title="진단용: 모델이 상태 입력의 집게 값을 따라 하므로(copycat), 항상 '열림'으로 넣어 청크 중간의 닫힘 시점을 씀"><input type="checkbox" id="gfo"> 집게 상태를 항상 열림으로 입력 (진단)</label>
   <label title="Real-Time Chunking: 다음 청크를 이미 실행 중인 계획에 이어지게 생성 (GO에서만). DiT=유도(guided), X-VLA는 항상 혼합(blend)">RTC <select id="rtc"><option value="blend">켜기 (blend)</option><option value="guided">guided (실험, 효과 약함)</option><option value="off">끄기</option><option value="nolook">미리 추론 끔 (청크 다 실행 후 추론)</option></select></label> <label title="이진: 0.5 기준 열림/닫힘 | 연속: 열림 확률 그대로 | 시그모이드: 1/(1+exp(-기울기(p-0.5))) | ReLU: clip(기울기(p-0.5),0,1) (0.5 아래 닫힘, 기울기 2 = 0.5~1 을 0~1 로)">집게 <select id="gmode"><option value="binary">이진 (0/1)</option><option value="linear">연속 (확률 그대로)</option><option value="sigmoid">시그모이드</option><option value="relu">ReLU</option></select></label> <label title="시그모이드/ReLU 기울기 (클수록 이진에 가까움; ReLU 는 2 권장)">기울기 <input type="number" id="ggain" step="1" style="width:46px"></label> <label title="이전 계획들을 평균 (켜면 뒤처진 실제 위치로 끌려가 뒤로 가는 동작이 생김)"><input type="checkbox" id="tens"> 시간 앙상블</label> <label title="궤적 전체 높이 보정 (첫 청크에서 한 번만 더함, + = 위로). 제한이 아님">높이 보정 mm <input type="number" id="zoff" step="5" style="width:60px"> <label title="deploy/height_calib.py measure 로 로봇이 잰 테이블 태그 높이 차이를 팔마다 자동으로 더함 (위 mm 값과 합산)"><input type="checkbox" id="hcal"> 로봇 높이 캘리브</label><select style="display:none"></select></label></div>
  <div class="card"><button id="dry" onclick="act('dry')">DRY-RUN 시작</button><button id="go" onclick="goRun()">실행 시작 (GO)</button>
   <button id="gostep" onclick="goStep()">스텝 모드로 시작 (GO)</button><br>
   <button id="step" onclick="stepOnce()">한 스텝 실행 ▶ (N)</button> <span id="stepinfo" class="k"></span><br>
   <button id="hold" onclick="hold()">정지 (부드럽게, Space)</button> <button id="estopbtn" onclick="fetch('/api/estop',{method:'POST'})" style="background:#c62828;color:#fff">E-STOP (즉시)</button> <button id="clrestop" onclick="conf('clear_estop','E-STOP 을 해제합니다 (팔은 현재 자세 유지, 움직이지 않음). 진행할까요?')">E-STOP 해제</button><br>
   <button id="ready" onclick="conf('ready','READY 자세로 천천히 이동합니다. 주변을 비웠나요?')">READY로 이동</button>
   <button id="startpose" onclick="conf('start','READY 에서 학습 시작 자세(테이블 위 약 3 cm)로 천천히 이동합니다. 두 팔이 READY 이고 천이 비켜 있나요?')">시작 자세로 이동</button>
   <button id="park" onclick="conf('park','휴식(park) 자세로 이동 후 토크를 끕니다. 진행할까요?')">휴식(park)</button>
   <div id="msg" style="margin-top:6px"></div></div>
  <div class="card"><b>지난 실행</b><div id="runs" style="max-height:220px;overflow:auto"></div></div>
 </div></div>
<script>
let S=null, hbOn=false;
async function post(u,b){const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});return r.json()}
function hold(){post('/api/hold')}
document.addEventListener('keydown',e=>{if(e.code==='Space'){e.preventDefault();hold()} else if(e.code==='KeyN'&&S&&S.mode==='go'&&S.step_mode){e.preventDefault();stepOnce()}});
async function act(m){const r=await post('/api/'+m);document.getElementById('msg').textContent=r.msg}
function goRun(){if(confirm('실제 로봇이 움직입니다 (GO). 사람이 옆에 있고, 두 팔이 READY 이고, 천이 놓였나요?\n정지 = 빨간 버튼 / Space'))act('go')}
function goStep(){if(confirm('스텝 모드 (GO): 로봇은 [한 스텝 실행] 을 누를 때만, 그때의 최신 계획 한 구간(커밋 스텝 수)만큼 움직입니다.\n사람이 옆에 있고, 두 팔이 READY 이고, 천이 놓였나요?\n정지 = 빨간 버튼 / Space'))act('gostep')}
async function stepOnce(){const r=await post('/api/step');document.getElementById('msg').textContent=r.msg}
function conf(m,t){if(confirm(t))act(m)}
setInterval(()=>{if(S&&S.mode==='go')post('/api/hb')},250);
function setv(id,v){const e=document.getElementById(id);if(document.activeElement!==e)e.value=v}
let inited=false;
async function save(){const p=document.getElementById('policy').value;const b={policy:p,speed:+speed.value,slow_start_speed:+slow.value,secs:+secs.value,dit_steps:+dsteps.value,show_umi_ref:umi.checked,active_arms:arms.value,grip_fake_open:gfo.checked,tens:document.getElementById('tens').checked,grip_mode:document.getElementById('gmode').value,grip_gain:+document.getElementById('ggain').value,z_offset_mm:+document.getElementById('zoff').value,height_calib:document.getElementById('hcal').checked,rtc:document.getElementById('rtc').value};
 b['ckpt_'+p]=ckpt.value;b['commit_'+p]=+commit.value;await post('/api/settings',b);inited=false}
['policy','ckpt','commit','speed','slow','secs','dsteps','umi','arms','gfo','rtc','tens','zoff','hcal','gmode','ggain'].forEach(i=>document.getElementById(i).addEventListener('change',save));
function fill(){const s=S.settings,p=s.policy;setv('policy',p);const ck=document.getElementById('ckpt');ck.innerHTML='';
 (S.ckpts[p]||[]).forEach(c=>{const o=document.createElement('option');o.value=c;o.textContent=c.split('/').slice(-3).join('/');ck.appendChild(o)});
 if(![...ck.options].some(o=>o.value===s['ckpt_'+p])){const o=document.createElement('option');o.value=s['ckpt_'+p];o.textContent=s['ckpt_'+p].split('/').slice(-3).join('/');ck.appendChild(o)}
 ck.value=s['ckpt_'+p];const cm=document.getElementById('commit');cm.innerHTML='';[1,2,4,8,12,16,24,32].forEach(v=>{const o=document.createElement('option');o.value=v;o.textContent=v;cm.appendChild(o)});
 cm.value=s['commit_'+p];setv('speed',s.speed);setv('slow',s.slow_start_speed);setv('secs',s.secs);setv('dsteps',s.dit_steps);umi.checked=!!s.show_umi_ref;setv('arms',s.active_arms||'both');gfo.checked=!!s.grip_fake_open;document.getElementById('tens').checked=!!s.tens;document.getElementById('gmode').value=s.grip_mode||(s.grip_cont===false?'binary':'linear');document.getElementById('ggain').value=s.grip_gain||10;document.getElementById('zoff').value=s.z_offset_mm||0;document.getElementById('hcal').checked=!!s.height_calib;document.getElementById('rtc').value=s.rtc||'blend';inited=true}
function c(v,good){return v?'<span class="'+(good?'ok':'bad')+'">':'<span class="'+(good?'bad':'ok')+'">'}
function strip(g){const cv=document.getElementById('strip'),x=cv.getContext('2d');x.clearRect(0,0,820,120);if(!g)return;
 ['L','R'].forEach((n,j)=>{x.fillStyle='#ccc';x.fillText(n,4,30+j*55);for(let k=0;k<32;k++){const v=g[k][j];const y=8+j*55;
  x.fillStyle=v>0.5?'#3c3':'rgb('+Math.round(40+180*Math.max(0,Math.min(1,v)))+',60,60)';x.fillRect(20+k*25,y,22,40);x.fillStyle='#fff';x.font='10px sans-serif';x.fillText(v.toFixed(2),21+k*25,y+25)}})}
function hist(h){const cv=document.getElementById('hist'),x=cv.getContext('2d');x.clearRect(0,0,820,160);if(!h||h.length<2)return;
 const t0=h[0][0],t1=h[h.length-1][0]||1;const X=t=>30+(t-t0)/Math.max(t1-t0,1e-3)*780,Y=v=>150-Math.max(-0.2,Math.min(1.2,v))/1.4*140-0.2/1.4*140*0;
 x.strokeStyle='#666';x.beginPath();x.moveTo(30,Y(0.5));x.lineTo(810,Y(0.5));x.stroke();x.fillStyle='#999';x.fillText('0.5',4,Y(0.5)+4);x.fillText('k1 이력',4,12);
 [['#4af',1],['#fa4',2]].forEach(([col,i])=>{x.strokeStyle=col;x.beginPath();h.forEach((p,n)=>{n?x.lineTo(X(p[0]),Y(p[i])):x.moveTo(X(p[0]),Y(p[i]))});x.stroke()})}
async function tick(){try{S=await (await fetch('/api/status')).json()}catch(e){return}
 if(!inited)fill();const b=S.bridge,m=S.mode;
 const ra=b.read_age_ms||{},mo=b.motors_ok||{};
 document.getElementById('status').innerHTML=
  '<span><span class="k">모드</span> <b>'+({idle:'대기',dry:'DRY-RUN',go:(S.step_mode?'실행(GO, 스텝)':'실행(GO)'),hold:'HOLD'}[m]||m)+'</b>'+(S.busy?' ('+S.busy+')':'')+'</span>'+
  '<span><span class="k">계획 Hz</span> '+(S.hz??'-')+'</span><span><span class="k">정책 지연</span> '+(S.lat_pol_ms??'-')+' ms</span><span><span class="k">루프</span> '+(S.lat_total_ms??'-')+' ms</span>'+
  '<span><span class="k">브리지</span> '+c(b.connected,true)+(b.connected?'연결':'끊김')+'</span> E-STOP '+c(b.estop,false)+(b.estop?'걸림':'없음')+'</span> park '+(b.parked&&b.parked.length?'<span class="warn">'+b.parked.join(',')+'</span>':'없음')+'</span>'+
  '<span><span class="k">읽기 지연</span> L '+(ra.left??'-')+' / R '+(ra.right??'-')+' ms</span>'+
  '<span><span class="k">모터</span> '+Object.entries(mo).map(([k,v])=>c(v,true)+k.replace('_arm','')+(v?' OK':' 이상')+'</span>').join(' ')+'</span>'+
  '<span><span class="k">TCP</span> L '+(S.tcp?S.tcp.left.join(', '):'-')+' | R '+(S.tcp?S.tcp.right.join(', '):'-')+'</span>'+
  (S.xvla?(S.xvla.up?'<span class="ok">X-VLA 서버 OK</span>':'<span class="bad">X-VLA 서버 꺼짐 (:8775)</span>'):'')+(S.replay?'<span class="bad">녹화 프레임(replay) 모드: GO 불가</span>':'')+(S.dry_only?'<span class="warn">DRY 전용 UI</span>':'')+(S.wave_ok?'':'<span class="bad">wave 미확인</span>');
 document.getElementById('flags').innerHTML=(S.flags||[]).map(f=>'<span>'+f+'</span>').join('')||'<span style="background:#223a22">없음</span>';
 document.getElementById('lims').textContent=(S.lims&&S.lims.length)?'제한기: '+S.lims.join(' '):'';
 document.getElementById('msg').textContent=S.msg;strip(S.grip);hist(S.k1);
 const running=m==='dry'||m==='go',idle=!running&&!S.busy;
 document.getElementById('dry').disabled=!idle;document.getElementById('go').disabled=!idle||S.dry_only||S.replay||!S.wave_ok;
 document.getElementById('gostep').disabled=!idle||S.dry_only||S.replay||!S.wave_ok;
 document.getElementById('step').disabled=!(m==='go'&&S.step_mode)||S.step_pending;
 document.getElementById('stepinfo').textContent=(m==='go'&&S.step_mode)?((S.step_pending?'실행 대기 중... ':'다음 스텝 대기 ')+'(실행한 스텝 '+S.steps_done+')'):'';
 document.getElementById('ready').disabled=!idle||S.dry_only;document.getElementById('park').disabled=!idle||S.dry_only;document.getElementById('startpose').disabled=!idle||S.dry_only;
 ['policy','ckpt','commit','speed','slow','secs','dsteps','arms'].forEach(i=>document.getElementById(i).disabled=running)}
async function runs(){const r=await (await fetch('/api/runs')).json();document.getElementById('runs').innerHTML='<table>'+r.map(x=>'<tr><td>'+x.name+'</td><td>'+(x.stop||'')+'</td><td><a href="/file?p='+encodeURIComponent(x.summary)+'" target=_blank>요약</a></td><td>'+(x.video?'<a href="/file?p='+encodeURIComponent(x.video)+'" target=_blank>영상</a>':'')+'</td></tr>').join('')+'</table>'}
setInterval(tick,300);setInterval(runs,5000);tick();runs();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8770); ap.add_argument("--bridge-url", default=None); ap.add_argument("--global-url", default="http://127.0.0.1:8092")
    ap.add_argument("--replay-global", default="", help="TEST ONLY (GO disabled)"); ap.add_argument("--dry-only", action="store_true", help="GO / READY / park disabled")
    a = ap.parse_args()
    from flask import Flask, Response, jsonify, request, send_file
    A = App(a); app = Flask("act_tagrig_ui")
    import logging; logging.getLogger("werkzeug").setLevel(logging.ERROR)

    def mjpeg(fn, fps):
        def gen():
            while True:
                j = fn()
                if j: yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n"
                time.sleep(1.0 / fps)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")

    app.add_url_rule("/", "i", lambda: PAGE)
    app.add_url_rule("/stream/global", "g", lambda: mjpeg(A.global_jpg, 10))
    app.add_url_rule("/stream/left", "l", lambda: mjpeg(lambda: A.wrist_jpg("left"), 8))
    app.add_url_rule("/stream/right", "r", lambda: mjpeg(lambda: A.wrist_jpg("right"), 8))
    app.add_url_rule("/api/status", "st", lambda: jsonify(A.api_status()))

    @app.post("/api/settings")
    def settings():
        j = request.json or {}
        for k, v in j.items():
            if k in DEFAULTS or k.startswith(("ckpt_", "commit_")): A.s[k] = v
        save_settings(A.s); return jsonify(ok=True, msg="설정 저장됨")

    for name, fn in (("dry", lambda: A.start_run(False)), ("go", lambda: A.start_run(True)), ("gostep", lambda: A.start_run(True, step=True)), ("step", A.step),
                     ("hold", A.hold), ("estop", A.estop), ("clear_estop", A.clear_estop), ("ready", A.go_ready), ("start", A.go_start), ("park", A.park)):
        app.add_url_rule(f"/api/{name}", name, (lambda f: lambda: (lambda r: jsonify(ok=r[0], msg=r[1]))(f()))(fn), methods=["POST"])

    @app.post("/api/hb")
    def hb():
        if A.ctl is not None: A.ctl.hb = time.monotonic()
        return jsonify(ok=True)

    @app.get("/api/runs")
    def runs():
        out = []
        for d in sorted(dep.OUT.glob("*_*_2*"), key=lambda d: d.stat().st_mtime, reverse=True)[:40]:
            f = d / "summary.json"
            if not f.exists(): continue
            try: j = json.loads(f.read_text())
            except Exception: continue
            out.append(dict(name=d.name, stop=j.get("stop_reason"), summary=str(f), video=j.get("video")))
        return jsonify(out)

    @app.get("/file")
    def file():
        p = pathlib.Path(request.args.get("p", "")).resolve()
        if not (str(p).startswith(str(dep.OUT.resolve())) or (p.parent == dep.DESKTOP.resolve() and p.name.startswith("act_"))) or not p.exists(): return "없음", 404
        return send_file(str(p))

    print(f"[UI] http://localhost:{a.port}/  (브리지 {A.rig.url}{', DRY 전용' if a.dry_only else ''})", flush=True)
    app.run(host="127.0.0.1", port=a.port, threaded=True)


if __name__ == "__main__":
    main()
