"""Run control behind the web UI: the per-run control object (stop request, heartbeat, step tokens, latest snapshot), the App
(bridge status poller, heartbeat watchdog, runs and motions, the live views and /api/status) and the READY move.
"""
from __future__ import annotations
import threading
import time
import traceback

import cv2
import numpy as np

from garment_pipeline.eval_deploy import tagrig_deploy as dep
from garment_pipeline.eval_deploy import act_tagrig_core as core
from garment_pipeline.eval_deploy.tagrig_ui.settings import CK_DIRS, ckpts, HB_TIMEOUT_S, load_settings, save_settings


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
