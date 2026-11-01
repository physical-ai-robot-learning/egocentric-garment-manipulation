"""One run on a started Rig (dry or --go): the receding-horizon loop -- observe, infer, re-anchor, limiter -> IK -> safety,
commit -- and the ACT temporal ensemble. The stages are described in the docstring of cli.py (= `act_tagrig_deploy.py --help`).
"""
from __future__ import annotations
import collections
import json
import time

import numpy as np

from garment_pipeline.eval_deploy import act_tagrig_core as core
from garment_pipeline.eval_deploy.tagrig_deploy.bridge import check_grip_limits, Streamer
from garment_pipeline.eval_deploy.tagrig_deploy.paths import HIST_TOL_S, OUT, say, START_FILE
from garment_pipeline.eval_deploy.tagrig_deploy.preview import Preview
from garment_pipeline.eval_deploy.tagrig_deploy.report import summarize


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
        from garment_pipeline.eval_deploy import height_calib as HC; hcal = HC.load(); z_off_done = False
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


def ensemble(ens, t_target, s, m):
    """ACT temporal ensemble on ABSOLUTE base-frame targets: every buffered chunk's step nearest t_target, weights exp(-m * i), i = 0 oldest"""
    from scipy.spatial.transform import Rotation as Rot
    P, R, W = [], [], []
    for i, (tp, Tc) in enumerate(ens):
        k = int(round((t_target - tp) / core.DT)) - 1
        if 0 <= k < 32: P.append(Tc[s][k][:3, 3]); R.append(Tc[s][k][:3, :3]); W.append(np.exp(-m * len(W)))
    W = np.array(W) / np.sum(W); T = np.eye(4); T[:3, 3] = np.sum(np.array(P) * W[:, None], 0); T[:3, :3] = Rot.from_matrix(np.array(R)).mean(weights=W).as_matrix(); return T
