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
import argparse
import json
import pathlib
import time

from garment_pipeline.eval_deploy import tagrig_deploy as dep
from garment_pipeline.eval_deploy.tagrig_ui.control import App
from garment_pipeline.eval_deploy.tagrig_ui.page import PAGE
from garment_pipeline.eval_deploy.tagrig_ui.settings import DEFAULTS, save_settings


def create_app(A):
    """the Flask app over a started App: the page, the MJPEG views and /api/* (rules, endpoints and JSON as before the split)"""
    from flask import Flask, Response, jsonify, request, send_file
    app = Flask("act_tagrig_ui")
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

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8770); ap.add_argument("--bridge-url", default=None); ap.add_argument("--global-url", default="http://127.0.0.1:8092")
    ap.add_argument("--replay-global", default="", help="TEST ONLY (GO disabled)"); ap.add_argument("--dry-only", action="store_true", help="GO / READY / park disabled")
    a = ap.parse_args()
    import flask  # noqa: F401 -- flask is checked before the rig (cameras, robot poller) opens, as before the split
    A = App(a); app = create_app(A)
    print(f"[UI] http://localhost:{a.port}/  (브리지 {A.rig.url}{', DRY 전용' if a.dry_only else ''})", flush=True)
    app.run(host="127.0.0.1", port=a.port, threaded=True)


if __name__ == "__main__":
    main()
