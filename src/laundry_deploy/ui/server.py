"""[2026-10-23] Inference web UI for the tag-rig laundry policies on the bimanual reBot (http://localhost:8770, Korean UI).
Built on the deploy session: the SAME Rig (cameras, live Orbbec, robot poller, kinematics) and run_session (receding
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
import sys
import time

from ..core.errors import DeployError
from ..session.paths import DESKTOP, RUN_OUT_DIR
from .app import UiApp
from .settings import DEFAULTS, saveSettings

APP_NAME = "act_tagrig_ui"  # Flask import name (as before the split)
PAGE_FILE = pathlib.Path(__file__).with_name("page.html")
HOST = "127.0.0.1"
DEFAULT_PORT = 8770
GLOBAL_FPS, WRIST_FPS = 10, 8
MAX_RUNS_LISTED = 40
RUN_GLOB = "*_*_2*"  # <policy>_<dry|go>_<yyyymmdd_...> (and mock_...)
SETTINGS_PREFIXES = ("ckpt_", "commit_")
DESKTOP_FILE_PREFIX = "act_"
HTTP_NOT_FOUND = 404


def loadPage() -> str:
    """The UI page (Korean, HTML + JS) served by GET /, byte-identical to the string that was inlined in act_tagrig_ui.py."""
    return PAGE_FILE.read_bytes().decode("utf-8")


def createApp(app: UiApp):
    """The Flask app over a started UiApp: the page, the MJPEG views and /api/* (rules, endpoints and JSON as always)."""
    import logging

    from flask import Flask, Response, jsonify, request, send_file

    tFlask = Flask(APP_NAME)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    tPage = loadPage()

    def mjpeg(frameFn, fps):
        def gen():
            while True:
                tJpg = frameFn()
                if tJpg:
                    yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + tJpg + b"\r\n"
                time.sleep(1.0 / fps)

        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")

    tFlask.add_url_rule("/", "i", lambda: tPage)
    tFlask.add_url_rule("/stream/global", "g", lambda: mjpeg(app.globalJpg, GLOBAL_FPS))
    tFlask.add_url_rule("/stream/left", "l", lambda: mjpeg(lambda: app.wristJpg("left"), WRIST_FPS))
    tFlask.add_url_rule("/stream/right", "r", lambda: mjpeg(lambda: app.wristJpg("right"), WRIST_FPS))
    tFlask.add_url_rule("/api/status", "st", lambda: jsonify(app.apiStatus()))

    @tFlask.post("/api/settings")
    def settings():
        tBody = request.json or {}
        for tKey, tValue in tBody.items():
            if tKey in DEFAULTS or tKey.startswith(SETTINGS_PREFIXES):
                app.mSettings[tKey] = tValue
        saveSettings(app.mSettings)
        return jsonify(ok=True, msg="설정 저장됨")

    def _route(fn):
        def view():
            tOk, tMsg = fn()
            return jsonify(ok=tOk, msg=tMsg)

        return view

    tActions = (
        ("dry", lambda: app.startRun(False)),
        ("go", lambda: app.startRun(True)),
        ("gostep", lambda: app.startRun(True, step=True)),
        ("step", app.step),
        ("hold", app.hold),
        ("estop", app.estop),
        ("clear_estop", app.clearEstop),
        ("ready", app.goReady),
        ("start", app.goStart),
        ("park", app.park),
    )
    for tName, tFn in tActions:
        tFlask.add_url_rule(f"/api/{tName}", tName, _route(tFn), methods=["POST"])

    @tFlask.post("/api/hb")
    def hb():
        if app.mCtl is not None:
            app.mCtl.mHb = time.monotonic()
        return jsonify(ok=True)

    @tFlask.get("/api/runs")
    def runs():
        tOut = []
        for tDir in sorted(RUN_OUT_DIR.glob(RUN_GLOB), key=lambda tD: tD.stat().st_mtime, reverse=True)[:MAX_RUNS_LISTED]:
            tFile = tDir / "summary.json"
            if not tFile.exists():
                continue
            try:
                tJ = json.loads(tFile.read_text())
            except Exception:
                continue
            tOut.append(dict(name=tDir.name, stop=tJ.get("stop_reason"), summary=str(tFile), video=tJ.get("video")))
        return jsonify(tOut)

    @tFlask.get("/file")
    def file():
        tPath = pathlib.Path(request.args.get("p", "")).resolve()
        tAllowed = str(tPath).startswith(str(RUN_OUT_DIR.resolve())) or (
            tPath.parent == DESKTOP.resolve() and tPath.name.startswith(DESKTOP_FILE_PREFIX)
        )
        if not tAllowed or not tPath.exists():
            return "없음", HTTP_NOT_FOUND
        return send_file(str(tPath))

    return tFlask


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tParser.add_argument("--port", type=int, default=DEFAULT_PORT)
    tParser.add_argument("--bridge-url", default=None)
    tParser.add_argument("--global-url", default="http://127.0.0.1:8092")
    tParser.add_argument("--replay-global", default="", help="TEST ONLY (GO disabled)")
    tParser.add_argument("--dry-only", action="store_true", help="GO / READY / park disabled")
    return tParser


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    import flask  # noqa: F401 -- flask is checked before the rig (cameras, robot poller) opens

    try:
        tApp = UiApp(tArgs)
    except DeployError as tExc:
        print(tExc, file=sys.stderr)
        return 1
    tFlask = createApp(tApp)
    print(f"[UI] http://localhost:{tArgs.port}/  (브리지 {tApp.mRig.mUrl}{', DRY 전용' if tArgs.dry_only else ''})", flush=True)
    tFlask.run(host=HOST, port=tArgs.port, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
