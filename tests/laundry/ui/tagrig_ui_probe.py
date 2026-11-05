"""Probe of the tag-rig deploy UI and its deploy core (laundry_deploy.ui / .session / .core): every route of the Flask app, the
persisted settings, the deploy CLI defaults and the pure helpers, with the rig (cameras, Orbbec, robot poller), the bridge and the
network stubbed and an empty HOME, so the output is the same on any machine.
Used by tests/laundry/test_ui_routes.py; it runs in its own interpreter because the UI reads paths and env at import.

usage: python tests/laundry/ui/tagrig_ui_probe.py <repo root> <out.json>      (needs flask, cv2, numpy, scipy)"""
import argparse, hashlib, json, os, pathlib, sys, tempfile, types

repo, out = sys.argv[1], sys.argv[2]
TMP = os.path.realpath(tempfile.mkdtemp())
os.environ["HOME"] = TMP
for k in ("EGO_DATA_ROOT", "EGO_TRAIN_ROOT", "EGO_CKPT_ROOT", "EGO_COCKPIT_ROOT", "EGO_DESKTOP"):
    os.environ[k] = os.path.join(TMP, k.lower())
os.environ.pop("TAGRIG_STATE_DIR", None)       # default state folder, as in the golden record
os.makedirs(os.environ["EGO_DESKTOP"], exist_ok=True)
SRC = os.path.realpath(os.path.join(repo, "src"))
sys.path.insert(0, SRC)

import requests
def _no_net(self, method, url, *a, **k): raise requests.ConnectionError(f"stubbed: {method} {url}")
requests.sessions.Session.request = _no_net
import numpy as np
import flask

from laundry_deploy.core import camera_model, chunk, collision, contract, images, kinematics, plot
from laundry_deploy.core.errors import DeployError
from laundry_deploy.session import bridge, cli, paths, report, ring, run_loop, streamer
from laundry_deploy.ui import app as uiApp, run_control, server, settings

SIDES = contract.SIDES
R = {}
def norm(o):
    """paths -> <SRC>/<TMP> placeholders; tuples -> lists"""
    if isinstance(o, dict): return {str(k): norm(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [norm(v) for v in o]
    if isinstance(o, str): return o.replace(SRC, "<SRC>").replace(TMP, "<TMP>")
    if hasattr(o, "__fspath__"): return norm(str(o))
    if isinstance(o, np.ndarray): return norm(o.tolist())
    if isinstance(o, (np.floating, np.integer)): return o.item()
    return o
def sha(b): return hashlib.sha256(b).hexdigest()

# ---- where the working files are (must not move: persisted UI settings, camera map, start pose)
R["files"] = norm(dict(ui_settings=settings.SETTINGS_FILE, cams=paths.CAMS_FILE, start=paths.START_FILE, out=paths.RUN_OUT_DIR, desktop=paths.DESKTOP,
                       uvc=paths.UVC_UTIL, here_ui=pathlib.Path(uiApp.__file__).parent, here_dep=pathlib.Path(run_loop.__file__).parent,
                       robot_cam=camera_model.ROBOT_CAM_FILE, robot_cam_ext=camera_model.ROBOT_CAM_EXT_FILE, ckpt=contract.ACT_CKPT,
                       dit_ckpt=contract.DIT_CKPT, ck_dirs=settings.CKPT_DIRS, xvla_ft=settings.XVLA_FT_CKPTS))
# ---- settings persistence (on a scratch copy of the settings file)
settings.SETTINGS_FILE = type(settings.SETTINGS_FILE)(os.path.join(TMP, "act_tagrig_ui_settings.json"))
R["defaults"] = norm(settings.DEFAULTS)
R["load_settings_nofile"] = norm(settings.loadSettings())
settings.SETTINGS_FILE.write_text('{"speed": 0.9, "policy": "dit", "extra_key": 3}')
R["load_settings_file"] = norm(settings.loadSettings())
settings.saveSettings({"a": 1, "한글": "값", "f": 0.25}); R["save_settings_text"] = settings.SETTINGS_FILE.read_text()
settings.SETTINGS_FILE.write_text("{broken"); R["load_settings_broken"] = norm(settings.loadSettings())
settings.SETTINGS_FILE.unlink()
R["ckpts"] = {k: settings.listCkpts(k) for k in settings.CKPT_DIRS}
R["hb_timeout"] = settings.HB_TIMEOUT_S

# ---- the deploy CLI as the UI sees it (single source of the run defaults) and every sub-command's defaults
p = cli.buildParser()
R["run_defaults"] = norm(vars(p.parse_args(["run"])))
for sub in ("cams", "wave", "start"):
    R[f"{sub}_defaults"] = norm(vars(p.parse_args([sub])))

# ---- run control object
c = run_control.UiRunControl(True)
R["uictl"] = [c.takeStep(), c.poll()]
c.mStepN = 1; R["uictl"] += [c.takeStep(), c.mStepsDone, c.mStepN]
c.mHb -= 10; R["uictl"].append(c.poll())
c.publish({"t": 1.5, "grip": np.array([[0.2, 0.8]])}); R["uictl"].append(norm(c.mK1))

# ---- the Flask app over a stubbed UiApp
class NS(types.SimpleNamespace): pass
class FakeBr:
    def __init__(self): self.s = requests.Session(); self.estops = 0
    def estop(self): self.estops += 1
rig = NS(mUrl="http://127.0.0.1:8031", mReplay=False, mWaveOk=False, mCamsCfg={s: {"location": "0x0000"} for s in SIDES},
         mGlobalCam=NS(mFrames=0, mRing=NS(latest=lambda: (0, None))), mWristCams={s: NS(mFrames=0, mRaw=None) for s in SIDES},
         mPoller=NS(mRing=NS(latest=lambda: (0, (None, None, None)))), mArms=None)
A = object.__new__(uiApp.UiApp)
import threading
A.mArgs = argparse.Namespace(dry_only=True, replay_global="", port=8770, bridge_url=None, global_url="http://127.0.0.1:8092")
A.mSettings = settings.loadSettings(); A.mRig = rig; A.mLock = threading.Lock(); A.mCtl = None; A.mThread = None; A.mMode = "idle"; A.mMsg = "준비됨"
A.mLast = None; A.mBusy = None; A.mBridgeStatus = {}; A.mMotors = {}; A.mXvla = None; A.mBridgeRo = FakeBr(); A.mUmiRef = {}; A.mJaw = {}

class Captured(Exception): pass
saved = (server.UiApp, flask.Flask.run)
def _cap(self, *a, **k): raise Captured(self)
server.UiApp = lambda a: A; flask.Flask.run = _cap
try:
    server.main(["--dry-only"]); raise SystemExit("app.run not reached")
except Captured as e:
    app = e.args[0]
finally:
    server.UiApp, flask.Flask.run = saved

R["url_map"] = sorted((r.rule, r.endpoint, sorted(r.methods)) for r in app.url_map.iter_rules())
R["app_name"] = app.name
cl = app.test_client()
page = cl.get("/").data
R["page_sha256"], R["page_len"] = sha(page), len(page)
def J(r): return [r.status_code, norm(r.get_json(silent=True)) if r.is_json else sha(r.data)]
R["GET /api/status"] = J(cl.get("/api/status"))
R["POST /api/settings"] = J(cl.post("/api/settings", json={"speed": 0.7, "ckpt_dit": "/x/ckpt", "commit_xvla": 5, "bogus": 1, "rtc": "nolook"}))
R["settings_file_after_post"] = norm(json.loads(settings.SETTINGS_FILE.read_text()))
R["A.s_after_post"] = norm(A.mSettings)
for name in ("hb", "step", "hold", "estop", "clear_estop", "go", "gostep", "ready", "start", "park"):
    R[f"POST /api/{name}"] = J(cl.post(f"/api/{name}"))
R["after_posts"] = dict(mode=A.mMode, msg=A.mMsg, estops=A.mBridgeRo.estops)
A.mBusy = "READY로 이동"; R["POST /api/dry busy"] = J(cl.post("/api/dry")); A.mBusy = None
run = paths.RUN_OUT_DIR / "act_dry_20261101_120000"; run.mkdir(parents=True)
(run / "summary.json").write_text(json.dumps({"stop_reason": "max time 60s", "video": None}))
R["GET /api/runs"] = J(cl.get("/api/runs"))
R["GET /file summary"] = J(cl.get("/file", query_string={"p": str(run / "summary.json")}))
R["GET /file outside"] = [cl.get("/file", query_string={"p": "/etc/hosts"}).status_code, cl.get("/file").status_code]
c2 = run_control.UiRunControl(True); c2.mStepMode = True; A.mCtl = c2; A.mMode = "go"
c2.publish(dict(t=2.0, grip=np.zeros((32, 2)) + 0.4, hz=4.0, lat_pol_ms=120.0, lat_total_ms=150.0, k0=1, flags=["LIM:tcp_speed_l", "WARN:x"],
                speed=0.5, g_cmd={"left": 0.0, "right": 1.0}, Tc={s: [np.eye(4)] * 32 for s in SIDES}))
R["GET /api/status running"] = J(cl.get("/api/status"))
R["POST /api/step running"] = J(cl.post("/api/step"))
R["POST /api/hold running"] = J(cl.post("/api/hold"))
R["after_hold"] = dict(mode=A.mMode, stop_req=c2.mStopReq)

# ---- pure helpers on synthetic input
rng = np.random.default_rng(0)
from scipy.spatial.transform import Rotation as Rt
def T():
    m = np.eye(4); m[:3, :3] = Rt.from_rotvec(rng.normal(size=3) * 0.3).as_matrix(); m[:3, 3] = rng.normal(size=3) * 0.2; return m
def a_(**k):
    d = dict(policy="act", commit=None, label="", speed=0.5, go=False, assume_ready=False, assume_start=False, replay_global="")
    d.update(k); return argparse.Namespace(**d)
for i, kw in enumerate([{}, dict(policy="dit"), dict(policy="dit", speed=2.0, commit=16), dict(policy="label"), dict(policy="label", label="x.npz"),
                        dict(commit=40), dict(go=True, assume_ready=True), dict(go=True, replay_global="d")]):
    a = a_(**kw)
    try: run_loop.checkRunArgs(a); R[f"check_run_args {i}"] = ["ok", a.commit]
    except DeployError as e: R[f"check_run_args {i}"] = ["exit", str(e)]
ens = [(0.05 * i, {s: [T() for _ in range(32)] for s in SIDES}) for i in range(4)]
R["ensemble"] = [np.round(run_loop.ensemble(ens, 0.4, s, 0.01), 10).tolist() for s in SIDES]
clock = [100.0]
real_time = streamer.time
streamer.time = types.SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda s: None, monotonic_ns=lambda: int(clock[0] * 1e9))
try:
    q0 = {s: np.zeros(6) for s in SIDES}
    st = streamer.Streamer(None, q0, {"left": 0.0, "right": 1.0}, 30.0, dt=0.05)
    rows = [{s: np.full(6, 0.01 * (j + 1)) for s in SIDES} for j in range(8)]
    st.setTimeline(rows[:4], 0.05, {"left": 1.0, "right": 1.0}); clock[0] += 0.12
    R["streamer_anchor"] = norm(st.anchor(clock[0]))
    R["streamer_merge"] = list(st.merge(st.executed(), rows, {"left": 0.0, "right": 0.0}))
    clock[0] += 0.07
    R["streamer_q"] = norm(st._currentTarget()); R["streamer_tl"] = sorted(st.mTimeline); R["streamer_n_dry"] = st.mDryTicks
finally:
    streamer.time = real_time
rg = ring.Ring(4)
for i in range(6): rg.put(int(i * 1e8), i)
R["ring"] = [rg.latest(), rg.near(int(2.6e8), 0.05), rg.near(int(2.6e8), 0.01)]
R["topdown"] = sha(report.topDownView({s: T() for s in SIDES}, {s: [T() for _ in range(32)] for s in SIDES},
                                      {s: (np.array([-0.1, -0.5, 0.0]), np.array([0.6, 0.5, 0.4])) for s in SIDES}, rng.normal(size=(20, 3)) * 0.2).tobytes())
R["core.rtc_weights"] = [chunk.rtcWeights(32, d, n).tolist() for d, n in ((2, 32), (5, 20), (0, 0), (40, 32))]
ch = rng.normal(size=(32, 20)) * 0.01
R["core.retime"] = np.round(chunk.retime(ch, contract.ROW_DT_S, 0.5), 10).tolist()
Tl, Tr = T(), T()
dec = chunk.decodeChunk(np.c_[np.tile(np.r_[0.01, 0, 0, 1, 0, 0, 0, 1, 0], (32, 2)), np.full((32, 2), 0.5)], Tl, Tr)
R["core.decode"] = norm({s: np.round(np.array(dec[s]), 10) for s in SIDES})
R["core.build_state"] = np.round(chunk.buildState(Tl, Tr, T(), T(), 1.0, 0.0), 10).tolist()
Ra, Rb = T()[:3, :3], T()[:3, :3]
R["core.rot"] = [np.round(kinematics.rotInterp(Ra, Rb, 0.3), 12).tolist(), round(kinematics.rotAngleDeg(Ra, Rb), 9)]
K = {"fx": 600.0, "fy": 600.0, "cx": 424.0, "cy": 240.0}
R["core.project"] = norm(np.round(camera_model.project(np.eye(4), K, np.array([[0.1, 0.0, 1.0], [0.0, 0.1, 0.01], [0.2, -0.1, 2.0]])), 9))
R["core.seg_dist"] = round(float(collision.segDist(np.zeros(3), np.array([1.0, 0, 0]), np.array([0.5, 1.0, 0]), np.array([0.5, 1.0, 1.0]))), 12)
img = (rng.random((480, 848, 3)) * 255).astype(np.uint8)
R["core.prep_global"] = sha(images.prepGlobal(img).tobytes())
R["core.prep_wrist"] = sha(images.prepWrist((rng.random((1080, 1920, 3)) * 255).astype(np.uint8)).tobytes())
R["core.cv_plot"] = sha(plot.cvPlot([(np.arange(10.0), np.sin(np.arange(10.0)), (0, 0, 0), "a")], (-1.2, 1.2), "t", hline=0.5).tobytes())
R["core.consts"] = norm(dict(IMG_WH=contract.IMG_WH, CAMS=contract.CAMS, TASK=contract.TASK, SIDES=contract.SIDES, GI=contract.GRIP_INDEX,
                             DT=contract.ROW_DT_S, SL={k: [v.start, v.stop] for k, v in contract.POSE_SLICE.items()},
                             ROBOTIZE_ARGS=kinematics.ROBOTIZE_ARGS, CAPSULES=collision.CAPSULES, FINGER_R=collision.FINGER_RADIUS_M,
                             TABLE_R_WRIST=collision.WRIST_TABLE_RADIUS_M, GRIP_OPEN_FRAC_MAX=bridge.GRIP_OPEN_FRAC_MAX,
                             HIST_TOL_S=paths.HIST_TOL_S, COL=report.ARM_COLOURS_BGR))
def api(*mods):     # the modules' own names: no imported modules, no classes / functions of other libraries
    return sorted({n for m in mods for n, v in vars(m).items() if not n.startswith("_") and not isinstance(v, types.ModuleType) and n != "annotations"
                   and str(getattr(v, "__module__", None) or "laundry_deploy").startswith(("laundry_deploy", "laundry_common", "laundry_shared.contracts"))
                   and str(getattr(v, "__module__", None) or m.__name__) == m.__name__})
from laundry_deploy.session import camera_commands, cameras, keys, preview, rig as rigMod, start_pose
from laundry_deploy.core import label_runner, policy_runner, xvla_runner, height_calib
from laundry_deploy.ui import ready_move
R["names"] = dict(session=api(paths, ring, cameras, camera_commands, bridge, streamer, keys, preview, rigMod, run_loop, start_pose, report, cli),
                  core=api(contract, images, chunk, kinematics, collision, camera_model, plot, height_calib, label_runner, policy_runner, xvla_runner),
                  ui=api(settings, run_control, uiApp, ready_move, server))   # the test accepts additions
json.dump(R, open(out, "w"), indent=1, sort_keys=True, ensure_ascii=False, default=str)
print("ok", len(R), "entries; page", R["page_len"], R["page_sha256"][:12])
