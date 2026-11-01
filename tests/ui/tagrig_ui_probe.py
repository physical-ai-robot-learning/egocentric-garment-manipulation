"""Probe of the tag-rig deploy UI and its deploy core (garment_pipeline.eval_deploy.act_tagrig_ui / _deploy / _core): every
route of the Flask app, the persisted settings, the deploy CLI defaults and the pure helpers, with the rig (cameras, Orbbec,
robot poller), the bridge and the network stubbed and an empty HOME, so the output is the same on any machine.
Used by tests/test_ui_routes.py; it runs in its own interpreter because the UI reads paths and env at import.

usage: python tests/ui/tagrig_ui_probe.py <repo root> <out.json>      (needs flask, cv2, numpy, scipy)"""
import argparse, hashlib, json, os, sys, tempfile, types

repo, out = sys.argv[1], sys.argv[2]
TMP = os.path.realpath(tempfile.mkdtemp())
os.environ["HOME"] = TMP
for k in ("EGO_DATA_ROOT", "EGO_TRAIN_ROOT", "EGO_CKPT_ROOT", "EGO_COCKPIT_ROOT", "EGO_DESKTOP"):
    os.environ[k] = os.path.join(TMP, k.lower())
os.makedirs(os.environ["EGO_DESKTOP"], exist_ok=True)
SRC = os.path.realpath(os.path.join(repo, "src"))
sys.path.insert(0, SRC)

import requests
def _no_net(self, method, url, *a, **k): raise requests.ConnectionError(f"stubbed: {method} {url}")
requests.sessions.Session.request = _no_net
import numpy as np
import flask

from garment_pipeline.eval_deploy import act_tagrig_core as core
from garment_pipeline.eval_deploy import act_tagrig_deploy as D
from garment_pipeline.eval_deploy import act_tagrig_ui as U

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
R["files"] = norm(dict(ui_settings=U.SETTINGS, cams=D.CAMS_CFG, start=D.START_FILE, out=D.OUT, desktop=D.DESKTOP, uvc=D.UVC,
                       here_ui=U.HERE, here_dep=D.HERE, robot_cam=core.ROBOT_CAM, robot_cam_ext=core.ROBOT_CAM_EXT,
                       ckpt=core.CKPT, dit_ckpt=core.DIT_CKPT, ck_dirs=U.CK_DIRS, xvla_ft=U.XVLA_FT))
# ---- settings persistence (on a scratch copy of the settings file)
smod = sys.modules[U.save_settings.__module__]
smod.SETTINGS = type(U.SETTINGS)(os.path.join(TMP, "act_tagrig_ui_settings.json"))
R["defaults"] = norm(U.DEFAULTS)
R["load_settings_nofile"] = norm(U.load_settings())
smod.SETTINGS.write_text('{"speed": 0.9, "policy": "dit", "extra_key": 3}')
R["load_settings_file"] = norm(U.load_settings())
U.save_settings({"a": 1, "한글": "값", "f": 0.25}); R["save_settings_text"] = smod.SETTINGS.read_text()
smod.SETTINGS.write_text("{broken"); R["load_settings_broken"] = norm(U.load_settings())
smod.SETTINGS.unlink()
R["ckpts"] = {k: U.ckpts(k) for k in U.CK_DIRS}
R["hb_timeout"] = U.HB_TIMEOUT_S

# ---- the deploy CLI as the UI sees it (single source of the run defaults) and every sub-command's defaults
p = U.dep_parser()
R["run_defaults"] = norm(vars(p.parse_args(["run"])))
for sub in ("cams", "wave", "start"):
    R[f"{sub}_defaults"] = norm(vars(p.parse_args([sub])))

# ---- run control object
c = U.UICtl(True)
R["uictl"] = [c.take_step(), c.poll()]
c.step_n = 1; R["uictl"] += [c.take_step(), c.steps_done, c.step_n]
c.hb -= 10; R["uictl"].append(c.poll())
c.publish({"t": 1.5, "grip": np.array([[0.2, 0.8]])}); R["uictl"].append(norm(c.k1))

# ---- the Flask app over a stubbed App
class NS(types.SimpleNamespace): pass
class FakeBr:
    def __init__(self): self.s = requests.Session(); self.estops = 0
    def estop(self): self.estops += 1
rig = NS(url="http://127.0.0.1:8031", replay=False, wave_ok=False, cfg={s: {"location": "0x0000"} for s in core.SIDES},
         gl=NS(n=0, ring=NS(latest=lambda: (0, None))), wr={s: NS(n=0, raw=None) for s in core.SIDES},
         rob=NS(ring=NS(latest=lambda: (0, (None, None, None)))), arms=None)
RealApp = U.App
A = object.__new__(RealApp)
import threading
A.a = argparse.Namespace(dry_only=True, replay_global="", port=8770, bridge_url=None, global_url="http://127.0.0.1:8092")
A.s = U.load_settings(); A.rig = rig; A.lock = threading.Lock(); A.ctl = None; A.thread = None; A.mode = "idle"; A.msg = "준비됨"
A.last = None; A.busy = None; A.status = {}; A.motors = {}; A.br_ro = FakeBr(); A.umi_ref = {}; A.jaw = {}

class Captured(Exception): pass
g = U.main.__globals__
saved = (g["App"], flask.Flask.run, sys.argv)
def _cap(self, *a, **k): raise Captured(self)
g["App"] = lambda a: A; flask.Flask.run = _cap; sys.argv = ["act_tagrig_ui.py", "--dry-only"]
try:
    U.main(); raise SystemExit("app.run not reached")
except Captured as e:
    app = e.args[0]
finally:
    g["App"], flask.Flask.run, sys.argv = saved

R["url_map"] = sorted((r.rule, r.endpoint, sorted(r.methods)) for r in app.url_map.iter_rules())
R["app_name"] = app.name
cl = app.test_client()
page = cl.get("/").data
R["page_sha256"], R["page_len"] = sha(page), len(page)
def J(r): return [r.status_code, norm(r.get_json(silent=True)) if r.is_json else sha(r.data)]
R["GET /api/status"] = J(cl.get("/api/status"))
R["POST /api/settings"] = J(cl.post("/api/settings", json={"speed": 0.7, "ckpt_dit": "/x/ckpt", "commit_xvla": 5, "bogus": 1, "rtc": "nolook"}))
R["settings_file_after_post"] = norm(json.loads(smod.SETTINGS.read_text()))
R["A.s_after_post"] = norm(A.s)
for name in ("hb", "step", "hold", "estop", "clear_estop", "go", "gostep", "ready", "start", "park"):
    R[f"POST /api/{name}"] = J(cl.post(f"/api/{name}"))
R["after_posts"] = dict(mode=A.mode, msg=A.msg, estops=A.br_ro.estops)
A.busy = "READY로 이동"; R["POST /api/dry busy"] = J(cl.post("/api/dry")); A.busy = None
run = D.OUT / "act_dry_20261101_120000"; run.mkdir(parents=True)
(run / "summary.json").write_text(json.dumps({"stop_reason": "max time 60s", "video": None}))
R["GET /api/runs"] = J(cl.get("/api/runs"))
R["GET /file summary"] = J(cl.get("/file", query_string={"p": str(run / "summary.json")}))
R["GET /file outside"] = [cl.get("/file", query_string={"p": "/etc/hosts"}).status_code, cl.get("/file").status_code]
c2 = U.UICtl(True); c2.step_mode = True; A.ctl = c2; A.mode = "go"
c2.publish(dict(t=2.0, grip=np.zeros((32, 2)) + 0.4, hz=4.0, lat_pol_ms=120.0, lat_total_ms=150.0, k0=1, flags=["LIM:tcp_speed_l", "WARN:x"],
                speed=0.5, g_cmd={"left": 0.0, "right": 1.0}, Tc={s: [np.eye(4)] * 32 for s in core.SIDES}))
R["GET /api/status running"] = J(cl.get("/api/status"))
R["POST /api/step running"] = J(cl.post("/api/step"))
R["POST /api/hold running"] = J(cl.post("/api/hold"))
R["after_hold"] = dict(mode=A.mode, stop_req=c2.stop_req)

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
    try: D.check_run_args(a); R[f"check_run_args {i}"] = ["ok", a.commit]
    except SystemExit as e: R[f"check_run_args {i}"] = ["exit", str(e)]
ens = [(0.05 * i, {s: [T() for _ in range(32)] for s in core.SIDES}) for i in range(4)]
R["ensemble"] = [np.round(D.ensemble(ens, 0.4, s, 0.01), 10).tolist() for s in core.SIDES]
clock = [100.0]
smod_t = sys.modules[D.Streamer.__module__]
real_time = smod_t.time
smod_t.time = types.SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda s: None, monotonic_ns=lambda: int(clock[0] * 1e9))
try:
    q0 = {s: np.zeros(6) for s in core.SIDES}
    st = D.Streamer(None, q0, {"left": 0.0, "right": 1.0}, 30.0, dt=0.05)
    rows = [{s: np.full(6, 0.01 * (j + 1)) for s in core.SIDES} for j in range(8)]
    st.set(rows[:4], 0.05, {"left": 1.0, "right": 1.0}); clock[0] += 0.12
    R["streamer_anchor"] = norm(st.anchor(clock[0]))
    R["streamer_merge"] = list(st.merge(st.executed(), rows, {"left": 0.0, "right": 0.0}))
    clock[0] += 0.07
    R["streamer_q"] = norm(st._q()); R["streamer_tl"] = sorted(st.tl); R["streamer_n_dry"] = st.n_dry
finally:
    smod_t.time = real_time
ring = D.Ring(4)
for i in range(6): ring.put(int(i * 1e8), i)
R["ring"] = [ring.latest(), ring.near(int(2.6e8), 0.05), ring.near(int(2.6e8), 0.01)]
R["topdown"] = sha(D.topdown({s: T() for s in core.SIDES}, {s: [T() for _ in range(32)] for s in core.SIDES},
                             {s: (np.array([-0.1, -0.5, 0.0]), np.array([0.6, 0.5, 0.4])) for s in core.SIDES}, rng.normal(size=(20, 3)) * 0.2).tobytes())
R["core.rtc_weights"] = [core.rtc_weights(32, d, n).tolist() for d, n in ((2, 32), (5, 20), (0, 0), (40, 32))]
ch = rng.normal(size=(32, 20)) * 0.01
R["core.retime"] = np.round(core.retime(ch, core.DT, 0.5), 10).tolist()
Tl, Tr = T(), T()
dec = core.decode(np.c_[np.tile(np.r_[0.01, 0, 0, 1, 0, 0, 0, 1, 0], (32, 2)), np.full((32, 2), 0.5)], Tl, Tr)
R["core.decode"] = norm({s: np.round(np.array(dec[s]), 10) for s in core.SIDES})
R["core.build_state"] = np.round(core.build_state(Tl, Tr, T(), T(), 1.0, 0.0), 10).tolist()
Ra, Rb = T()[:3, :3], T()[:3, :3]
R["core.rot"] = [np.round(core.rot_interp(Ra, Rb, 0.3), 12).tolist(), round(core.rot_angle_deg(Ra, Rb), 9)]
K = {"fx": 600.0, "fy": 600.0, "cx": 424.0, "cy": 240.0}
R["core.project"] = norm(np.round(core.project(np.eye(4), K, np.array([[0.1, 0.0, 1.0], [0.0, 0.1, 0.01], [0.2, -0.1, 2.0]])), 9))
R["core.seg_dist"] = round(float(core.seg_dist(np.zeros(3), np.array([1.0, 0, 0]), np.array([0.5, 1.0, 0]), np.array([0.5, 1.0, 1.0]))), 12)
img = (rng.random((480, 848, 3)) * 255).astype(np.uint8)
R["core.prep_global"] = sha(core.prep_global(img).tobytes())
R["core.prep_wrist"] = sha(core.prep_wrist((rng.random((1080, 1920, 3)) * 255).astype(np.uint8)).tobytes())
R["core.cv_plot"] = sha(core.cv_plot([(np.arange(10.0), np.sin(np.arange(10.0)), (0, 0, 0), "a")], (-1.2, 1.2), "t", hline=0.5).tobytes())
R["core.consts"] = norm(dict(IMG_WH=core.IMG_WH, CAMS=core.CAMS, TASK=core.TASK, SIDES=core.SIDES, GI=core.GI, DT=core.DT,
                             SL={k: [v.start, v.stop] for k, v in core.SL.items()}, ROBOTIZE_ARGS=core.ROBOTIZE_ARGS, CAPSULES=core.CAPSULES,
                             FINGER_R=core.FINGER_R, TABLE_R_WRIST=core.TABLE_R_WRIST, GRIP_OPEN_FRAC_MAX=D.GRIP_OPEN_FRAC_MAX,
                             HIST_TOL_S=D.HIST_TOL_S, COL=D.COL))
def api(m):     # the module's own names: no imported modules, no classes / functions of other libraries
    return sorted(n for n, v in vars(m).items() if not n.startswith("_") and not isinstance(v, types.ModuleType) and n != "annotations"
                  and str(getattr(v, "__module__", None) or "garment_pipeline").startswith(("garment_pipeline", "contracts", "egogarment")))
R["names"] = dict(deploy=api(D), core=api(core), ui=api(U))      # the test accepts additions (the shims may add names)
json.dump(R, open(out, "w"), indent=1, sort_keys=True, ensure_ascii=False, default=str)
print("ok", len(R), "entries; page", R["page_len"], R["page_sha256"][:12])
