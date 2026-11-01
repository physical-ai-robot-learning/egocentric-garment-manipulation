"""Probe of the 11-01 height-calibration path of the tag-rig deploy (--height-calib), robot-free and deterministic:
  start_cmd (dry, no --go) from a stub label start pose, with / without --height-calib -> the start TCP target shifts by
    -xy_B_minus_robot per arm (start_pose.json in a temporary TAGRIG_STATE_DIR)
  run_session (dry, no --go) on a stub rig (fake joints, black frames, fixed synthetic chunk, fake clock), 3 cycles, commit 8,
    with / without --height-calib -> every committed row's candidate z differs by dz_at(row xy) * min(1, j / 16), xy unchanged
Needs pinocchio / pink and the bimanual reBot URDF (egogarment.paths.HANDUMI_SW_ROOT); used by tests/test_ui_routes.py.
usage: python tests/ui/height_calib_probe.py <repo root> <out.json>"""
import json, os, pathlib, socket, sys, tempfile, time as _rt, types
repo, out = sys.argv[1], sys.argv[2]
STATE = tempfile.mkdtemp(); os.environ["TAGRIG_STATE_DIR"] = STATE
for k in ("EGO_TRAIN_ROOT", "EGO_DESKTOP"):
    os.environ[k] = os.path.join(STATE, k.lower())
def _no(self, *a, **k): raise OSError("socket blocked by height_calib_probe")
socket.socket.connect = _no; socket.socket.connect_ex = _no
sys.path.insert(0, os.path.join(repo, "src"))
import numpy as np
from garment_pipeline.eval_deploy import act_tagrig_core as core, act_tagrig_deploy as D, height_calib as HC
from garment_pipeline.eval_deploy.act_tagrig_ui import dep_parser
from visual_teleop.tools.robot_moves import READY_Q_DEG

CAL = {"created": "synthetic", "fit": {
    "left": {"n": 5, "dz_mean_mm": -3.0, "plane_dz": {"a": 0.08, "b": -0.03, "c": -0.004}, "xy_B_minus_robot": [0.006, -0.004]},
    "right": {"n": 5, "dz_mean_mm": 9.0, "plane_dz": {"a": -0.05, "b": 0.02, "c": 0.011}, "xy_B_minus_robot": [-0.003, 0.008]}}}
HC.OUT.write_text(json.dumps(CAL))
arms, ra = core.make_arms()
qr = {s: np.radians(READY_Q_DEG[s]) for s in core.SIDES}
R = {"state_files": [str(HC.OUT), str(D.START_FILE)]}

# ---- start_cmd: stub label start pose = READY TCP 8 cm lower
P0 = {s: core.fk(arms, s, qr[s]) for s in core.SIDES}
for s in core.SIDES: P0[s][2, 3] -= 0.08
real_LR = core.LabelRunner
core.LabelRunner = lambda path, *a: types.SimpleNamespace(T_at=lambda s, t: P0[s].copy())
try:
    for tag, extra in (("hc", ["--height-calib"]), ("nohc", [])):
        D.start_cmd(dep_parser().parse_args(["start", "--label", "stub.npz", "--start-lift", "0"] + extra))
        R[f"start_{tag}"] = json.loads(D.START_FILE.read_text())
finally:
    core.LabelRunner = real_LR
R["start_shift"] = {s: (np.array(R["start_hc"]["T"][s])[:2, 3] - np.array(R["start_nohc"]["T"][s])[:2, 3]).tolist() for s in core.SIDES}

# ---- run_session: stub rig, fixed chunk, fake clock
G = D.run_session.__globals__
class FakeTime:
    def __init__(self): self.t = 1000.0
    def monotonic(self): self.t += 0.001; return self.t
    def monotonic_ns(self): return int(self.monotonic() * 1e9)
    def sleep(self, s): self.t += max(0.0, float(s))
    def __getattr__(self, k): return getattr(_rt, k)
class FakePreview:
    def __init__(self, *a, **k): self.res = None; self.stop = False
    def start(self): pass
    def submit(self, *a, **k): pass
class FakeRunner:
    kind = "fake"
    def workspace(self, margin=0.05): return {s: (np.full(3, -5.0), np.full(3, 5.0)) for s in core.SIDES}
    def item(self, imgs, st, prev=None): return None
    def predict(self, item, seed=0, **k):
        ch = np.zeros((32, 20))
        for kk in range(32):
            for i, s in enumerate(core.SIDES):
                ch[kk, core.SL[s]] = [0.002 * (kk + 1), (0.001 if i else -0.001) * (kk + 1), -0.0015 * (kk + 1), 1, 0, 0, 0, 1, 0]
                ch[kk, core.GI[s]] = 0.6
        return None, ch
cap = {}
def fake_summarize(L, a, stop_reason, *rest): cap["L"] = dict(L); cap["stop"] = stop_reason
q0 = {s: np.asarray(R["start_nohc"]["q"][s], float) for s in core.SIDES}
from visual_teleop import config as vcfg
c = vcfg.load()
for tag, extra in (("hc", ["--height-calib"]), ("nohc", [])):
    FT = FakeTime()
    def ring(v): return types.SimpleNamespace(latest=lambda: (int(FT.t * 1e9), v), near=lambda t, tol: None)
    saved = {k: G[k] for k in ("OUT", "Preview", "time", "summarize")}
    G.update(OUT=pathlib.Path(STATE) / "runs", Preview=FakePreview, time=FT, summarize=fake_summarize)
    rob = types.SimpleNamespace(read=lambda: (q0, {s: 1.0 for s in core.SIDES}, False), ring=ring((q0, {s: 1.0 for s in core.SIDES}, False)), error=None)
    rig = types.SimpleNamespace(arms=arms, ra=ra, col=core.Collision(arms), rob=rob, gl=types.SimpleNamespace(ring=ring((np.zeros((480, 848, 3), np.uint8), b""))),
                                wr={s: types.SimpleNamespace(ring=ring(np.zeros((216, 384, 3), np.uint8))) for s in core.SIDES}, qr=q0, cfg={},
                                url=c.robot["base_url"], c=c, wave_ok=False, replay=False, fake_q=q0, runner=lambda *a_: FakeRunner())
    n = {"c": 0}
    def poll():
        n["c"] += 1; return "test end" if n["c"] > 3 else None
    ctl = types.SimpleNamespace(poll=poll, publish=lambda d: None, take_step=lambda: False)
    try: D.run_session(dep_parser().parse_args(["run", "--policy", "act", "--ckpt", "fake", "--commit", "8", "--secs", "0", "--assume-start"] + extra), rig, ctl)
    finally: G.update(saved)
    R[f"run_{tag}_stop"] = cap["stop"]; R[f"run_{tag}_T_cand"] = np.asarray(cap["L"]["T_cand"]).tolist(); R[f"run_{tag}_flags"] = cap["L"]["flags"]
h, n_ = np.array(R["run_hc_T_cand"][0]), np.array(R["run_nohc_T_cand"][0])        # first cycle: (commit rows, sides, 4, 4)
R["run_dz_err"] = max(abs((h[j, i, 2, 3] - n_[j, i, 2, 3]) - HC.dz_at(CAL, s, h[j, i, :2, 3]) * min(1.0, (j + 1) / 16.0))
                      for j in range(h.shape[0]) for i, s in enumerate(core.SIDES))
R["run_dz_mm_row8"] = {s: float((h[7, i, 2, 3] - n_[7, i, 2, 3]) * 1e3) for i, s in enumerate(core.SIDES)}
R["run_xy_equal"] = bool(np.allclose(h[..., :2, 3], n_[..., :2, 3]))
json.dump(R, open(out, "w"), default=str)
print("ok", R["start_shift"], R["run_dz_err"], R["run_dz_mm_row8"])
