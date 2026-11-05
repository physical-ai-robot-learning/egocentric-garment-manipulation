"""Probe of the 11-01 height-calibration path of the tag-rig deploy (--height-calib), robot-free and deterministic:
  startCmd (dry, no --go) from a stub label start pose, with / without --height-calib -> the start TCP target shifts by
    -xy_B_minus_robot per arm (start_pose.json in a temporary TAGRIG_STATE_DIR)
  runSession (dry, no --go) on a stub rig (fake joints, black frames, fixed synthetic chunk, fake clock), 3 cycles, commit 8,
    with / without --height-calib -> every committed row's candidate z differs by dzAt(row xy) * min(1, j / 16), xy unchanged
Needs pinocchio / pink and the bimanual reBot URDF (laundry_common.paths.HANDUMI_SW_ROOT); used by tests/laundry/test_ui_routes.py.
usage: python tests/laundry/ui/height_calib_probe.py <repo root> <out.json>"""
import json, os, pathlib, socket, sys, tempfile, time as _rt, types
repo, out = sys.argv[1], sys.argv[2]
STATE = tempfile.mkdtemp(); os.environ["TAGRIG_STATE_DIR"] = STATE
for k in ("EGO_TRAIN_ROOT", "EGO_DESKTOP"):
    os.environ[k] = os.path.join(STATE, k.lower())
def _no(self, *a, **k): raise OSError("socket blocked by height_calib_probe")
socket.socket.connect = _no; socket.socket.connect_ex = _no
sys.path.insert(0, os.path.join(repo, "src"))
import numpy as np
from laundry_deploy.core import height_calib as HC
from laundry_deploy.core.collision import Collision
from laundry_deploy.core.contract import GRIP_INDEX, POSE_SLICE, SIDES
from laundry_deploy.core.kinematics import fk, makeArms
from laundry_deploy.session import run_loop, start_pose
from laundry_deploy.session.cli import buildParser as dep_parser
from laundry_shared.visual_teleop.tools.robot_moves import READY_Q_DEG

CAL = {"created": "synthetic", "fit": {
    "left": {"n": 5, "dz_mean_mm": -3.0, "plane_dz": {"a": 0.08, "b": -0.03, "c": -0.004}, "xy_B_minus_robot": [0.006, -0.004]},
    "right": {"n": 5, "dz_mean_mm": 9.0, "plane_dz": {"a": -0.05, "b": 0.02, "c": 0.011}, "xy_B_minus_robot": [-0.003, 0.008]}}}
HC.CALIB_FILE.write_text(json.dumps(CAL))
arms, ra = makeArms()
qr = {s: np.radians(READY_Q_DEG[s]) for s in SIDES}
R = {"state_files": [str(HC.CALIB_FILE), str(start_pose.START_FILE)]}

# ---- start_cmd: stub label start pose = READY TCP 8 cm lower
P0 = {s: fk(arms, s, qr[s]) for s in SIDES}
for s in SIDES: P0[s][2, 3] -= 0.08
real_LR = start_pose.LabelRunner
start_pose.LabelRunner = lambda path, *a: types.SimpleNamespace(poseAt=lambda s, t: P0[s].copy())
try:
    for tag, extra in (("hc", ["--height-calib"]), ("nohc", [])):
        start_pose.startCmd(dep_parser().parse_args(["start", "--label", "stub.npz", "--start-lift", "0"] + extra))
        R[f"start_{tag}"] = json.loads(start_pose.START_FILE.read_text())
finally:
    start_pose.LabelRunner = real_LR
R["start_shift"] = {s: (np.array(R["start_hc"]["T"][s])[:2, 3] - np.array(R["start_nohc"]["T"][s])[:2, 3]).tolist() for s in SIDES}

# ---- run_session: stub rig, fixed chunk, fake clock
G = run_loop.runSession.__globals__
class FakeTime:
    def __init__(self): self.t = 1000.0
    def monotonic(self): self.t += 0.001; return self.t
    def monotonic_ns(self): return int(self.monotonic() * 1e9)
    def sleep(self, s): self.t += max(0.0, float(s))
    def __getattr__(self, k): return getattr(_rt, k)
class FakePreview:
    def __init__(self, *a, **k): self.mResult = None; self.mStop = False
    def start(self): pass
    def submit(self, *a, **k): pass
class FakeRunner:
    mKind = "fake"
    def workspace(self, margin=0.05): return {s: (np.full(3, -5.0), np.full(3, 5.0)) for s in SIDES}
    def item(self, imgs, st, prev=None): return None
    def predict(self, item, seed=0, **k):
        ch = np.zeros((32, 20))
        for kk in range(32):
            for i, s in enumerate(SIDES):
                ch[kk, POSE_SLICE[s]] = [0.002 * (kk + 1), (0.001 if i else -0.001) * (kk + 1), -0.0015 * (kk + 1), 1, 0, 0, 0, 1, 0]
                ch[kk, GRIP_INDEX[s]] = 0.6
        return None, ch
cap = {}
def fake_summarize(L, a, stop_reason, *rest): cap["L"] = dict(L); cap["stop"] = stop_reason
q0 = {s: np.asarray(R["start_nohc"]["q"][s], float) for s in SIDES}
from laundry_shared.visual_teleop import config as vcfg
c = vcfg.load()
for tag, extra in (("hc", ["--height-calib"]), ("nohc", [])):
    FT = FakeTime()
    def ring(v): return types.SimpleNamespace(latest=lambda: (int(FT.t * 1e9), v), near=lambda t, tol: None)
    saved = {k: G[k] for k in ("RUN_OUT_DIR", "Preview", "time", "summarize")}
    G.update(RUN_OUT_DIR=pathlib.Path(STATE) / "runs", Preview=FakePreview, time=FT, summarize=fake_summarize)
    rob = types.SimpleNamespace(read=lambda: (q0, {s: 1.0 for s in SIDES}, False), mRing=ring((q0, {s: 1.0 for s in SIDES}, False)), mError=None)
    rig = types.SimpleNamespace(mArms=arms, mRobotArgs=ra, mCollision=Collision(arms), mPoller=rob,
                                mGlobalCam=types.SimpleNamespace(mRing=ring((np.zeros((480, 848, 3), np.uint8), b""))),
                                mWristCams={s: types.SimpleNamespace(mRing=ring(np.zeros((216, 384, 3), np.uint8))) for s in SIDES}, mQReady=q0, mCamsCfg={},
                                mUrl=c.robot["base_url"], mTeleopCfg=c, mWaveOk=False, mReplay=False, mFakeQ=q0, runner=lambda *a_: FakeRunner())
    n = {"c": 0}
    def poll():
        n["c"] += 1; return "test end" if n["c"] > 3 else None
    ctl = types.SimpleNamespace(poll=poll, publish=lambda d: None, takeStep=lambda: False)
    try: run_loop.runSession(dep_parser().parse_args(["run", "--policy", "act", "--ckpt", "fake", "--commit", "8", "--secs", "0", "--assume-start"] + extra), rig, ctl)
    finally: G.update(saved)
    R[f"run_{tag}_stop"] = cap["stop"]; R[f"run_{tag}_T_cand"] = np.asarray(cap["L"]["T_cand"]).tolist(); R[f"run_{tag}_flags"] = cap["L"]["flags"]
h, n_ = np.array(R["run_hc_T_cand"][0]), np.array(R["run_nohc_T_cand"][0])        # first cycle: (commit rows, sides, 4, 4)
R["run_dz_err"] = max(abs((h[j, i, 2, 3] - n_[j, i, 2, 3]) - HC.dzAt(CAL, s, h[j, i, :2, 3]) * min(1.0, (j + 1) / 16.0))
                      for j in range(h.shape[0]) for i, s in enumerate(SIDES))
R["run_dz_mm_row8"] = {s: float((h[7, i, 2, 3] - n_[7, i, 2, 3]) * 1e3) for i, s in enumerate(SIDES)}
R["run_xy_equal"] = bool(np.allclose(h[..., :2, 3], n_[..., :2, 3]))
json.dump(R, open(out, "w"), default=str)
print("ok", R["start_shift"], R["run_dz_err"], R["run_dz_mm_row8"])
