"""[2026-10-23] Shared core of the tag-rig ACT real-robot deploy (deploy/act_tagrig_deploy.py) and its parity test
(deploy/act_tagrig_parity.py).  Everything that decides WHAT the policy sees and HOW its output becomes a robot target lives here,
so the parity test exercises the exact deploy code path.

Contract (datasets/tag_rig_to_lerobot.py, training/tagrig_common.py, checkpoint act_tagrig_v1_1m):
  images   observation.images.{global, left_wrist, right_wrist}, RGB float [0, 1] CHW, 384x216
           global = Orbbec colour 848x480 (OrbbecCapture, JPEG q95)  -> INTER_AREA 384x216 (slight 1.767 -> 1.778 stretch, as trained)
           wrist  = Arducam fisheye 1920x1080 -> INTER_AREA 1280x720 (JPEG q92, umi_jaw record) -> INTER_AREA 384x216
  state    30D = proprio_rows_lr: [inv(T_L0) T_L | inv(T_R0) T_R | gL | gR | inv(T_L) T_R | 1]; T_X0 = pose at the EPISODE start
           (deploy: the measured pose of the first cycle), g binary 1 = open (deploy: the last commanded binary jaw)
  action   (32, 20) rows k = 1..32: [inv(T(t)) T(t+k) L pose9 | R pose9 | gL | gR], rot6d = first two ROWS
  frame    T = FK(q) of <side>_tcp in the reBot base B (PinkIkSolver: URDF + tcp/joint/base overrides) = the robotize_ik label frame
           (R_B_E = R_B_tcp @ R_U_E).  Left arm = base +y.
Policy loading / pre / post-processing = evaluation/tagrig_offline_eval.load (the checkpoint's own processors)."""
from __future__ import annotations
import json, pathlib, sys
from types import SimpleNamespace

import cv2
import numpy as np

from egogarment import paths as egopaths  # noqa: E402
LDT = egopaths.TRAIN_ROOT
from contracts.action_contract import T_to_pose9, inv_T, make_T, pose9_to_T  # noqa: E402

CKPT = LDT / "checkpoints_pod/act_tagrig_v1_1m/checkpoints/0500000"
IMG_WH = (384, 216)
CAMS = ("global", "left_wrist", "right_wrist")
TASK = "laundry folding"
SIDES = ("left", "right")
SL = {"left": slice(0, 9), "right": slice(9, 18)}
GI = {"left": 18, "right": 19}
DT = 0.05
ROBOT_CAM = egopaths.DATA_CALIB / "robot_cam_v001.json"
# 10-29: the Orbbec was raised on 10-26 -> the overlay extrinsic is v005 (fixed tags + table plane, 10-26 12:08; FK TCP lands on the
# real jaw tips, checked on the live frame), intrinsics still from v001 (v002+ store no intrinsics).  Display only (UI + run video).
ROBOT_CAM_EXT = egopaths.DATA_CALIB / "robot_cam_v005.json"


# ---------------------------------------------------------------- images
def prep_global(bgr):
    """Orbbec colour BGR (848x480, already JPEG q95 like the take files) -> RGB uint8 384x216 (converter load_img)"""
    return cv2.cvtColor(cv2.resize(bgr, IMG_WH, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def prep_wrist(bgr, jpeg_q=92, align=None):
    """wrist BGR 1920x1080 -> 1280x720 INTER_AREA (umi_jaw record) -> JPEG q92 round trip (take files) -> converter load_img.
    align: optional deterministic framing transform (act_tagrig_domain_gap.py), applied on the 1280x720 frame; None = as trained"""
    x = bgr if bgr.shape[:2] == (720, 1280) else cv2.resize(bgr, (1280, 720), interpolation=cv2.INTER_AREA)
    if align is not None: x = apply_align(x, align)
    if jpeg_q: x = cv2.imdecode(cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_q)])[1], cv2.IMREAD_COLOR)
    return prep_global(x)


def apply_align(x, al):
    """similarity about the image centre: scale s, shift (dx, dy) px of the 1280x720 frame, rotation 0/180 deg, optional h-flip"""
    h, w = x.shape[:2]
    if al.get("rot180"): x = cv2.rotate(x, cv2.ROTATE_180)
    if al.get("hflip"): x = cv2.flip(x, 1)
    s, dx, dy = float(al.get("scale", 1.0)), float(al.get("dx", 0.0)), float(al.get("dy", 0.0))
    if s != 1.0 or dx or dy:
        M = np.float32([[s, 0, (1 - s) * w / 2 + dx], [0, s, (1 - s) * h / 2 + dy]])
        x = cv2.warpAffine(x, M, (w, h), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return x


def make_item(imgs_rgb: dict, state30):
    """uint8 HWC RGB per camera + state -> the dict LeRobotDataset.__getitem__ would give (float32 CHW in [0, 1])"""
    import torch
    im = lambda x: torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float() / 255.0
    it = {f"observation.images.{c}": im(imgs_rgb[c]) for c in CAMS}; it["observation.state"] = torch.from_numpy(np.asarray(state30, np.float32))
    it["task"] = TASK
    return it


def make_item2(imgs_rgb, state30, prev=None):
    """n_obs_steps = 2 item (DiT): (2, ...) stacks [t - 0.05 s, t]; prev = (imgs_rgb, state30) of t - 0.05 s, None -> the current
    frame repeated (= LeRobot's clamped padding at an episode start)"""
    import torch
    cur = make_item(imgs_rgb, state30); old = make_item(*prev) if prev is not None else cur
    out = {k: torch.stack([old[k], cur[k]]) for k in cur if k != "task"}; out["task"] = TASK; return out


# ---------------------------------------------------------------- state / action
def build_state(T_L, T_R, anchor_L, anchor_R, gL, gR):
    """contract proprio of the CURRENT frame with the episode anchor -- umi_to_lerobot.proprio_rows_lr on [anchor, current]"""
    from umi_to_lerobot import proprio_rows_lr
    TL = np.stack([anchor_L, T_L]); TR = np.stack([anchor_R, T_R])
    return proprio_rows_lr(TL, TR, np.array([gL, gL], float), np.array([gR, gR], float))[1]


def decode(chunk, T_L, T_R):
    """(32, 20) relative chunk + measured T(t) -> absolute targets {side: (32, 4, 4)} = T(t) @ dT_k (contracts.decode_action_chunk)"""
    from contracts.action_contract import decode_action_chunk
    d = decode_action_chunk(chunk, T_L, T_R); return {"left": d.T_L, "right": d.T_R}


def rtc_weights(H, d, n_prev):
    """[10-29] Real-Time Chunking (Black, Galliker, Levine 2025, eq. 5) soft mask over H rows of the NEW chunk: rows < d (executed while
    this chunk is being computed) = 1, rows d .. n_prev-1 (still covered by the previous plan) fade c (e^c - 1) / (e - 1),
    c = (n_prev - i) / (n_prev - d + 1), rows >= n_prev (beyond the previous plan) = 0."""
    W = np.zeros(H); n_prev = int(np.clip(n_prev, 0, H)); d = int(np.clip(d, 0, n_prev)); W[:d] = 1.0
    if n_prev > d:
        c = (n_prev - np.arange(d, n_prev)) / (n_prev - d + 1); W[d:n_prev] = c * np.expm1(c) / np.expm1(1.0)
    return W


class PolicyRunner:
    """tagrig_offline_eval.load + the predict() lines of tagrig_offline_eval, returning the normalised AND the denormalised chunk
    (32, 20) rows k = 1..32.  ACT: predict_action_chunk.  Multi-Task DiT (n_obs 2, horizon 33): DDIM 10 (offline-eval sampler),
    torch.manual_seed(seed) before sampling, rows 1..32 of the 33; n_samples > 1 = mean of K batched samples (dit_serving style)."""
    def __init__(self, ckpt=CKPT, device="mps", n_samples=1):
        import torch
        from garment_pipeline.eval_deploy.tagrig_offline_eval import load
        self.torch = torch; self.device = device; self.n_samples = int(n_samples)
        self.cfg, self.pol, self.pre, self.post = load(str(ckpt), device)
        self.kind = self.cfg.type; self.n_obs = int(self.cfg.n_obs_steps)
        assert list(self.cfg.image_features) == [f"observation.images.{c}" for c in CAMS], self.cfg.image_features
        assert (self.kind == "act" and self.cfg.chunk_size == 32 and self.n_obs == 1) or (self.kind == "multi_task_dit" and self.cfg.horizon == 33 and self.n_obs == 2), self.cfg
        pm = pathlib.Path(ckpt) / "pretrained_model"
        from safetensors.numpy import load_file
        self.stats = load_file(str(next(pm.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))))
        # 10-29: runs trained with GRIP_STATE_MASK=1 (training/tagrig_common.py) are named *-GSM (or carry a GRIP_STATE_MASK file) -> same input
        self.grip_mask = "-GSM" in str(ckpt) or any((p_ / "GRIP_STATE_MASK").exists() for p_ in (pathlib.Path(ckpt), pathlib.Path(ckpt).parent, pathlib.Path(ckpt).parent.parent))
        self.grip_fake_open = False      # 10-29 diagnostic (UI): feed the jaw state as OPEN to a copycat checkpoint

    def _jaw(self, state30):
        if not (self.grip_mask or self.grip_fake_open): return state30
        s = np.array(state30, np.float32, copy=True); s[18:20] = 0.0 if self.grip_mask else 1.0; return s

    def item(self, imgs_rgb, state30, prev=None):
        if prev is not None: prev = (prev[0], self._jaw(prev[1]))
        return make_item(imgs_rgb, self._jaw(state30)) if self.n_obs == 1 else make_item2(imgs_rgb, self._jaw(state30), prev)

    def predict(self, item, seed=0, rtc=None):
        """rtc (DiT only): dict(Y=(32, 20) denormalised contract rows k = 1..32 of the previous plan re-anchored at this observation,
        W=(32,) rtc_weights, g0=(2,) jaw at row 0, mode "guided" | "blend", beta) -> Real-Time Chunking sampler (_sample_rtc)"""
        torch = self.torch
        with torch.no_grad():
            batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for k, v in item.items() if k.startswith("observation") or k == "task"}
            b = self.pre(batch); torch.manual_seed(seed)
            if self.kind == "act":
                n = self.pol.predict_action_chunk(b); d = self.post(n.clone())
                return n[0].float().cpu().numpy(), d[0].float().cpu().numpy()
            bb = self.pol._prepare_batch(dict(b)); cond = self.pol.observation_encoder.encode(bb); K = self.n_samples
            if rtc is not None and K == 1: n = self._sample_rtc(cond, rtc)
            else: n = self.pol.objective.conditional_sample(self.pol.noise_predictor, K, cond.expand(K, -1) if K > 1 else cond)
            if hasattr(self.pol, "grip_head"):          # 10-29 GRIP_BCE checkpoints: jaw rows = the BCE head's P(open), in the [-1, 1] space
                lg = self.pol.grip_logits(cond.expand(n.shape[0], -1) if n.shape[0] > 1 else cond); n[..., 18:20] = 2.0 * torch.sigmoid(lg).to(n.dtype) - 1.0
            d = self.post(n.clone()).float().cpu().numpy(); n = n.float().cpu().numpy()
            return n.mean(0)[1:], d.mean(0)[1:]

    def normalize(self, rows33):
        """denormalised contract rows (33, 20) -> the DiT's MIN_MAX action space (inverse of self.post)"""
        lo, hi = self.stats["action.min"].astype(np.float64), self.stats["action.max"].astype(np.float64)
        return 2.0 * (np.asarray(rows33, np.float64) - lo) / np.maximum(hi - lo, 1e-8) - 1.0

    def _sample_rtc(self, cond, rtc):
        """[10-29] Real-Time Chunking on the DDIM sampler.  Y = row 0 (identity + jaw) + the previous plan's rows, W = 1 on row 0 and
        rtc["W"] on rows 1..32.  "guided" = RTC's pseudo-inverse guidance (PiGDM) in epsilon form: x0_hat = (x - sqrt(1-a) eps) / sqrt(a),
        g = (dx0_hat/dx)^T W (Y - x0_hat) (one vector-Jacobian product per step), eps <- eps - sqrt(1-a) min(beta, 1 / r2) g, r2 = 1 - a
        (variance of x0 given x_t for unit-variance data).  "blend" = gradient-free: x0_hat <- x0_hat + W (Y - x0_hat), eps recomputed."""
        torch = self.torch; obj = self.pol.objective; net = self.pol.noise_predictor; sch = obj.noise_scheduler
        dev = cond.device; dt_ = next(net.parameters()).dtype
        Y33 = np.vstack([np.r_[T_to_pose9(np.eye(4)), T_to_pose9(np.eye(4)), rtc["g0"]][None], rtc["Y"]])
        Yn = torch.from_numpy(self.normalize(Y33)).to(dev, dt_)[None]; W = torch.from_numpy(np.r_[1.0, rtc["W"]]).to(dev, dt_).view(1, -1, 1)
        x = torch.randn((1, obj.horizon, obj.action_dim), dtype=dt_, device=dev); sch.set_timesteps(obj.num_inference_steps)
        ac = sch.alphas_cumprod.to(dev, dt_); beta = float(rtc.get("beta", 5.0)); guided = rtc.get("mode", "guided") == "guided"
        for t in sch.timesteps:
            a_ = ac[int(t)]; tt = torch.full((1,), int(t), dtype=torch.long, device=dev)
            if guided:
                with torch.enable_grad():
                    xg = x.detach().requires_grad_(True); eps = net(xg, tt, conditioning_vec=cond)
                    x0 = (xg - (1 - a_).sqrt() * eps) / a_.sqrt()
                    g, = torch.autograd.grad(x0, xg, grad_outputs=(W * (Yn - x0)).detach())
                eps = eps.detach() - (1 - a_).sqrt() * min(beta, 1.0 / max(float(1 - a_), 1e-6)) * g
            else:
                eps = net(x, tt, conditioning_vec=cond); x0 = (x - (1 - a_).sqrt() * eps) / a_.sqrt(); x0 = x0 + W * (Yn - x0)
                eps = (x - a_.sqrt() * x0) / (1 - a_).sqrt().clamp_min(1e-6)
            x = sch.step(eps, t, x).prev_sample
        return x

    def workspace(self, margin=0.05):
        """training TCP range (pose_abs min/max of the TRAIN root, from the checkpoint's normalizer) +- margin, per arm xyz"""
        lo, hi = self.stats["pose_abs.min"], self.stats["pose_abs.max"]
        return {s: (lo[SL[s]][:3] - margin, hi[SL[s]][:3] + margin) for s in SIDES}


class XvlaRunner:
    """[10-28] Exp 3 X-VLA (REL32-bin, GOLD-281 domain 21) through deploy/xvla_server.py (separate process, :8775), same interface
    as PolicyRunner.  Frames: the deploy (= label A) uses eef_kin axes; the X-VLA set uses dataset-TCP axes T_x = T_A @ Ry(-90)
    (scratchpad exp3/gold_export.py), so state rel_x = Ry^T rel_A Ry and chunk rel_A = Ry rel_x Ry^T (Ry = Ry(-90)).
    state20 = RELCART20 task anchor = proprio[0:20] (episode-anchor poses + grippers) in those axes; chunk out (32, 20) contract rows
    with grippers = P(open).  Workspace = the TCP range of the GOLD train root (Exp 1 DiT checkpoint stats, same data)."""
    RY = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
    WS_STATS = (egopaths.CKPT_ROOT / "ckpts/runpod/pod1/DITFT-SFPRE300K-FLAT300-A-GOLD281-BS16-300K/checkpoints/300000/pretrained_model")

    def __init__(self, ckpt, url="http://127.0.0.1:8775"):
        import requests
        self.s = requests.Session(); self.url = url; self.kind = "xvla"; self.n_obs = 1
        self.ckpt = str(ckpt)
        r = self.s.post(url + "/load", json={"ckpt": str(ckpt)}, timeout=300).json()
        if not r.get("ok"): raise SystemExit(f"X-VLA 서버 체크포인트 로드 실패: {r}")
        self.info = r["info"]
        from safetensors.numpy import load_file
        self.stats = load_file(str(next(self.WS_STATS.glob("policy_preprocessor_step_*_normalizer_processor.safetensors"))))
        self.Rh = make_T(self.RY, np.zeros(3))

    def _to_x(self, p9): return T_to_pose9(inv_T(self.Rh) @ pose9_to_T(np.asarray(p9, np.float64)) @ self.Rh)

    def _from_x(self, p9): return T_to_pose9(self.Rh @ pose9_to_T(np.asarray(p9, np.float64)) @ inv_T(self.Rh))

    grip_fake_open = False              # 10-29 diagnostic (UI): feed the jaw state as OPEN (the k1 jaw copies the input state)

    def item(self, imgs_rgb, state30, prev=None):
        s = np.asarray(state30, np.float64)
        gl_, gr_ = (0.0, 0.0) if "-GSM" in str(getattr(self, "ckpt", "")) else (1.0, 1.0) if self.grip_fake_open else (s[18], s[19])   # 10-29 *-GSM: trained with the jaw state = 0
        st20 = np.r_[self._to_x(s[SL["left"]]), self._to_x(s[SL["right"]]), gl_, gr_].astype(np.float32)
        return dict(imgs={c: np.ascontiguousarray(imgs_rgb[c]) for c in CAMS}, state20=st20)

    def predict(self, item, seed=0, rtc=None):
        """rtc: dict(Y=(32, 20) previous plan rows in the deploy's relative contract (eef_kin axes, jaws = P(open)), W=(32,)) ->
        sent in the X-VLA dataset-TCP axes; the server blends the flow's clean-chunk prediction (deploy/xvla_server.py install_rtc)"""
        import io
        ex = {}
        if rtc is not None:
            Y = np.zeros((32, 32)); Yd = np.asarray(rtc["Y"], np.float64)
            Y[:, 0:9] = np.stack([self._to_x(r) for r in Yd[:, SL["left"]]]); Y[:, 10:19] = np.stack([self._to_x(r) for r in Yd[:, SL["right"]]])
            Y[:, 9], Y[:, 19] = Yd[:, GI["left"]], Yd[:, GI["right"]]; ex = dict(rtc_Y=Y.astype(np.float32), rtc_W=np.asarray(rtc["W"], np.float32))
        buf = io.BytesIO(); np.savez(buf, state20=item["state20"], seed=np.int64(seed), **item["imgs"], **ex)
        r = self.s.post(self.url + "/predict", data=buf.getvalue(), timeout=30); r.raise_for_status()
        a = np.load(io.BytesIO(r.content))["chunk"]                                  # (32, 32) [L xyz rot6d g | R ... | 0]
        ch = np.zeros((32, 20)); ch[:, SL["left"]] = self._from_x(a[:, 0:9]); ch[:, SL["right"]] = self._from_x(a[:, 10:19])
        ch[:, GI["left"]], ch[:, GI["right"]] = a[:, 9], a[:, 19]
        if self.time_scale is not None: ch = self._retime(ch)       # None = raw 15 Hz rows (deploy look-ahead runs them at their own tick)
        return ch.copy(), ch

    ROW_DT = 1.0 / 15.0                 # X-VLA REL32 rows are 15 Hz; the deploy loop executes rows every DT = 0.05 s (20 Hz)
    time_scale = 1.0                    # 10-29: set from the UI speed (run_session) -- slow motion ALONG the predicted path

    def _retime(self, ch):
        """[10-29] 15 Hz rows k/15 -> deploy rows at k * DT * time_scale.  Before this the 15 Hz rows were played at 20 Hz (1.33x too fast)
        and the UI speed only shrank each target toward the current pose (jittery creep)."""
        return retime(ch, self.ROW_DT, self.time_scale)

    def workspace(self, margin=0.05):
        lo, hi = self.stats["pose_abs.min"], self.stats["pose_abs.max"]
        return {s: (lo[SL[s]][:3] - margin, hi[SL[s]][:3] + margin) for s in SIDES}


def retime(ch, row_dt, time_scale):
    """[10-29] relative chunk (32, 20), row j = pose at (j + 1) * row_dt (row 0 = identity at t = 0) -> rows at k * DT * time_scale,
    k = 1..32 (position lerp, rotation slerp, jaw lerp): slow motion ALONG the predicted path (X-VLA 15 Hz rows, DiT 20 Hz rows)."""
    from scipy.spatial.transform import Rotation as Rot, Slerp
    ts = np.r_[0.0, np.arange(1, 33) * row_dt]; tq = np.clip(np.arange(1, 33) * DT * time_scale, 0.0, ts[-1]); out = np.array(ch, np.float64)
    for s in SIDES:
        T = np.stack([np.eye(4)] + [pose9_to_T(p) for p in out[:, SL[s]]])
        P = np.stack([np.interp(tq, ts, T[:, 0, 3]), np.interp(tq, ts, T[:, 1, 3]), np.interp(tq, ts, T[:, 2, 3])], 1)
        R = Slerp(ts, Rot.from_matrix(T[:, :3, :3]))(tq).as_matrix()
        out[:, SL[s]] = np.stack([T_to_pose9(make_T(R[k], P[k])) for k in range(32)])
        g = np.r_[ch[0, GI[s]], np.asarray(ch)[:, GI[s]]]; out[:, GI[s]] = np.interp(tq, ts, g)
    return out


class LabelRunner:
    """[10-27] oracle 'policy' for label replay (A' vs Cg test): the training label of one demo (robotize_ik ik_v1/demoNN.npz,
    <side>_p_rob / <side>_R_rob = the action frame, <side>_g = umi_jaw state 1 = open) played through the unchanged deploy loop.
    predict() returns rows k = 1..32 of inv(T(tau)) T(tau + k DT rate) at label time tau = (now - t0) * rate (rate < 1 = slow motion),
    i.e. the same relative chunk a perfect policy would output; the deploy loop composes it onto the measured FK pose as for a policy."""
    kind = "label"; n_obs = 1

    def __init__(self, path, rate=0.5):
        from scipy.spatial.transform import Rotation, Slerp
        z = np.load(path, allow_pickle=True); self.path = str(path); self.rate = float(rate); self.t0 = None; self.cursor = None     # cursor (label s): set by deploy --go/dry chaining
        t = (z["t_ns"].astype(np.int64) - int(z["t_ns"][0])) / 1e9; self.t = t; self.dur = float(t[-1])
        self.P = {s: z[f"{s}_p_rob"].astype(float) for s in SIDES}; self.G = {s: z[f"{s}_g"].astype(float) for s in SIDES}
        self.S = {s: Slerp(t, Rotation.from_matrix(z[f"{s}_R_rob"])) for s in SIDES}

    def T_at(self, s, tau):
        tau = float(np.clip(tau, 0.0, self.dur)); T = np.eye(4)
        T[:3, 3] = [np.interp(tau, self.t, self.P[s][:, j]) for j in range(3)]; T[:3, :3] = self.S[s](tau).as_matrix(); return T

    def g_at(self, s, tau): return float(self.G[s][min(np.searchsorted(self.t, np.clip(tau, 0, self.dur)), len(self.t) - 1)])

    def tau(self):
        import time
        return 0.0 if self.t0 is None else (time.monotonic() - self.t0) * self.rate

    def item(self, imgs_rgb, state30, prev=None): return None

    def predict(self, item, seed=0):
        tau = self.tau() if self.cursor is None else self.cursor; ch = np.zeros((32, 20))
        ts = np.clip(tau + np.arange(33) * DT * self.rate, 0.0, self.dur)          # one vectorised slerp / interp per arm (216 -> ~2 ms)
        for s in SIDES:
            R = self.S[s](ts).as_matrix(); P = np.stack([np.interp(ts, self.t, self.P[s][:, j]) for j in range(3)], 1)
            T = np.tile(np.eye(4), (33, 1, 1)); T[:, :3, :3] = R; T[:, :3, 3] = P; T0i = inv_T(T[0])
            for k in range(1, 33): ch[k - 1, SL[s]] = T_to_pose9(T0i @ T[k])
            ch[:, GI[s]] = self.G[s][np.minimum(np.searchsorted(self.t, ts[1:]), len(self.t) - 1)]
        return ch, ch

    def done(self): return self.cursor is not None and self.cursor >= self.dur - 1e-6

    def workspace(self, margin=0.05):
        return {s: (self.P[s].min(0) - margin, self.P[s].max(0) + margin) for s in SIDES}


ActRunner = PolicyRunner
DIT_CKPT = LDT / "checkpoints_pod/dit_tagrig_v1/checkpoints/last"


# ---------------------------------------------------------------- robot kinematics (reuse robotize_ik.Arm = umi_replay_bi settings)
ROBOTIZE_ARGS = dict(ik_step=7.5, ori_cost=0.1, table_z=0.003, tip_clear=0.005, tip_clear_left_extra=0.005, jaw_half=0.020, level_max=0.0,
                     roll_thr=115.0, roll_relax=1.5)


def make_arms(**over):
    """robotize_ik.Arm per side: Pink solver with the bridge joint limits (0.5 deg inside), joint/base/TCP overrides, orientation
    cost 0.1, 7.5 deg/step bound; corner_lift (finger-corner table floor); link-origin set for the 120 mm gap"""
    from garment_pipeline.labels_dataset.robotize_ik import Arm
    a = SimpleNamespace(**{**ROBOTIZE_ARGS, **over})
    return {s: Arm(s, a) for s in SIDES}, a


def fk(arms, side, q):
    return arms[side].pk.fk(np.asarray(q, np.float64))


# ---------------------------------------------------------------- capsule collision model (no link meshes in the repo)
# 10-23: the URDF references meshes_b601_gripper/link1-6.STL + gripper STLs, but only the three gripper STLs exist on this Mac
# (bh-egocentric/hra/hra_a100/urdf_gripper) and no code path loads them -> link-link collision uses CAPSULES between consecutive
# frame origins of the corrected pinocchio model (PinkIkSolver.full: base + joint offsets).  Radii are conservative half-widths
# of the B601 links (tape / photos, 10-23), fingers = segment finger-frame origin -> finger tip at the TCP plane.
CAPSULES = [("link1", "link2", 0.045), ("link2", "link3", 0.040), ("link3", "link4", 0.035), ("link4", "link5", 0.032),
            ("link5", "link6", 0.032), ("link6", "gbody", 0.032)]
# 10-23: "gbody" = midpoint of the two finger-joint frames (jaw base).  NOT gripper_link: with the tcp override (-0.019 m) the URDF
# gripper_link origin sits 19 mm BEYOND the real jaw tips, so a link6 -> gripper_link capsule dipped into the table on 48 % of the
# training grasp frames.  Table check = link capsules only; the fingers are covered by robotize_ik corner_lift (finger-corner floor).
FINGER_R = 0.012
TABLE_R_WRIST = 0.012


class Collision:
    def __init__(self, arms):
        import pinocchio as pin
        pk = arms["right"].pk; self.pin = pin; self.m = pk.full; self.d = self.m.createData(); self.q0 = pk.q_full_pin.copy()
        self.qidx = pk.qidx; self.joff = pk.joff
        self.fid = {f"{s}_{n}": self.m.getFrameId(f"{s}_{n}") for s in SIDES for n in
                    ("link1", "link2", "link3", "link4", "link5", "link6", "gripper_link", "gripper_left", "gripper_right", "tcp")}

    def capsules(self, q):
        """q {side: (6,) bridge rad} -> {side: [(a, b, r, name)]}"""
        pin = self.pin; qq = self.q0.copy()
        for s in SIDES: qq[self.qidx[s]] = np.asarray(q[s], np.float64) + self.joff[s]
        pin.framesForwardKinematics(self.m, self.d, qq); P = {k: self.d.oMf[i].translation.copy() for k, i in self.fid.items()}
        for s in SIDES: P[f"{s}_gbody"] = (P[f"{s}_gripper_left"] + P[f"{s}_gripper_right"]) / 2
        out = {}
        for s in SIDES:
            C = [(P[f"{s}_{a}"], P[f"{s}_{b}"], r, f"{a}-{b}") for a, b, r in CAPSULES]
            T = self.d.oMf[self.fid[f"{s}_tcp"]]; ax = T.rotation[:, 2]; tcp = T.translation
            for fn in ("gripper_left", "gripper_right"):
                pf = P[f"{s}_{fn}"]; lat = (pf - tcp) - ax * float(ax @ (pf - tcp))       # finger offset across the tool axis
                C.append((pf - 0.02 * ax, tcp + lat, FINGER_R, fn))
            out[s] = C
        return out

    def check(self, q, table_z=0.003, arm_clear=0.010, table_clear=0.0):
        """-> dict(min_arm_clear_m, pair, min_table_clear_m (non-finger capsules), finger_low_m)"""
        C = self.capsules(q); best = (np.inf, None)
        for a0, b0, r0, n0 in C["left"]:
            for a1, b1, r1, n1 in C["right"]:
                dd = seg_dist(a0, b0, a1, b1) - r0 - r1
                if dd < best[0]: best = (dd, f"L.{n0} / R.{n1}")
        # table: link capsules; the wrist capsule (link6-gbody) with TABLE_R_WRIST (its 32 mm arm-arm radius put 5.4 % of the training
        # grasp frames "under" the table -- the real jaw housing is narrower toward the fingers); fingers -> corner_lift limiter
        tab = min(min(a[2], b[2]) - (TABLE_R_WRIST if n == "link6-gbody" else r) - table_z for s in SIDES for a, b, r, n in C[s] if not n.startswith("gripper_"))
        fl = min(min(a[2], b[2]) - r - table_z for s in SIDES for a, b, r, n in C[s] if n.startswith("gripper_"))
        return dict(arm_clear_m=float(best[0]), pair=best[1], table_clear_m=float(tab), finger_low_m=float(fl),
                    ok_arm=bool(best[0] >= arm_clear), ok_table=bool(tab >= table_clear))


def seg_dist(p1, q1, p2, q2):
    """minimum distance between segments p1q1 and p2q2 (Ericson, Real-Time Collision Detection 5.1.9)"""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2; a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a <= 1e-12 and e <= 1e-12: return float(np.linalg.norm(r))
    if a <= 1e-12: s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e <= 1e-12: t, s = 0.0, np.clip(-c / a, 0, 1)
        else:
            b = d1 @ d2; den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0: t, s = 0.0, np.clip(-c / a, 0, 1)
            elif t > 1: t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def link_gap(arms, q):
    """umi_replay_bi / robotize_ik link-origin distance (m): min over link3-6, gripper_link, fingers, tcp of both arms"""
    return float(np.linalg.norm(arms["left"].links(q["left"])[:, None] - arms["right"].links(q["right"])[None], axis=2).min())


# ---------------------------------------------------------------- geometry helpers
def rot_interp(Ra, Rb, s):
    from scipy.spatial.transform import Rotation as Rot
    return (Rot.from_matrix(Ra) * Rot.from_rotvec(s * Rot.from_matrix(Ra.T @ Rb).as_rotvec())).as_matrix()


def rot_angle_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def project(T_C_B, K, P_B):
    """base-frame points (N, 3) -> Orbbec pixels (N, 2) (848x480 intrinsics), points behind the camera -> nan"""
    P = (T_C_B[:3, :3] @ np.asarray(P_B).T).T + T_C_B[:3, 3]; z = P[:, 2:3]
    uv = np.c_[K["fx"] * P[:, 0] / P[:, 2] + K["cx"], K["fy"] * P[:, 1] / P[:, 2] + K["cy"]]
    uv[(z[:, 0] <= 0.05)] = np.nan; return uv


def robot_cam():
    j = json.loads(ROBOT_CAM.read_text()); return np.asarray(json.loads(ROBOT_CAM_EXT.read_text())["T_C_B"]), j["intrinsics"]


def cv_plot(series, ylim, title="", size=(900, 260), hline=None, xlabel=""):
    """tiny cv2 line plot (no matplotlib in .venv-mac): series = [(x, y, BGR colour, label)] -> BGR image"""
    W, H = size; img = np.full((H, W, 3), 255, np.uint8); l, r, t, b = 50, 10, 24, 22
    xs = np.concatenate([np.asarray(x, float) for x, *_ in series]); x0, x1 = float(xs.min()), float(max(xs.max(), xs.min() + 1e-9)); y0, y1 = ylim
    P = lambda x, y: (int(l + (x - x0) / (x1 - x0) * (W - l - r)), int(t + (y1 - np.clip(y, y0, y1)) / (y1 - y0) * (H - t - b)))
    cv2.rectangle(img, P(x0, y1), P(x1, y0), (0, 0, 0), 1)
    for v in (y0, (y0 + y1) / 2, y1): cv2.putText(img, f"{v:g}", (2, P(x0, v)[1] + 4), 0, 0.4, (0, 0, 0), 1)
    if hline is not None: cv2.line(img, P(x0, hline), P(x1, hline), (150, 150, 150), 1)
    for i, (x, y, c, lab) in enumerate(series):
        pts = np.array([P(a, b_) for a, b_ in zip(x, y) if np.isfinite(b_)], np.int32)
        if len(pts) > 1: cv2.polylines(img, [pts], False, c, 2)
        cv2.putText(img, lab, (W - 190, t + 16 + 16 * i), 0, 0.45, c, 1)
    cv2.putText(img, title, (l, 16), 0, 0.5, (0, 0, 0), 1); cv2.putText(img, f"{x0:.3g} .. {x1:.3g} {xlabel}", (l, H - 6), 0, 0.4, (0, 0, 0), 1)
    return img
