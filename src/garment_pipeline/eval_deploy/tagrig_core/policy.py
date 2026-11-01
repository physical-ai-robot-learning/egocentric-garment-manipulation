"""What the policy sees and how its output becomes a robot target: image and state preparation, chunk decode, RTC weights,
and the policy runners (ACT / Multi-Task DiT in process, X-VLA through xvla_server.py, a recorded label as an oracle).
The input / output contract is described in act_tagrig_core.py, the module the deploy code imports as `core`.
"""
from __future__ import annotations
import pathlib

import cv2
import numpy as np

from egogarment import paths as egopaths
from contracts.action_contract import T_to_pose9, inv_T, make_T, pose9_to_T


LDT = egopaths.TRAIN_ROOT
CKPT = LDT / "checkpoints_pod/act_tagrig_v1_1m/checkpoints/0500000"
IMG_WH = (384, 216)
CAMS = ("global", "left_wrist", "right_wrist")
TASK = "laundry folding"
SIDES = ("left", "right")
SL = {"left": slice(0, 9), "right": slice(9, 18)}
GI = {"left": 18, "right": 19}
DT = 0.05


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
