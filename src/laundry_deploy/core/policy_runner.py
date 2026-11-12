"""ACT / Multi-Task DiT policies in this process (deploy stage): core.checkpoint.loadPolicy + the predict() of the offline
evaluator, returning the normalised AND the denormalised chunk, plus the [10-29] Real-Time Chunking sampler for the DiT.

Input: a LeRobot-style item (core.images). Output: (32, 20) chunks, rows k = 1..32 (core.contract).
"""

from __future__ import annotations
import pathlib

import numpy as np

from laundry_shared.contracts.action_contract import T_to_pose9

from .checkpoint import loadPolicy
from .contract import ACT_CKPT, CAMS, CHUNK_ROWS, GRIP_COLUMNS, POSE_SLICE, SIDES
from .errors import CheckpointError
from .images import makeItem, makeItemPair
from .pgrasp import checkpointVersion

ACT_KIND = "act"
DIT_KIND = "multi_task_dit"
ACT_CHUNK = CHUNK_ROWS  # ACT checkpoints predict the 32 contract rows
DIT_HORIZON = CHUNK_ROWS + 1  # DiT: row 0 (identity) + the 32 contract rows
NORMALIZER_GLOB = "policy_preprocessor_step_*_normalizer_processor.safetensors"
GSM_TAG = "-GSM"  # 10-29: runs trained with GRIP_STATE_MASK=1 are named *-GSM ...
GSM_FILE = "GRIP_STATE_MASK"  # ... or carry this file next to / above the step folder
WORKSPACE_MARGIN_M = 0.05  # default margin around the training TCP range
RTC_BETA_DEFAULT = 5.0  # RTC guidance weight clip (paper: 5)
RTC_GUIDED, RTC_BLEND = "guided", "blend"
MIN_NOISE_VAR = 1e-6  # floor of 1 - alpha_cumprod in the RTC guidance / blend
NORMALIZE_EPS = 1e-8  # MIN_MAX range floor


class PolicyRunner:
    """ACT: predict_action_chunk. Multi-Task DiT (n_obs 2, horizon 33): DDIM 10 (offline-eval sampler), torch.manual_seed(seed)
    before sampling, rows 1..32 of the 33; nSamples > 1 = mean of K batched samples (dit_serving style)."""

    mPGrasp = None  # 11-06: PGRASP version of the checkpoint (set in __init__)

    def __init__(self, ckpt=ACT_CKPT, device: str = "mps", nSamples: int = 1) -> None:
        self.mDevice = device
        self.mNSamples = int(nSamples)
        self.mCfg, self.mPolicy, self.mPre, self.mPost = loadPolicy(str(ckpt), device)
        self.mKind = self.mCfg.type
        self.mNObs = int(self.mCfg.n_obs_steps)
        if list(self.mCfg.image_features) != [f"observation.images.{tCam}" for tCam in CAMS]:
            raise CheckpointError(str(self.mCfg.image_features))
        tActOk = self.mKind == ACT_KIND and self.mCfg.chunk_size == ACT_CHUNK and self.mNObs == 1
        tDitOk = self.mKind == DIT_KIND and self.mCfg.horizon == DIT_HORIZON and self.mNObs == 2
        if not (tActOk or tDitOk):
            raise CheckpointError(str(self.mCfg))
        tPm = pathlib.Path(ckpt) / "pretrained_model"
        from safetensors.numpy import load_file

        self.mStats = load_file(str(next(tPm.glob(NORMALIZER_GLOB))))
        tCk = pathlib.Path(ckpt)
        self.mGripMask = GSM_TAG in str(ckpt) or any((tDir / GSM_FILE).exists() for tDir in (tCk, tCk.parent, tCk.parent.parent))
        self.mGripFakeOpen = False  # 10-29 diagnostic (UI): feed the jaw state as OPEN to a copycat checkpoint
        # 11-06 PGRASP checkpoints: state 18, 19 = robot-side p_grasp, written into the state by the run loop (core.pgrasp)
        self.mPGrasp = checkpointVersion(ckpt)

    def setDitSteps(self, steps: int) -> None:
        self.mPolicy.objective.num_inference_steps = int(steps)

    def _jawState(self, state30):
        if self.mPGrasp and not self.mGripFakeOpen:
            return state30
        if not (self.mGripMask or self.mGripFakeOpen):
            return state30
        tS = np.array(state30, np.float32, copy=True)
        tS[GRIP_COLUMNS] = 0.0 if self.mGripMask else 1.0
        return tS

    def item(self, imgsRgb: dict, state30, prev=None) -> dict:
        if prev is not None:
            prev = (prev[0], self._jawState(prev[1]))
        if self.mNObs == 1:
            return makeItem(imgsRgb, self._jawState(state30))
        return makeItemPair(imgsRgb, self._jawState(state30), prev)

    def predict(self, item: dict, seed: int = 0, rtc: dict | None = None):
        """rtc (DiT only): dict(Y=(32, 20) denormalised contract rows k = 1..32 of the previous plan re-anchored at this observation,
        W=(32,) rtcWeights, g0=(2,) jaw at row 0, mode "guided" | "blend", beta) -> Real-Time Chunking sampler (_sampleRtc)."""
        import torch

        with torch.no_grad():
            tBatch = {
                tKey: (tValue.unsqueeze(0) if torch.is_tensor(tValue) else [tValue])
                for tKey, tValue in item.items()
                if tKey.startswith("observation") or tKey == "task"
            }
            tB = self.mPre(tBatch)
            torch.manual_seed(seed)
            if self.mKind == ACT_KIND:
                tNorm = self.mPolicy.predict_action_chunk(tB)
                tDenorm = self.mPost(tNorm.clone())
                return tNorm[0].float().cpu().numpy(), tDenorm[0].float().cpu().numpy()
            tPrepared = self.mPolicy._prepare_batch(dict(tB))
            tCond = self.mPolicy.observation_encoder.encode(tPrepared)
            tK = self.mNSamples
            if rtc is not None and tK == 1:
                tNorm = self._sampleRtc(tCond, rtc)
            else:
                tNorm = self.mPolicy.objective.conditional_sample(
                    self.mPolicy.noise_predictor, tK, tCond.expand(tK, -1) if tK > 1 else tCond
                )
            if hasattr(
                self.mPolicy, "grip_head"
            ):  # 10-29 GRIP_BCE checkpoints: jaw rows = the BCE head's P(open), in the [-1, 1] space
                tLogits = self.mPolicy.grip_logits(tCond.expand(tNorm.shape[0], -1) if tNorm.shape[0] > 1 else tCond)
                tNorm[..., GRIP_COLUMNS] = 2.0 * torch.sigmoid(tLogits).to(tNorm.dtype) - 1.0
            tDenorm = self.mPost(tNorm.clone()).float().cpu().numpy()
            tNorm = tNorm.float().cpu().numpy()
            return tNorm.mean(0)[1:], tDenorm.mean(0)[1:]

    def normalize(self, rows33) -> np.ndarray:
        """Denormalised contract rows (33, 20) -> the DiT's MIN_MAX action space (inverse of mPost)."""
        tLo = self.mStats["action.min"].astype(np.float64)
        tHi = self.mStats["action.max"].astype(np.float64)
        return 2.0 * (np.asarray(rows33, np.float64) - tLo) / np.maximum(tHi - tLo, NORMALIZE_EPS) - 1.0

    def _sampleRtc(self, cond, rtc: dict):
        """[10-29] Real-Time Chunking on the DDIM sampler.  Y = row 0 (identity + jaw) + the previous plan's rows, W = 1 on row 0 and
        rtc["W"] on rows 1..32.  "guided" = RTC's pseudo-inverse guidance (PiGDM) in epsilon form: x0_hat = (x - sqrt(1-a) eps) / sqrt(a),
        g = (dx0_hat/dx)^T W (Y - x0_hat) (one vector-Jacobian product per step), eps <- eps - sqrt(1-a) min(beta, 1 / r2) g, r2 = 1 - a
        (variance of x0 given x_t for unit-variance data).  "blend" = gradient-free: x0_hat <- x0_hat + W (Y - x0_hat), eps recomputed."""
        import torch

        tObjective = self.mPolicy.objective
        tNet = self.mPolicy.noise_predictor
        tScheduler = tObjective.noise_scheduler
        tDev = cond.device
        tDtype = next(tNet.parameters()).dtype
        tIdentity = T_to_pose9(np.eye(4))
        tY33 = np.vstack([np.r_[tIdentity, tIdentity, rtc["g0"]][None], rtc["Y"]])
        tYn = torch.from_numpy(self.normalize(tY33)).to(tDev, tDtype)[None]
        tW = torch.from_numpy(np.r_[1.0, rtc["W"]]).to(tDev, tDtype).view(1, -1, 1)
        tX = torch.randn((1, tObjective.horizon, tObjective.action_dim), dtype=tDtype, device=tDev)
        tScheduler.set_timesteps(tObjective.num_inference_steps)
        tAlphaCum = tScheduler.alphas_cumprod.to(tDev, tDtype)
        tBeta = float(rtc.get("beta", RTC_BETA_DEFAULT))
        tGuided = rtc.get("mode", RTC_GUIDED) == RTC_GUIDED
        for tStep in tScheduler.timesteps:
            tA = tAlphaCum[int(tStep)]
            tTt = torch.full((1,), int(tStep), dtype=torch.long, device=tDev)
            if tGuided:
                with torch.enable_grad():
                    tXg = tX.detach().requires_grad_(True)
                    tEps = tNet(tXg, tTt, conditioning_vec=cond)
                    tX0 = (tXg - (1 - tA).sqrt() * tEps) / tA.sqrt()
                    (tGrad,) = torch.autograd.grad(tX0, tXg, grad_outputs=(tW * (tYn - tX0)).detach())
                tEps = tEps.detach() - (1 - tA).sqrt() * min(tBeta, 1.0 / max(float(1 - tA), MIN_NOISE_VAR)) * tGrad
            else:
                tEps = tNet(tX, tTt, conditioning_vec=cond)
                tX0 = (tX - (1 - tA).sqrt() * tEps) / tA.sqrt()
                tX0 = tX0 + tW * (tYn - tX0)
                tEps = (tX - tA.sqrt() * tX0) / (1 - tA).sqrt().clamp_min(MIN_NOISE_VAR)
            tX = tScheduler.step(tEps, tStep, tX).prev_sample
        return tX

    def workspace(self, margin: float = WORKSPACE_MARGIN_M) -> dict:
        """Training TCP range (pose_abs min/max of the TRAIN root, from the checkpoint's normalizer) +- margin, per arm xyz."""
        tLo, tHi = self.mStats["pose_abs.min"], self.mStats["pose_abs.max"]
        return {tSide: (tLo[POSE_SLICE[tSide]][:3] - margin, tHi[POSE_SLICE[tSide]][:3] + margin) for tSide in SIDES}
