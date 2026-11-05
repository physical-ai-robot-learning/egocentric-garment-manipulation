"""BCE jaw head for the Multi-Task DiT (training stage; was tagrig_common.install_grip_bce).

[2026-10-29] UMI-style recipe (user plan "gripper loss: BCE 별도 head/loss"): Multi-Task DiT gets `grip_head`, an MLP on the
observation conditioning vector -> (horizon, 2) jaw logits, trained with BCEWithLogits against the jaw rows of the contract chunk
(action dims 18/19, MIN_MAX-normalised [-1, 1] -> target (x + 1) / 2 in [0, 1]), masked like the jaw dims (chunk mask + padding).
The diffusion MSE no longer sees the jaw dims (the update hook zeroes their mask); at inference the denoised jaw rows are REPLACED
by 2 sigmoid(logit) - 1 (deploy core.PolicyRunner / evaluation tagrig_offline_eval). Installed when GRIP_BCE=1 at training;
loading checkpoints: core / eval install it when the safetensors contain grip_head.* (strict=False loading otherwise drops it).
The names grip_head / grip_logits are part of the checkpoint and of the deploy code and never change.
"""
from __future__ import annotations

import torch

from . import env_options
from .layout import GRIP_A
from .loss_masks import BATCH

GRIP_HIDDEN = 512                    # hidden width of the jaw MLP
N_JAWS = 2
JAW_OPEN_P = 0.5                     # accuracy: target above this = open


def installGripBce(policyModule) -> None:
    """patch policyModule.MultiTaskDiTPolicy (once): grip_head in __init__, grip_logits, forward = diffusion loss +
    GRIP_BCE_W * BCE"""
    import torch.nn as nn
    tPolicy = policyModule.MultiTaskDiTPolicy
    if getattr(tPolicy, "_grip_bce", False):
        return
    tInit0 = tPolicy.__init__

    def __init__(self, config, **kw):
        tInit0(self, config, **kw)
        tDim = self.observation_encoder.conditioning_dim
        self.grip_head = nn.Sequential(nn.Linear(tDim, GRIP_HIDDEN), nn.GELU(), nn.Linear(GRIP_HIDDEN, config.horizon * N_JAWS))

    def grip_logits(self, cond):
        return self.grip_head(cond).view(cond.shape[0], self.config.horizon, N_JAWS)

    def forward(self, batch):
        if env_options.GRIP_HEAD_ONLY and not getattr(self, "_head_only_set", False):
            for tName, tParam in self.named_parameters():
                tParam.requires_grad_(tName.startswith("grip_head."))
            if env_options.GRIP_HEAD_RESET:
                for tModule in self.grip_head:
                    if hasattr(tModule, "reset_parameters"):
                        tModule.reset_parameters()
            self._head_only_set = True
            tTrainable = sum(tP.numel() for tP in self.parameters() if tP.requires_grad)
            print(f"[tagrig] GRIP_HEAD_ONLY: trainable {tTrainable} params (grip_head), reset {env_options.GRIP_HEAD_RESET}",
                  flush=True)
        batch = self._prepare_batch(batch)
        if env_options.GRIP_HEAD_ONLY:
            with torch.no_grad():
                tCond = self.observation_encoder.encode(batch)
            tLoss = None
        else:
            tCond = self.observation_encoder.encode(batch)
            tLoss = self.objective.compute_loss(self.noise_predictor, batch, tCond)
        tLogits = grip_logits(self, tCond).float()
        tTarget = ((batch["action"][..., GRIP_A].float() + 1.0) / 2.0).clamp(0.0, 1.0)
        tM = BATCH.get("Mg")
        tM = torch.ones_like(tTarget) if tM is None or tM.shape != tTarget.shape else tM.to(tTarget.device, tTarget.dtype)
        tPad = batch.get("action_is_pad")
        if tPad is not None:
            tM = tM * (~tPad.to(tTarget.device)).unsqueeze(-1).to(tTarget.dtype)
        tW = BATCH.get("w")
        tBceAll = torch.nn.functional.binary_cross_entropy_with_logits(tLogits, tTarget, reduction="none")
        if env_options.WEIGHT_GRIP_ONLY and tW is not None and tW.shape[0] == tM.shape[0]:
            tMw = tM * tW.to(tM.device, tM.dtype).reshape(-1, *([1] * (tM.dim() - 1)))
            tBce = (tBceAll * tMw).sum() / tM.sum().clamp_min(1.0)
        else:
            tBce = (tBceAll * tM).sum() / tM.sum().clamp_min(1.0)
        BATCH.setdefault("groups", {})["grip_bce"] = float(tBce.detach())
        tAcc = (((tLogits > 0).float() == (tTarget > JAW_OPEN_P).float()).float() * tM).sum() / tM.sum().clamp_min(1.0)
        BATCH["groups"]["grip_acc"] = float(tAcc)
        if tLoss is None:
            return env_options.GRIP_BCE_W * tBce, None
        return tLoss + env_options.GRIP_BCE_W * tBce.to(tLoss.dtype), None

    tPolicy.__init__ = __init__
    tPolicy.forward = forward
    tPolicy.grip_logits = grip_logits
    tPolicy._grip_bce = True
    print(f"[tagrig] GRIP_BCE: grip_head (cond -> horizon x 2 logits), BCE weight {env_options.GRIP_BCE_W}, "
          "jaw dims removed from the diffusion MSE", flush=True)
