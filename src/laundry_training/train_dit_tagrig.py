"""lerobot-train for Multi-Task DiT on the tag-rig laundry dataset (training stage; [2026-10-22], was
training/train_dit_tagrig.py).

rel_chunk window (horizon 33, n_obs 2, row 0 identity) exactly as training/train_dit.py, but with the train-only FIXED stats of
stats.installStats (std/range floors, grippers min 0 / max 1 -> [-1, 1]), the loss masks / episode weights of loss_masks and the
metrics.jsonl logging of metrics. The Soft-Fold launcher (train_dit.py / smoke_train.sh) is untouched.
GRIP_BCE=1 adds the BCE jaw head (grip_head); without it the gripper dims are denoised like the other dims (MSE).

    python -m laundry_training.train_dit_tagrig <stock lerobot-train flags>
Environment: SCHED_HORIZON (cosine LR horizon, default --steps), GRAD_CLIP (overrides the preset grad clip; 0 = none) and the
switches of env_options / loss_masks / metrics.
"""
from __future__ import annotations

import os
import sys

from .rel_chunk import HORIZON

N_OBS_STEPS = 2


def _makeDatasetContract(lt, makeDataset):
    from .datasets import patchDitDataset
    from .loss_masks import installMasks, installWeights
    from .stats import installStats

    def make_dataset_contract(cfg):
        if getattr(cfg.policy, "horizon", None) != HORIZON or getattr(cfg.policy, "n_obs_steps", None) != N_OBS_STEPS:
            raise SystemExit(f"contract chunk needs policy.horizon={HORIZON} and policy.n_obs_steps={N_OBS_STEPS}")
        lt._TAGRIG_OUT = str(cfg.output_dir)              # 10-29: DEADLINE_TS extra checkpoint (metrics.installMetrics)
        lt._TAGRIG_CFG = cfg
        # same window transform as rel_chunk.patchDataset, picklable
        return patchDitDataset(installMasks(installWeights(installStats(makeDataset(cfg), "dit")), "dit"))
    return make_dataset_contract


def installPatches():
    """patch lerobot_train for this recipe: dataset factory, LR-schedule horizon, grad clip, metrics hook, BCE jaw head
    -> lerobot_train"""
    import lerobot.policies.multi_task_dit.modeling_multi_task_dit as MD
    import lerobot.scripts.lerobot_train as LT
    from lerobot.optim.schedulers import DiffuserSchedulerConfig

    from . import env_options
    from .grip_head import installGripBce
    from .metrics import installMetrics

    LT.make_dataset = _makeDatasetContract(LT, LT.make_dataset)
    # resumability (10-22): the DiT cosine LR schedule is built over SCHED_HORIZON steps (default = --steps). The 500k runs use
    # SCHED_HORIZON=1000000 so lr(500k) = 0.5 * peak and a resume with --steps=1000000 (same SCHED_HORIZON) continues the same
    # curve.
    tBuild = DiffuserSchedulerConfig.build
    DiffuserSchedulerConfig.build = lambda self, optimizer, num_training_steps: tBuild(
        self, optimizer, int(os.environ.get("SCHED_HORIZON", num_training_steps)))
    # 10-29 user ("클리핑 없이", "쓸데없이 이런거 걸지 말고"): lerobot's AdamConfig preset clips the global grad norm at 10 by default.
    # GRAD_CLIP=<x> overrides it (0 = no clipping; the norm is still computed and logged). Saved in train_config.json, so a resume
    # keeps it.
    if os.environ.get("GRAD_CLIP") is not None:
        import lerobot.policies.multi_task_dit.configuration_multi_task_dit as MC
        tPreset = MC.MultiTaskDiTConfig.get_optimizer_preset

        def presetNoClip(self):
            tOpt = tPreset(self)
            tOpt.grad_clip_norm = float(os.environ["GRAD_CLIP"])
            return tOpt
        MC.MultiTaskDiTConfig.get_optimizer_preset = presetNoClip
    installMetrics(LT, MD, "mse_loss")
    if env_options.GRIP_BCE:
        installGripBce(MD)                      # 10-29 UMI-style recipe: BCE jaw head
    return LT


def main(argv: list[str] | None = None) -> int:
    if argv is not None:
        sys.argv = [sys.argv[0], *argv]         # lerobot_train parses sys.argv itself
    installPatches().main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
