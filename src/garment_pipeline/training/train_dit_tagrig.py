#!/usr/bin/env python3
"""[2026-10-22] lerobot-train for Multi-Task DiT on the tag-rig laundry dataset: rel_chunk window (horizon 33, n_obs 2, row 0 identity)
exactly as training/train_dit.py, but with the train-only FIXED stats of training/tagrig_common.py (std/range floors, grippers
min 0 / max 1 -> [-1, 1]) and metrics.jsonl logging.  The Soft-Fold launcher (train_dit.py / smoke_train.sh) is untouched.
NOTE: Multi-Task DiT has no BCE head; the gripper dims are denoised like the other dims (MSE on the diffusion target)."""
import pathlib, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(pathlib.Path.home() / "laundry_system"))
import lerobot.scripts.lerobot_train as LT  # noqa: E402
import lerobot.policies.multi_task_dit.modeling_multi_task_dit as MD  # noqa: E402
from rel_chunk import HORIZON  # noqa: E402
from tagrig_common import install_masks, install_weights, install_metrics, install_stats, patch_dit_dataset  # noqa: E402

_make = LT.make_dataset


def make_dataset_contract(cfg):
    if getattr(cfg.policy, "horizon", None) != HORIZON or getattr(cfg.policy, "n_obs_steps", None) != 2:
        raise SystemExit(f"contract chunk needs policy.horizon={HORIZON} and policy.n_obs_steps=2")
    LT._TAGRIG_OUT = str(cfg.output_dir); LT._TAGRIG_CFG = cfg     # 10-29: DEADLINE_TS extra checkpoint (tagrig_common.install_metrics)
    return patch_dit_dataset(install_masks(install_weights(install_stats(_make(cfg), "dit")), "dit"))   # same window transform as rel_chunk.patch_dataset, picklable


LT.make_dataset = make_dataset_contract

# resumability (10-22): the DiT cosine LR schedule is built over SCHED_HORIZON steps (default = --steps).  The 500k runs use
# SCHED_HORIZON=1000000 so lr(500k) = 0.5 * peak and a resume with --steps=1000000 (same SCHED_HORIZON) continues the same curve.
import os  # noqa: E402
from lerobot.optim.schedulers import DiffuserSchedulerConfig  # noqa: E402
_build = DiffuserSchedulerConfig.build
DiffuserSchedulerConfig.build = lambda self, optimizer, num_training_steps: _build(self, optimizer, int(os.environ.get("SCHED_HORIZON", num_training_steps)))
# 10-29 user ("클리핑 없이", "쓸데없이 이런거 걸지 말고"): lerobot's AdamConfig preset clips the global grad norm at 10 by default.
# GRAD_CLIP=<x> overrides it (0 = no clipping; the norm is still computed and logged).  Saved in train_config.json, so a resume keeps it.
if os.environ.get("GRAD_CLIP") is not None:
    import lerobot.policies.multi_task_dit.configuration_multi_task_dit as MC  # noqa: E402
    _preset = MC.MultiTaskDiTConfig.get_optimizer_preset
    def _preset_noclip(self):
        o = _preset(self); o.grad_clip_norm = float(os.environ["GRAD_CLIP"]); return o
    MC.MultiTaskDiTConfig.get_optimizer_preset = _preset_noclip
install_metrics(LT, MD, "mse_loss")
from tagrig_common import GRIP_BCE, install_grip_bce  # noqa: E402
if GRIP_BCE: install_grip_bce(MD)                      # 10-29 UMI-style recipe: BCE jaw head

if __name__ == "__main__":
    LT.main()
