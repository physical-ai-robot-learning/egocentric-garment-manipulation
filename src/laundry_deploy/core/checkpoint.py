"""Loads an ACT or Multi-Task DiT checkpoint of the tag-rig policies with its own pre / post-processors (deploy and offline
evaluation). Multi-Task DiT is sampled with DDIM 10 (the offline-eval sampler); GRIP_BCE checkpoints get their jaw head first.

Input: a checkpoint step folder (or its pretrained_model/). Output: (config, policy, preprocessor, postprocessor).
"""

from __future__ import annotations
import pathlib

DDIM_STEPS = 10  # the offline-eval sampler of the DiT checkpoints
PRETRAINED_DIR = "pretrained_model"
GRIP_HEAD_PREFIX = "grip_head."  # 10-29 GRIP_BCE checkpoints carry these weights


def loadPolicy(ckpt, device: str):
    """-> (cfg, policy, pre, post) on `device`, eval mode."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors

    tPm = pathlib.Path(ckpt)
    tPm = tPm / PRETRAINED_DIR if (tPm / PRETRAINED_DIR).is_dir() else tPm
    tCfg = PreTrainedConfig.from_pretrained(str(tPm))
    tCfg.device = device
    if tCfg.type == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy as tPolicyCls
    else:
        import lerobot.policies.multi_task_dit.modeling_multi_task_dit as tMd
        from safetensors import safe_open

        with safe_open(str(tPm / "model.safetensors"), "pt") as tFile:
            if any(tKey.startswith(GRIP_HEAD_PREFIX) for tKey in tFile.keys()):  # build the jaw head before loading
                from laundry_training import installGripBce

                installGripBce(tMd)
        tPolicyCls = tMd.MultiTaskDiTPolicy
    tPolicy = tPolicyCls.from_pretrained(str(tPm), config=tCfg).to(device).eval()
    tPre, tPost = make_pre_post_processors(
        tCfg, pretrained_path=str(tPm), preprocessor_overrides={"device_processor": {"device": device}}
    )
    if tCfg.type != "act":
        from diffusers import DDIMScheduler

        tPolicy.objective.noise_scheduler = DDIMScheduler.from_config(tPolicy.objective.noise_scheduler.config)
        tPolicy.objective.num_inference_steps = DDIM_STEPS
    return tCfg, tPolicy, tPre, tPost
