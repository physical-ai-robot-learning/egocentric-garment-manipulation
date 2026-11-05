"""Environment switches of the tag-rig trainers that are read once, when the package is imported (training stage).

Each one is a documented run setting of the launch scripts; names and defaults are part of the recipe and never change:

    AUG=1               train-time image augmentation of DitRelDataset (11-01); sizes from the AUG_* values below
    AUG_G_SHIFT ...     augmentation sizes (see AUG_DEFAULTS); AUG_B / AUG_C default to AUG_BC (11-02)
    GRIP_STATE_MASK=1   the jaw state in observation.state is fixed to 0 (10-29)
    GRIP_BCE=1          BCE jaw head on the Multi-Task DiT (10-29); GRIP_BCE_W = its loss weight
    WEIGHT_GRIP_ONLY=1  the dataset `weight` column scales only the BCE jaw loss (10-30)
    GRIP_HEAD_ONLY=1    only grip_head trains, loss = BCE only; GRIP_HEAD_RESET=1 re-initialises it at the first forward (10-30)
    GRAD_ACCUM=K        an update = K micro-batches of the configured batch (10-29)

Switches read at call time stay with their readers: MASK_CROSS_GAP and USE_ACTION_MASK / USE_EPISODE_WEIGHT (loss_masks,
metrics), METRICS_EVERY / GRAD_SPLIT_EVERY / RANK / RESUME / DEADLINE_TS (metrics), GRAD_CLIP / SCHED_HORIZON
(train_dit_tagrig), MIX_SPEC / MIX_STATS (train_dit_mix).
"""
from __future__ import annotations

import os

ON = "1"


def _flag(name: str) -> bool:
    return os.environ.get(name, "0") == ON


GRIP_STATE_MASK = _flag("GRIP_STATE_MASK")
# 11-05 FLAT250-PGRASP: PGRASP=<version> (e.g. grasp-v2-W0.4) -> the GRIP_STATE_MASK jaw slots (state 18, 19) carry the causal
# L / R p_grasp of <dataset root>/meta/pgrasp_<version>.npy (per global frame index; written by python -m laundry_grasp.sidefile)
# instead of 0. Needs GRIP_STATE_MASK=1.
PGRASP = os.environ.get("PGRASP", "")

# 11-01 user "이미지 증강 넣어보자": TRAIN-time image augmentation (AUG=1), applied per sample in the dataset (CPU workers, images in
# [0, 1], before the policy normaliser; the evaluator / deploy never see it). Sized to the measured robot-vs-UMI gaps: the Orbbec
# moved 5-13 px / 0.5-1 deg between days -> global: shift +-AUG_G_SHIFT of the size, rot +-AUG_G_ROT deg; robot wrist cams mount
# 11-19 deg / 25-44 mm off the UMI -> wrists: shift +-AUG_W_SHIFT, rot +-AUG_W_ROT, zoom +-AUG_W_ZOOM; every camera: brightness /
# contrast +-AUG_BC, per-channel gain +-AUG_CH. Labels are untouched (the image changes, the action does not).
AUG = _flag("AUG")
AUG_DEFAULTS = dict(AUG_G_SHIFT=0.045, AUG_G_ROT=2.0, AUG_G_ZOOM=0.02, AUG_W_SHIFT=0.08, AUG_W_ROT=5.0, AUG_W_ZOOM=0.05,
                    AUG_BC=0.15, AUG_CH=0.05, AUG_SAT=0.0, AUG_HUE=0.0)
AUG_P = {tKey: float(os.environ.get(tKey, tDefault)) for tKey, tDefault in AUG_DEFAULTS.items()}
# 11-02 user (UMI appendix "같게"): UMI ColorJitter = brightness 0.3 / contrast 0.4 / saturation 0.5 / hue 0.08 -> AUG_B / AUG_C
# (default AUG_BC), AUG_SAT, AUG_HUE (default 0 = the 11-01 behaviour); UMI RandomCrop 0.95 ~ AUG_*_ZOOM 0.05 + shift 0.025,
# RandomRotation +-5 deg ~ AUG_*_ROT 5.
AUG_P["AUG_B"] = float(os.environ.get("AUG_B", AUG_P["AUG_BC"]))
AUG_P["AUG_C"] = float(os.environ.get("AUG_C", AUG_P["AUG_BC"]))

GRIP_BCE = _flag("GRIP_BCE")
GRIP_BCE_W = float(os.environ.get("GRIP_BCE_W", "1.0"))
# 10-30 user "그리퍼만 weight를 더 주게": WEIGHT_GRIP_ONLY=1 -> the per-sample dataset `weight` column scales ONLY the BCE jaw loss
# (the motor-jaw flattening demos are built with EP_WEIGHT=3); the diffusion MSE (pose) stays unweighted.
WEIGHT_GRIP_ONLY = _flag("WEIGHT_GRIP_ONLY")
# 10-30 user plan A (gripper relabel): GRIP_HEAD_ONLY=1 -> everything except grip_head frozen, observation encoder run under
# no_grad, diffusion loss not computed (loss = BCE only). GRIP_HEAD_RESET=1 -> grip_head re-initialised at the first forward
# (after the checkpoint weights are loaded), so old-head vs new-head comparisons start from the same arm weights and a fresh head.
GRIP_HEAD_ONLY = _flag("GRIP_HEAD_ONLY")
GRIP_HEAD_RESET = _flag("GRIP_HEAD_RESET")

GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "1"))
