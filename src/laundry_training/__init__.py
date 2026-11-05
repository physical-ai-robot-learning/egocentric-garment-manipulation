"""Training stage of the laundry pipeline: Multi-Task DiT / ACT on the tag-rig LeRobot datasets (relative contract chunks,
train-only fixed stats, per-arm loss masks, episode weights, image augmentation, BCE jaw head, metrics.jsonl).

Interface used by other packages:
    ActRelDataset, DitRelDataset   LeRobotDataset subclasses serving the relative chunk (constructor = LeRobotDataset's)
    installGripBce(policyModule)   add the BCE jaw head (grip_head / grip_logits) to a Multi-Task DiT policy module

The names are loaded on first access (the dataset classes need lerobot, the jaw head torch), so importing the package needs
neither; the light modules (layout, env_options, ...) import without them.

Modules: env_options (import-time env switches), layout (vector index layout), rel_chunk (relative windows), stats (fixed stats),
loss_masks (chunk masks + episode weights), augment (image augmentation), datasets (ActRelDataset / DitRelDataset), grip_head (BCE
jaw head), metrics (update hook + metrics.jsonl), train_dit_tagrig / train_dit_mix (launchers).
"""
from __future__ import annotations

_LAZY = {"ActRelDataset": "datasets", "DitRelDataset": "datasets", "installGripBce": "grip_head"}

__all__ = ["ActRelDataset", "DitRelDataset", "installGripBce"]


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module
        return getattr(import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
