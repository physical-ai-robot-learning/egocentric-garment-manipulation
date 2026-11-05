"""LeRobot datasets that serve the relative contract chunk to the tag-rig trainers (training stage; was part of
training/tagrig_common.py). Module-level classes so they pickle under spawn DataLoader workers.

ActRelDataset  action (32, 20) absolute -> contract rows k = 1..32 anchored at pose_abs(t)
DitRelDataset  action window (33, 20) -> rows 0..32 (row 0 identity), the twin of rel_chunk.patchDataset's class;
               GRIP_STATE_MASK=1 hides the jaw state, AUG=1 augments the camera images (env_options)
"""
from __future__ import annotations

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from . import env_options
from .augment import augImage
from .layout import GRIP_STATE_DIMS
from .rel_chunk import H_ACT, HORIZON, actRelative, toContract33

IMAGE_PREFIX = "observation.images."
PAD_SUFFIX = "_is_pad"
GLOBAL_SUFFIX = ".global"
MIN_IMAGE_DIMS = 3                   # (C, H, W) or more


class ActRelDataset(LeRobotDataset):
    """action (32, 20) abs -> contract rows k = 1..32 anchored at pose_abs(t)"""

    def __getitem__(self, idx):
        tItem = super().__getitem__(idx)
        tAction = tItem["action"]
        if tAction.ndim == 2 and tAction.shape[0] == H_ACT:
            tRel = actRelative(tItem["pose_abs"].numpy().astype(np.float64), tAction.numpy().astype(np.float64))
            tItem["action"] = torch.from_numpy(tRel.astype(np.float32))
        return tItem


class DitRelDataset(LeRobotDataset):
    """window (33, 20) -> rows 0..32 (row 0 identity).
    [10-29] GRIP_STATE_MASK=1: the jaw state in observation.state is fixed to 0 (both obs steps). Exp 1/2 learned to COPY the
    input jaw state (deploy: closed in -> closed out forever, the jaw never opened); with the state hidden the jaw timing must
    come from vision."""

    def __getitem__(self, idx):
        tItem = super().__getitem__(idx)
        tAction = tItem["action"]
        if tAction.ndim == 2 and tAction.shape[0] == HORIZON:
            tItem["action"] = torch.from_numpy(toContract33(tAction.numpy()))
        if env_options.GRIP_STATE_MASK and "observation.state" in tItem:
            tState = tItem["observation.state"].clone()
            tState[..., list(GRIP_STATE_DIMS)] = 0.0
            if env_options.PGRASP:
                # 11-05 FLAT250-PGRASP: obs steps (t - 1 frame, t) <- p_grasp of those frames; frame 0 of an episode repeats its own
                # value (that obs step is the padded copy of frame 0 too)
                if getattr(self, "mPgrasp", None) is None:
                    self.mPgrasp = np.load(self.root / "meta" / f"pgrasp_{env_options.PGRASP}.npy", mmap_mode="r")
                tIdx = int(tItem["index"])
                tFrame = int(tItem["frame_index"])
                tRows = [tIdx - 1 if tFrame > 0 else tIdx, tIdx][-tState.shape[0]:] if tState.dim() == 2 else [tIdx]
                tPgrasp = torch.from_numpy(np.asarray(self.mPgrasp[tRows], np.float32))
                tState[..., list(GRIP_STATE_DIMS)] = tPgrasp.reshape(tState.shape[:-1] + (len(GRIP_STATE_DIMS),))
            tItem["observation.state"] = tState
        if env_options.AUG:
            # 11-01: skip the <cam>_is_pad masks
            tKeys = [tK for tK in tItem if tK.startswith(IMAGE_PREFIX) and not tK.endswith(PAD_SUFFIX)
                     and torch.is_tensor(tItem[tK])
                     and tItem[tK].dim() >= MIN_IMAGE_DIMS]
            tP = env_options.AUG_P
            for tKey in tKeys:
                tGlobal = tKey.endswith(GLOBAL_SUFFIX)
                tItem[tKey] = augImage(tItem[tKey], tP["AUG_G_SHIFT"] if tGlobal else tP["AUG_W_SHIFT"],
                                       tP["AUG_G_ROT"] if tGlobal else tP["AUG_W_ROT"],
                                       tP["AUG_G_ZOOM"] if tGlobal else tP["AUG_W_ZOOM"])
        return tItem


def patchActDataset(ds):
    """a plain LeRobotDataset -> ActRelDataset, in place (was tagrig_common.patch_act_dataset)"""
    assert type(ds) is LeRobotDataset, type(ds)
    ds.__class__ = ActRelDataset
    return ds


def patchDitDataset(ds):
    """a plain LeRobotDataset -> DitRelDataset, in place (was tagrig_common.patch_dit_dataset)"""
    assert type(ds) is LeRobotDataset, type(ds)
    ds.__class__ = DitRelDataset
    return ds
