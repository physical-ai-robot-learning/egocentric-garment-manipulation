"""Train-time image augmentation of the tag-rig DiT dataset (training stage; was part of training/tagrig_common.py).

One random similarity (shift, rotation, zoom; border padding) + brightness / contrast / per-channel gain per sample, the SAME for
all frames of the sample (n_obs_steps); optional saturation (AUG_SAT) and hue (AUG_HUE) jitter. Sizes come from env_options.AUG_P.
Images in, images out: float [0, 1] or uint8 [0, 255] (same dtype / range as the input). Labels are never touched.
"""
from __future__ import annotations

import math

import torch

from . import env_options

# 11-02: TF.adjust_hue (RGB->HSV->RGB) cost 62 ms per camera sample on the CPU workers -> the 4xH100 run was data-bound at ~3
# s/update. Hue shift as a rotation of the chroma plane in YIQ (one 3x3 matrix, ~1 ms); hue_factor in turns like adjust_hue (0.08
# = +-28.8 deg).
_YIQ = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], dtype=torch.float32)
_YIQ_INV = torch.linalg.inv(_YIQ)
UINT8_MAX = 255.0


def hueRotate(images, hueFactor: float):
    """images (N, 3, H, W) float in [0, 1]; hueFactor in turns (was tagrig_common.hue_rotate)"""
    tA = 2.0 * math.pi * hueFactor
    tRot = torch.tensor([[1.0, 0.0, 0.0], [0.0, math.cos(tA), -math.sin(tA)], [0.0, math.sin(tA), math.cos(tA)]])
    tM = (_YIQ_INV @ tRot @ _YIQ).to(images.dtype)
    return torch.einsum("ij,njhw->nihw", tM, images).clamp(0.0, 1.0)


def augImage(x, shift: float, rotDeg: float, zoom: float):
    """x (..., C, H, W) float in [0, 1] or uint8: one random similarity (shift, rotation, zoom; border padding) + brightness /
    contrast / channel gain, the SAME for all frames of this sample (n_obs_steps) (was tagrig_common.aug_image)"""
    tP = env_options.AUG_P
    # 11-01: the 4090 video backend yields uint8 0..255 (the Mac pyav path float 0..1) -> work in float, return the input
    # dtype/range
    tDtype = x.dtype
    tIsInt = not torch.is_floating_point(x)
    x = x.float() / UINT8_MAX if tIsInt else x
    tShape = x.shape
    tY = x.reshape(-1, *tShape[-3:])
    tU = lambda r: (torch.rand(()) * 2 - 1) * r
    tTheta = math.radians(float(tU(rotDeg)))
    tZoom = 1.0 + float(tU(zoom))
    tTx, tTy = float(tU(2 * shift)), float(tU(2 * shift))                     # affine_grid units: [-1, 1]
    tAffine = torch.tensor([[math.cos(tTheta) / tZoom, -math.sin(tTheta) / tZoom, tTx],
                            [math.sin(tTheta) / tZoom, math.cos(tTheta) / tZoom, tTy]], dtype=tY.dtype).expand(tY.shape[0], 2, 3)
    tGrid = torch.nn.functional.affine_grid(tAffine, list(tY.shape), align_corners=False)
    tY = torch.nn.functional.grid_sample(tY, tGrid, mode="bilinear", padding_mode="border", align_corners=False)
    tB, tC = 1.0 + float(tU(tP["AUG_B"])), 1.0 + float(tU(tP["AUG_C"]))
    tCh = 1.0 + (torch.rand(3, dtype=tY.dtype) * 2 - 1) * tP["AUG_CH"]
    tMean = tY.mean(dim=(-3, -2, -1), keepdim=True)
    tY = ((tY - tMean) * tC + tMean) * tB * tCh.view(1, 3, 1, 1)
    if tP["AUG_SAT"] > 0 or tP["AUG_HUE"] > 0:
        import torchvision.transforms.functional as TF
        tY = tY.clamp(0.0, 1.0)
        if tP["AUG_SAT"] > 0:
            tY = TF.adjust_saturation(tY, 1.0 + float(tU(tP["AUG_SAT"])))
        if tP["AUG_HUE"] > 0:
            tY = hueRotate(tY, float(tU(tP["AUG_HUE"])))
    tY = tY.clamp(0.0, 1.0).reshape(tShape)
    return (tY * UINT8_MAX).round().to(tDtype) if tIsInt else tY.to(tDtype)
