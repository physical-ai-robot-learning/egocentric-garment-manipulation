"""The jaw-ROI grasp classifier: ImageNet ResNet-18 fine-tuned to one logit = P(not held) (was grasp_cnn_v1.train_fold / predict and
grasp_render_prob.train_all).

Recipe: per gold close event, 4 crops at close + {0.3, 0.5, 0.7, 0.9} s (clipped to 0.05 s before the open, at least 0.1 s after the
close), label 1 for empty; since 11-05 also 4 crops of the open approach before the close (close - 0.6 / 0.45 / 0.3 / 0.15 s) labelled
1 (not held), so the classifier itself outputs ~0 grasp on open frames. 8 epochs, batch 32, AdamW 1e-4 / wd 1e-2, cosine schedule,
BCE with pos_weight = negatives / positives, random horizontal flip and brightness / contrast jitter per batch, seed 0, MPS.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torchvision

from .crops import MODEL_PX
from .gold import LOOSE, SLIP

EVENT_FRAME_OFFSETS_S = (0.3, 0.5, 0.7, 0.9)
APPROACH_OFFSETS_S = (0.6, 0.45, 0.3, 0.15)
LAST_FRAME_MARGIN_S = 0.05            # event frames stay this far before the open ...
MIN_EVENT_OFFSET_S = 0.1              # ... but at least this far after the close
DEVICE = "mps"
EPOCHS = 8
BATCH = 32
SEED = 0
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
FLIP_P = 0.5
CONTRAST_RANGE = (0.8, 1.2)
BRIGHTNESS_RANGE = (-0.2, 0.2)
MIN_POSITIVE_SHARE = 1e-3
FEATURES = 512                        # ResNet-18 penultimate width
RESNET18_WEIGHTS = "resnet18-f37072fd.pth"   # torchvision ImageNet weights in the torch hub cache
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
NS_PER_S = 1e9


def eventFrameTimes(tClose: int, tOpen: int) -> list[float]:
    """times of the 4 training / scoring crops of a close event"""
    return [tClose + min(tDt * NS_PER_S, max(tOpen - tClose - LAST_FRAME_MARGIN_S * NS_PER_S, MIN_EVENT_OFFSET_S * NS_PER_S))
            for tDt in EVENT_FRAME_OFFSETS_S]


def approachFrameTimes(tClose: int) -> list[float]:
    """times of the 4 open-approach crops before a close (the not-held negatives)"""
    return [tClose - tDt * NS_PER_S for tDt in APPROACH_OFFSETS_S]


def toTensor(images: np.ndarray) -> torch.Tensor:
    """(n, 224, 224, 3) uint8 RGB -> ImageNet-normalized NCHW float32"""
    return torch.tensor(np.ascontiguousarray(((images.astype(np.float32) / 255 - IMAGENET_MEAN) / IMAGENET_STD).transpose(0, 3, 1, 2)))


def trainFold(xTrain: np.ndarray, yTrain: np.ndarray, device: str = DEVICE) -> torch.nn.Module:
    """fine-tune on (n, frames, 224, 224, 3) crops with one 0/1 label per sample (1 = not held); every frame carries its sample's label"""
    torch.manual_seed(SEED)
    tModel = torchvision.models.resnet18()
    tModel.load_state_dict(torch.load(Path(torch.hub.get_dir()) / "checkpoints" / RESNET18_WEIGHTS))
    tModel.fc = torch.nn.Linear(FEATURES, 1)
    tModel = tModel.to(device)
    tImages = xTrain.reshape(-1, MODEL_PX, MODEL_PX, 3)
    tLabels = np.repeat(yTrain, xTrain.shape[1]).astype(np.float32)
    tPosWeight = torch.tensor((1 - tLabels.mean()) / max(tLabels.mean(), MIN_POSITIVE_SHARE), device=device)
    tOpt = torch.optim.AdamW(tModel.parameters(), LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    tBatches = int(np.ceil(len(tImages) / BATCH))
    tSched = torch.optim.lr_scheduler.CosineAnnealingLR(tOpt, EPOCHS * tBatches)
    tRng = np.random.default_rng(SEED)
    for _ in range(EPOCHS):
        tModel.train()
        for tIdx in np.array_split(tRng.permutation(len(tImages)), tBatches):
            tBatch = tImages[tIdx].copy()
            if tRng.random() < FLIP_P:
                tBatch = tBatch[:, :, ::-1]
            tX = toTensor(tBatch).to(device)
            tX = tX * float(tRng.uniform(*CONTRAST_RANGE)) + float(tRng.uniform(*BRIGHTNESS_RANGE))
            tLoss = torch.nn.functional.binary_cross_entropy_with_logits(
                tModel(tX).squeeze(1), torch.tensor(tLabels[tIdx], device=device), pos_weight=tPosWeight)
            tOpt.zero_grad()
            tLoss.backward()
            tOpt.step()
            tSched.step()
    return tModel


@torch.no_grad()
def predict(model: torch.nn.Module, x: np.ndarray, device: str = DEVICE) -> np.ndarray:
    """P(not held) per sample of (n, frames, 224, 224, 3) crops = mean over its frames (float32)"""
    model.eval()
    tP = torch.sigmoid(model(toTensor(x.reshape(-1, MODEL_PX, MODEL_PX, 3)).to(device)).squeeze(1)).cpu().numpy()
    return tP.reshape(len(x), -1).mean(1)


def loadModel(path: Path, device: str = DEVICE) -> torch.nn.Module:
    """a saved state dict (model.pt of a frozen generator) -> eval model on `device`"""
    tModel = torchvision.models.resnet18()
    tModel.fc = torch.nn.Linear(FEATURES, 1)
    tModel.load_state_dict(torch.load(path, map_location="cpu"))
    return tModel.to(device).eval()


def trainingSet(goldRows: list[dict], overrides: dict, cropperFor: Callable) -> tuple[np.ndarray, np.ndarray]:
    """train_all samples: per gold close (override 'slip' skipped, 'loose' -> empty) its 4 event crops (label = gold empty) and its
    4 open-approach crops (label 1); `cropperFor(take, cam)` returns a crop-at-time function"""
    tX, tY = [], []
    for tRow in goldRows:
        tOverride = overrides.get(tRow["id"])
        if tOverride == SLIP:
            continue
        if tOverride == LOOSE:
            tRow = dict(tRow, gold="empty")
        tCrop = cropperFor(tRow["take"], tRow["cam"])
        tClose, tOpen = int(tRow["t_close_ns"]), int(tRow["t_open_ns"])
        tX.append(np.stack([tCrop(tT) for tT in eventFrameTimes(tClose, tOpen)]))
        tY.append(int(tRow["gold"] == "empty"))
        tX.append(np.stack([tCrop(tT) for tT in approachFrameTimes(tClose)]))
        tY.append(1)
    return np.stack(tX), np.array(tY)


def trainAll(goldRows: list[dict], overrides: dict, cropperFor: Callable, device: str = DEVICE) -> torch.nn.Module:
    """the classifier trained on ALL gold (the frozen generator's model)"""
    tX, tY = trainingSet(goldRows, overrides, cropperFor)
    tModel = trainFold(tX, tY, device)
    print(f"CNN trained on {len(tY)} samples = v2 gold closes + one open-approach negative each ({sum(tY)} not-held)", flush=True)
    return tModel
