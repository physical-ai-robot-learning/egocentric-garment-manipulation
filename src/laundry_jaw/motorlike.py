"""Motor-like jaw labels for camera-only takes (2026-10-30; was scripts/pose/jaw_motorlike.py).

Pipeline stage: jaw state (stage 2), relabelling of demos recorded before the 10-29 jaw servo. User: "281개 데이터가 너무 그리퍼가
안좋아" -> relabel GOLD-281's gripper. The wrist-camera jaw classifier (umi_jaw v003) that labelled GOLD-281 agrees with the Feetech
servo only ~70-80 % per frame on the motor takes: event timing is fine (median lag < 0.06 s) but ~40 % of its open/close events are
not in the servo trace. Here a per-hand ResNet18 is trained on the SAME wrist crops (umi_jaw ROI, 224x112 grey) with the servo as the
target, so it can relabel the camera-only takes. Two heads (10-30 review): [0] open logit (BCE on the servo open/closed state, same
hysteresis as the evaluation) and [1] motor-like aperture (MSE of sigmoid on the servo g_cont normalised per hand -- right servo
plateaus at ~0.65 open: stiff linkage).

  build  <out_dir>                      -> <out_dir>/<hand>.npz  (crops, target, take id) from every motor take, every FRAME_STRIDE-th frame
  train  <out_dir> --hand H --val TAKE  -> <out_dir>/jaw_motorlike_<hand>.pt (TorchScript, same I/O as umi_jaw models)
  eval   <out_dir> --hand H TAKE...     -> per take: frame F1 / agreement, event recall / precision / F1, lag, new model AND v003
  train --val-date 20261029            -> cross-date test: hold out every take of that date (domain-shortcut check)
  apply  <model_dir> TAKE...            -> <take>/<wrist dir>/umi_jaw_motorlike.npz (same keys as umi_jaw.npz); umi_jaw.npz untouched
Motor takes are read from EGO_DATA_ROOT/datasets/tag_rig.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from laundry_common.paths import TAG_RIG
from laundry_shared.visual_teleop.tools import umi_jaw

from .filters import MIN_SPAN_S, NS_PER_S, medianWindowLen, slidingMedian

WRIST_DIRS = {"right": "wrist", "left": "wrist_left"}
# Takes recorded with the servo: 10-29 and every 10-3x date.
MOTOR_TAKE_GLOBS = ("2026102[9]_*", "2026103*_*")
MOTOR_MODEL_MARK = "motor"
# servo g_cont when fully open (10-30 histograms: left 0.8-1.0, right 0.6-0.7)
OPEN_PLATEAU = {"left": 1.0, "right": 0.65}
# Hysteresis on the normalised servo AND on the model output (same rule both sides).
HYST_HI = 0.55
HYST_LO = 0.35
DWELL_S = 0.2
# The v003 classifier's own event hysteresis (umi_jaw infer).
V003_HI = 0.6
V003_LO = 0.4
# 10-31 (user review: right hand wrong on GOLD 10-26 pm = black cloth over black jaws): a frame is UNKNOWN when the model is unsure
# (p in (UNSURE_LO, UNSURE_HI)) for most (UNSURE_FRAC) of a +-UNSURE_WIN_S window, or the crop is mostly (DARK_FRAC) dark (grey
# below DARK_GREY_MAX: cloth over the lens / jaws invisible). UNKNOWN -> `valid` 0 in umi_jaw_motorlike.npz (the dataset builder
# masks the BCE jaw loss there); `state` keeps the hysteresis hold.
UNSURE_LO = 0.2
UNSURE_HI = 0.8
UNSURE_WIN_S = 0.25
UNSURE_FRAC = 0.5
DARK_GREY_MAX = 40
DARK_FRAC = 0.6
FRAME_STRIDE = 3
# g_cont of the applied label: model open probability median-filtered over this long a window.
MEDIAN_WINDOW_S = 0.2
# A 0/1 flag or state (or its interpolation between frames) counts as 1 above this.
BINARY_HALF = 0.5
# training
RESNET_FEATURES = 512
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
BATCH = 64
EVAL_BATCH = 128
MIN_POS_SHARE = 1e-3            # guards the class weight when no open frame is in the training split
INPUT_SHAPE = (1, 3, 224, 224)
SEED = 0
# inference
PREDICT_BATCH = 64
PREDICT_THREADS = 6
# event matching: a model event counts when the same kind is within this many seconds of a servo event
MATCH_TOL_S = 1.0
F1_EPS = 1e-9
LAG_PERCENTILE = 90
DEFAULT_EPOCHS = 3


def motorTakes() -> list[Path]:
    """Takes under TAG_RIG whose left-hand umi_jaw.npz came from the servo."""
    tOut = []
    for tTake in sorted(TAG_RIG.glob(MOTOR_TAKE_GLOBS[0])) + sorted(TAG_RIG.glob(MOTOR_TAKE_GLOBS[1])):
        tJaw = tTake / "wrist_left" / "umi_jaw.npz"
        if tJaw.exists() and MOTOR_MODEL_MARK in str(np.load(tJaw, allow_pickle=True)["model"]):
            tOut.append(tTake)
    return tOut


def loadCrops(wristDir: Path, hand: str, count: int) -> np.ndarray:
    """The recorder's crops.npz (first `count`), else re-crop the saved frames with the same ROI."""
    try:
        tCrops = np.load(wristDir / "crops.npz")["crops"]
        if len(tCrops) >= count:
            return tCrops[:count]
    except Exception:  # noqa: BLE001  crops.npz missing / truncated / unreadable in any way -> re-crop
        pass
    import cv2
    return np.stack([umi_jaw.crop(cv2.imread(str(wristDir / f"wrist_{tK:05d}.jpg")), umi_jaw.ROIS[hand]) for tK in range(count)])


def servoTarget(wristDir: Path, hand: str, tNs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(servo aperture at `tNs` normalised by the hand's open plateau, clipped to [0, 1]; servo valid flag at `tNs`)."""
    tJaw = np.load(wristDir / "umi_jaw.npz", allow_pickle=True)
    tServoT = tJaw["t_ns"].astype(np.float64)
    tG = np.interp(tNs.astype(np.float64), tServoT, tJaw["g_cont"].astype(np.float64)) / OPEN_PLATEAU[hand]
    if "valid" in tJaw.files:
        tValid = np.interp(tNs.astype(np.float64), tServoT, tJaw["valid"].astype(np.float64)) > BINARY_HALF
    else:
        tValid = np.ones(len(tNs), bool)
    return np.clip(tG, 0, 1).astype(np.float32), tValid


def build(outDir: Path) -> None:
    """outDir/<hand>.npz: crops, servo target, servo state and take name of every FRAME_STRIDE-th valid frame of every motor take."""
    outDir.mkdir(parents=True, exist_ok=True)
    for tHand, tDirName in WRIST_DIRS.items():
        tCrops, tTargets, tStates, tTakeNames = [], [], [], []
        for tTake in motorTakes():
            tWrist = tTake / tDirName
            if not (tWrist / "t_ns.npy").exists():
                continue
            tT = np.load(tWrist / "t_ns.npy")
            tAll = loadCrops(tWrist, tHand, len(tT))
            tN = min(len(tT), len(tAll))
            tTarget, tValid = servoTarget(tWrist, tHand, tT[:tN])
            tState, _ = umi_jaw.events(tT[:tN], tTarget, HYST_HI, HYST_LO, DWELL_S)
            tIdx = np.arange(0, tN, FRAME_STRIDE)
            tIdx = tIdx[tValid[tIdx]]
            tCrops.append(tAll[tIdx])
            tTargets.append(tTarget[tIdx])
            tStates.append(tState[tIdx].astype(np.float32))
            tTakeNames += [tTake.name] * len(tIdx)
            print(f"{tHand} {tTake.name}: {len(tIdx)} frames, open share {float(tState[tIdx].mean()):.2f}", flush=True)
        np.savez(outDir / f"{tHand}.npz", crops=np.concatenate(tCrops), target=np.concatenate(tTargets), state=np.concatenate(tStates),
                 take=np.array(tTakeNames))
        print(f"== {tHand}: {sum(map(len, tTargets))} frames", flush=True)


def _device(torch) -> str:
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def train(outDir: Path, hand: str, val: list[str], valDate: str, epochs: int) -> Path:
    """Fine-tune the v003 classifier of `hand` into the two-head motor-like model; returns the TorchScript file written.
    Held out: every take of `valDate` when given, else the takes in `val`; with neither, the model `apply` uses."""
    import torch
    import torchvision
    tData = np.load(outDir / f"{hand}.npz")
    tX, tY, tS, tK = tData["crops"], tData["target"], tData["state"], tData["take"]
    tIsVal = np.array([tName.startswith(valDate) for tName in tK]) if valDate else np.isin(tK, val)
    tTrain = np.where(~tIsVal)[0]
    tVal = np.where(tIsVal)[0]
    tDevice = _device(torch)
    tRng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    tNet = torchvision.models.resnet18()
    tNet.fc = torch.nn.Linear(RESNET_FEATURES, 1)
    tNet.load_state_dict(torch.jit.load(str(umi_jaw.MODELS[hand])).state_dict())     # start from v003 (same crops, same architecture)
    tHead = torch.nn.Linear(RESNET_FEATURES, 2)                                       # [0] open logit (v003 head copied), [1] aperture
    with torch.no_grad():
        tHead.weight[0] = tNet.fc.weight[0]
        tHead.bias[0] = tNet.fc.bias[0]
    tNet.fc = tHead
    tNet.to(tDevice)
    tOptimizer = torch.optim.AdamW(tNet.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    tPosShare = float(tS[tTrain].mean())
    tPosWeight = (1 - tPosShare) / max(tPosShare, MIN_POS_SHARE)                      # open frames are 10-30 % -> class-balanced BCE
    print(f"{hand}: train {len(tTrain)} val {len(tVal)} frames ({valDate or ','.join(val)} held out), open share {tPosShare:.2f} "
          f"(pos weight {tPosWeight:.1f}), {tDevice}", flush=True)
    tValIdx = tVal[::2]
    tXVal = torch.tensor(np.stack([umi_jaw.to_input(tX[tI]) for tI in tValIdx])) if len(tVal) else None
    for tEpoch in range(epochs):
        tNet.train()
        tRng.shuffle(tTrain)
        tTotal = 0.0
        for tB in range(0, len(tTrain), BATCH):
            tIdx = tTrain[tB:tB + BATCH]
            tInput = torch.tensor(np.stack([umi_jaw.to_input(umi_jaw.augment(tX[tI], tRng)) for tI in tIdx])).to(tDevice)
            tState = torch.tensor(tS[tIdx]).to(tDevice)
            tTarget = torch.tensor(tY[tIdx]).to(tDevice)
            tOut = tNet(tInput)
            tBce = (torch.nn.functional.binary_cross_entropy_with_logits(tOut[:, 0], tState, reduction="none")
                    * torch.where(tState > BINARY_HALF, tPosWeight, 1.0)).mean()
            tMse = torch.nn.functional.mse_loss(torch.sigmoid(tOut[:, 1]), tTarget)
            tLoss = tBce + tMse
            tOptimizer.zero_grad()
            tLoss.backward()
            tOptimizer.step()
            tTotal += float(tLoss) * len(tIdx)
        tMessage = f"epoch {tEpoch + 1}: loss {tTotal / len(tTrain):.4f}"
        if tXVal is not None:
            tNet.eval()
            with torch.no_grad():
                tOut = torch.cat([tNet(tXVal[tI:tI + EVAL_BATCH].to(tDevice)).cpu() for tI in range(0, len(tXVal), EVAL_BATCH)]).numpy()
            tTrue = tS[tValIdx] > BINARY_HALF
            tPred = tOut[:, 0] > 0
            tTp = float((tPred & tTrue).sum())
            tF1 = 2 * tTp / max(float(tPred.sum() + tTrue.sum()), 1)
            tMae = float(np.abs(1 / (1 + np.exp(-tOut[:, 1])) - tY[tValIdx]).mean())
            tMessage += f" | held-out frame acc {float((tPred == tTrue).mean()):.3f} open-F1 {tF1:.3f} aperture MAE {tMae:.3f}"
        print(tMessage, flush=True)
    tNet.eval().cpu()
    tTag = f"_val{valDate}" if valDate else f"_val{'-'.join(val)}" if val else ""   # no holdout = the model `apply` uses
    tModelFile = outDir / f"jaw_motorlike_{hand}{tTag}.pt"
    torch.jit.trace(tNet, torch.zeros(*INPUT_SHAPE)).save(str(tModelFile))
    print("saved", tModelFile)
    return tModelFile


def predict(model: Path, crops: np.ndarray, both: bool = False):
    """sigmoid of head 0 (open probability); both=True also returns head 1 (motor-like aperture). v003 has only head 0."""
    import torch
    tNet = torch.jit.load(str(model)).eval()
    torch.set_num_threads(PREDICT_THREADS)
    with torch.no_grad():
        tOut = torch.sigmoid(torch.cat([tNet(torch.tensor(np.stack([umi_jaw.to_input(tCrop) for tCrop in crops[tI:tI + PREDICT_BATCH]])))
                                        for tI in range(0, len(crops), PREDICT_BATCH)])).numpy()
    if both:
        return tOut[:, 0], tOut[:, 1] if tOut.shape[1] > 1 else tOut[:, 0]
    return tOut[:, 0]


def matchEvents(reference: list, events: list, tol: float = MATCH_TOL_S) -> tuple[float, float, list[float]]:
    """(recall, precision, lags s) of `events` against `reference`: greedy, each event used once, same kind within `tol` s."""
    tUsed = set()
    tHits = 0
    tLags = []
    for tRefT, tKind in reference:
        tCandidates = [(abs(tEvT - tRefT), tI, tEvT) for tI, (tEvT, tEvKind) in enumerate(events)
                       if tEvKind == tKind and tI not in tUsed and abs(tEvT - tRefT) / NS_PER_S <= tol]
        if tCandidates:
            _, tI, tEvT = min(tCandidates)
            tUsed.add(tI)
            tHits += 1
            tLags.append((tEvT - tRefT) / NS_PER_S)
    return tHits / max(len(reference), 1), tHits / max(len(events), 1), tLags


def evaluate(outDir: Path, hand: str, tag: str, takes: list[str]) -> list[dict]:
    """Per take: the motor-like model (outDir/jaw_motorlike_<hand><tag>.pt) and v003 against the servo; writes eval_<hand><tag>.json."""
    tResults = []
    for tTake in takes:
        tWrist = TAG_RIG / tTake / WRIST_DIRS[hand]
        tT = np.load(tWrist / "t_ns.npy")
        tCrops = loadCrops(tWrist, hand, len(tT))
        tN = min(len(tT), len(tCrops))
        tT = tT[:tN]
        tTarget, tValid = servoTarget(tWrist, hand, tT)
        tStateRef, tEventsRef = umi_jaw.events(tT, tTarget, HYST_HI, HYST_LO, DWELL_S)
        for tName, tModel in (("motorlike", outDir / f"jaw_motorlike_{hand}{tag}.pt"), ("v003", umi_jaw.MODELS[hand])):
            tP = predict(tModel, tCrops[:tN])
            tHi, tLo = (HYST_HI, HYST_LO) if tName == "motorlike" else (V003_HI, V003_LO)
            tState, tEvents = umi_jaw.events(tT, tP, tHi, tLo, DWELL_S)
            tRecall, tPrecision, tLags = matchEvents(tEventsRef, tEvents)
            tPred, tRef = tState[tValid] > 0, tStateRef[tValid] > 0
            tFrameF1 = 2 * float((tPred & tRef).sum()) / max(float(tPred.sum() + tRef.sum()), 1)
            tRow = dict(take=tTake, hand=hand, model=tName, frame_agree=float((tPred == tRef).mean()), frame_F1=tFrameF1, recall=tRecall,
                        precision=tPrecision, event_F1=2 * tRecall * tPrecision / max(tRecall + tPrecision, F1_EPS), n_ref=len(tEventsRef),
                        n_ev=len(tEvents), lag_med=float(np.median(tLags)) if tLags else None,
                        lag_abs_p90=float(np.percentile(np.abs(tLags), LAG_PERCENTILE)) if tLags else None)
            tResults.append(tRow)
            print({tKey: round(tValue, 3) if isinstance(tValue, float) else tValue for tKey, tValue in tRow.items()}, flush=True)
    for tName in ("motorlike", "v003"):
        tRows = [tRow for tRow in tResults if tRow["model"] == tName]
        tNRef = sum(tRow["n_ref"] for tRow in tRows)
        tNEv = sum(tRow["n_ev"] for tRow in tRows)
        tHits = sum(tRow["recall"] * tRow["n_ref"] for tRow in tRows)
        tRecall, tPrecision = tHits / max(tNRef, 1), tHits / max(tNEv, 1)
        print(f"== {hand} {tName}{tag}: event recall {tRecall:.3f} precision {tPrecision:.3f} "
              f"F1 {2 * tRecall * tPrecision / max(tRecall + tPrecision, F1_EPS):.3f} | "
              f"frame F1 {np.mean([tRow['frame_F1'] for tRow in tRows]):.3f} agree {np.mean([tRow['frame_agree'] for tRow in tRows]):.3f}",
              flush=True)
    with open(outDir / f"eval_{hand}{tag}.json", "w") as tHandle:
        json.dump(tResults, tHandle, indent=1)
    return tResults


def applyModel(modelDir: Path, takes: list[str]) -> None:
    """Write <take>/<wrist dir>/umi_jaw_motorlike.npz with modelDir/jaw_motorlike_<hand>.pt for every hand that has frames."""
    for tTakeArg in takes:
        tTake = Path(tTakeArg)
        for tHand, tDirName in WRIST_DIRS.items():
            tWrist = tTake / tDirName
            if not (tWrist / "t_ns.npy").exists():
                continue
            tT = np.load(tWrist / "t_ns.npy")
            tCrops = loadCrops(tWrist, tHand, len(tT))
            tN = min(len(tT), len(tCrops))
            tT = tT[:tN]
            tModel = modelDir / f"jaw_motorlike_{tHand}.pt"
            tP, tAperture = predict(tModel, tCrops[:tN], both=True)
            tState, tEvents = umi_jaw.events(tT, tP, HYST_HI, HYST_LO, DWELL_S)
            tG = slidingMedian(tP, medianWindowLen(tT, MEDIAN_WINDOW_S, tN), np.float32)
            tFps = (tN - 1) / max((tT[-1] - tT[0]) / NS_PER_S, MIN_SPAN_S)
            tHalfWindow = max(1, int(round(UNSURE_WIN_S * tFps)))
            tUnsure = ((tP > UNSURE_LO) & (tP < UNSURE_HI)).astype(float)
            tUnsure = np.convolve(tUnsure, np.ones(2 * tHalfWindow + 1) / (2 * tHalfWindow + 1), mode="same") > UNSURE_FRAC
            tDark = (tCrops[:tN] < DARK_GREY_MAX).reshape(tN, -1).mean(1) > DARK_FRAC
            tValid = (~tUnsure) & (~tDark)
            np.savez(tWrist / "umi_jaw_motorlike.npz", t_ns=tT, p_open=tP, g_cont=tG, aperture=tAperture.astype(np.float32), state=tState,
                     valid=tValid.astype(np.uint8), unsure=tUnsure, dark=tDark, hand=tHand, model=tModel.name,
                     ev_t=np.array([tEvent[0] for tEvent in tEvents], np.int64), ev_kind=np.array([tEvent[1] for tEvent in tEvents]))
            tOld = np.load(tWrist / "umi_jaw.npz", allow_pickle=True) if (tWrist / "umi_jaw.npz").exists() else None
            if tOld is not None:
                tOldState = np.interp(tT.astype(float), tOld["t_ns"].astype(float), tOld["state"].astype(float))[:tN] > BINARY_HALF
                tAgree = float((tOldState.astype(int) == tState).mean())
            else:
                tAgree = float("nan")
            print(f"{tTake.name} {tHand}: open share {tState.mean():.2f}, events {len(tEvents)} "
                  f"(old {len(tOld['ev_t']) if tOld is not None else '-'}), "
                  f"p in (0.2,0.8) {float(((tP > UNSURE_LO) & (tP < UNSURE_HI)).mean()):.2f}, UNKNOWN {1 - tValid.mean():.2f} "
                  f"(unsure {tUnsure.mean():.2f}, dark {tDark.mean():.2f}), agree with old label {tAgree:.2f}", flush=True)


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tSub = tParser.add_subparsers(dest="cmd", required=True)
    tBuild = tSub.add_parser("build")
    tBuild.add_argument("out")
    tTrain = tSub.add_parser("train")
    tTrain.add_argument("out")
    tTrain.add_argument("--hand", required=True)
    tTrain.add_argument("--val", nargs="*", default=[])
    tTrain.add_argument("--val-date", default="")
    tTrain.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    tEval = tSub.add_parser("eval")
    tEval.add_argument("out")
    tEval.add_argument("--hand", required=True)
    tEval.add_argument("--tag", default="")
    tEval.add_argument("takes", nargs="+")
    tApply = tSub.add_parser("apply")
    tApply.add_argument("out")
    tApply.add_argument("takes", nargs="+")
    return tParser


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    tOut = Path(tArgs.out)
    if tArgs.cmd == "build":
        build(tOut)
    elif tArgs.cmd == "train":
        train(tOut, tArgs.hand, tArgs.val, tArgs.val_date, tArgs.epochs)
    elif tArgs.cmd == "eval":
        evaluate(tOut, tArgs.hand, tArgs.tag, tArgs.takes)
    else:
        applyModel(tOut, tArgs.takes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
