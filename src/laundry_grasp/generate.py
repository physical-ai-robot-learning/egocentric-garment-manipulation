"""FROZEN p_grasp generator "grasp-v2-W0.4" of the FLAT250-PGRASP run (was grasp_pgrasp_gen.py).

1. classifier: the jaw-ROI ResNet-18 (laundry_grasp.model.trainAll: v2 gold closes x 4 frames + open-approach negatives, 'slip'
   excluded, 'loose' = not held; user overrides gold_overrides_v2.json) trained on ALL gold -> OUT/model.pt + model_meta.json.
   An existing OUT/model.pt is loaded instead (the frozen model is never retrained by a re-run).
2. per demo (take/demoNN of the 11-03 + 11-04 (+ 11-05) S03G train / val / test sets): causal p_grasp(t) at 15 Hz over the demo
   window (+ 1 s margins), W = 0.4 s window restarted at each v2 jaw state change (open > 0.30 / closed < 0.15), scored on open AND
   closed frames -> OUT/demos/<take>_demoNN.npz (version, t_ns, L_p, L_closed, R_p, R_closed; L = wrist_left cam, R = wrist cam).
   Existing demo files are skipped.
NOTE: gold events are in the classifier's training set, so p on those events is in-sample (slightly overconfident).
The p values depend on the OpenCV build (JPEG decode / resize): the 11-05 files were written in the env with opencv 5.0 + torch 2.10;
opencv 4.13 gives per-tick differences up to ~0.1.

Demo keys: <lerobot-root>/flat1103_1104_S03G_{train,val,test}/meta/split.json; when they list fewer than 300 keys (the aggregated
train set lives on the cluster only) also the train / val / test keys of the per-take split docs flat1103n_* / flat1104_* /
flat1105_* in --split-dir.

    python -m laundry_grasp.generate OUT_DIR [--lerobot-root DIR] [--split-dir DIR] [--gold CSV] [--overrides JSON]
        [--demo take/demoNN ...] [--workers 6]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from laundry_common import paths as egopaths

from .crops import JawRoiCropper
from .demos import HAND_CAMS, demoWindow, parseDemoKey, splitDocKeys
from .events import jawV2
from .gold import GOLD_V2, OVERRIDES_V2, loadOverrides, readCsv
from .model import DEVICE, loadModel, trainAll
from .pgrasp import HZ, VERSION, WINDOW_S, causalWindowMean, changeTimes, closedAt, cropsAt, framePGrasp, lastChangeTimes, tickTimes

MARGIN_S = 1.0                        # p_grasp series extend this far before / after the demo window
PREDICT_BATCH = 128
CROP_WORKERS = 6
PROGRESS_EVERY = 10
NS_PER_S = 1e9
LEROBOT_SET = "flat1103_1104_S03G_{split}"
SPLITS = ("train", "val", "test")
MIN_SET_KEYS = 300                    # fewer keys in the local LeRobot split files -> add the per-take split docs
GEN_SPLIT_DOCS = ("flat1103n_*_S03G_split.json", "flat1104_*_S03G_split.json", "flat1105_*_S03G_split.json")
MODEL_META = dict(                     # model_meta.json after version / created
    arch="torchvision resnet18, fc -> 1 logit = P(not held)",
    input="jaw-ROI 256 px crop (tip from grasp_label_v0.jaw_mask_and_tip), top at tip_y - 153 px, RGB 224, ImageNet norm",
    train="data/grasp_labels_v2/gold_v2.csv + gold_overrides_v2.json (slip excluded, loose = not held) + open-approach negatives; EPOCHS 8",
    rule=f"p_grasp = 1 - P(not held); causal mean over [max(last v2 jaw change, t - {WINDOW_S}), t] at {HZ:.0f} Hz; "
         "v2 jaw hysteresis open>0.30 closed<0.15",
    human_reference="data/grasp_labels_v2/ts_cv/human_reference_W0.4.json")


def demoKeys(lerobotRoot: Path, splitDir: Path) -> list[str]:
    tKeys: list[str] = []
    for tSplit in SPLITS:
        tFile = lerobotRoot / LEROBOT_SET.format(split=tSplit) / "meta/split.json"
        if tFile.exists():
            tKeys += json.loads(tFile.read_text())["episodes"][tSplit]
    if len(tKeys) < MIN_SET_KEYS:
        tKeys += sorted(splitDocKeys(splitDir, GEN_SPLIT_DOCS, lambda tSplit: tSplit in SPLITS))
    return sorted(set(tKeys))


def demoSeries(model, take: str, demo: int, croppers: dict, pool, device: str = DEVICE) -> dict:
    """the generator output of one demo: t_ns + per hand <H>_p (float32) and <H>_closed (int8)"""
    tT0, tT1 = demoWindow(take, demo)
    tTs = tickTimes(tT0 - MARGIN_S * NS_PER_S, tT1 + MARGIN_S * NS_PER_S, HZ)
    tOut = dict(t_ns=tTs)
    for tHand, tCam in HAND_CAMS:
        if (take, tCam) not in croppers:
            croppers[(take, tCam)] = JawRoiCropper.forCamera(take, tCam)
        tJawT, tState, tEvents = jawV2(take, tCam)
        tFrame = framePGrasp(model, cropsAt(croppers[(take, tCam)], tTs, pool), PREDICT_BATCH, device)
        tLast = lastChangeTimes(tTs, changeTimes(tEvents))
        tOut[f"{tHand}_p"] = np.array(causalWindowMean(tTs, tFrame, tLast, WINDOW_S), np.float32)
        tOut[f"{tHand}_closed"] = closedAt(tJawT, tState, tTs).astype(np.int8)
    return tOut


def frozenModel(out: Path, goldCsv: Path, overridesJson: Path, device: str = DEVICE):
    """OUT/model.pt if present, else train on all gold and write model.pt + model_meta.json"""
    tPath = out / "model.pt"
    if tPath.exists():
        return loadModel(tPath, device)
    tCroppers: dict = {}

    def tCropperFor(take: str, cam: str) -> JawRoiCropper:
        if (take, cam) not in tCroppers:
            tCroppers[(take, cam)] = JawRoiCropper.forCamera(take, cam)
        return tCroppers[(take, cam)]

    tModel = trainAll(readCsv(goldCsv), loadOverrides(overridesJson), tCropperFor, device)
    torch.save(tModel.state_dict(), tPath)
    tMeta = dict(version=VERSION, created=time.strftime("%Y-%m-%d %H:%M"), **MODEL_META)
    (out / "model_meta.json").write_text(json.dumps(tMeta, indent=1))
    return tModel


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tParser.add_argument("out_dir")
    tParser.add_argument("--lerobot-root", default=str(egopaths.LEROBOT_ROOT))
    tParser.add_argument("--split-dir", default=str(egopaths.GRASP_SPLITS), help="per-take split docs (EGO_GRASP_SPLITS)")
    tParser.add_argument("--gold", default=str(egopaths.GRASP_LABELS_V2 / GOLD_V2))
    tParser.add_argument("--overrides", default=str(egopaths.GRASP_LABELS_V2 / OVERRIDES_V2))
    tParser.add_argument("--demo", action="append", default=None, help="only these take/demoNN keys (repeatable)")
    tParser.add_argument("--workers", type=int, default=CROP_WORKERS, help="crop decoding threads")
    return tParser


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    tOut = Path(tArgs.out_dir).expanduser()
    (tOut / "demos").mkdir(parents=True, exist_ok=True)
    tModel = frozenModel(tOut, Path(tArgs.gold), Path(tArgs.overrides))
    tKeys = demoKeys(Path(tArgs.lerobot_root), Path(tArgs.split_dir))
    if tArgs.demo:
        tKeys = [tK for tK in tKeys if tK in set(tArgs.demo)]
    print(f"{VERSION}: {len(tKeys)} demos", flush=True)
    tCroppers: dict = {}
    with ThreadPoolExecutor(tArgs.workers) as tPool:
        for tN, tKey in enumerate(tKeys):
            tTake, tDemo = parseDemoKey(tKey)
            tFile = tOut / "demos" / f"{tTake}_{tKey.split('/')[1]}.npz"
            if tFile.exists():
                continue
            tSeries = demoSeries(tModel, tTake, tDemo, tCroppers, tPool)
            np.savez(tFile, version=VERSION, **tSeries)
            if tN % PROGRESS_EVERY == 0:
                print(f"  {tN}/{len(tKeys)} {tKey}  L mean {tSeries['L_p'].mean():.2f}  R mean {tSeries['R_p'].mean():.2f}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
