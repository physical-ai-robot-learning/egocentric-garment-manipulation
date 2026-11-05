"""Take-grouped cross-validation of the grasp classifier (was grasp_ts_cv.py + the CV part of grasp_v2_events.py).

Folds: the takes of the gold events, shuffled with seed 1105, split into 5 groups; every event of a take is in the same fold.

Frame level (CLI, the conditioning signal; was grasp_ts_cv): per fold the trainAll recipe (v2 gold closes x 4 frames + open-approach
negatives; 'slip' excluded, 'loose' = not held, 'empty' override = empty) on the train takes; held-out events scored per frame at
15 Hz from close - 1 s to open + 0.5 s; p(t) = causal window mean (laundry_grasp.pgrasp) for W in WINDOWS. "judged" = jaw closed
and t >= close + 0.3 s. Metrics per gold class, event level: held median / false-drop (any judged tick < 0.2); empty median / tail
(event median > 0.9, any > 0.9); loose median; slip: p < 0.5 before the jaw opens + lead time; open approach (close - 1 s .. close)
median / share > 0.5. Writes OUT/ts_cv_raw.npy (per held-out event: ts, per-frame p, last change) and OUT/ts_cv_report.txt.

Event level (eventLevelCv, run by laundry_grasp.events): the event CNN on v2 crops, P(empty) per event -> cnn_cv_v2.npz and P/R/F1
next to the v1 crops' CV.

    python -m laundry_grasp.cv OUT_DIR [report] [--gold CSV] [--overrides JSON]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from laundry_common import paths as egopaths

from .crops import JawRoiCropper
from .events import jawV2
from .gold import GOLD_V2, NOT_HELD, OVERRIDES_V2, SLIP, eventClass, loadOverrides, readCsv
from .model import approachFrameTimes, eventFrameTimes, predict, trainFold
from .pgrasp import HZ, WARM_S, causalWindowMean, changeTimes, framePGrasp, lastChangeTimes, tickTimes

FOLDS = 5
FOLD_SEED = 1105
WINDOWS = (0.2, 0.3, 0.4, 0.6)
BEFORE_CLOSE_S = 1.0                  # scored span: close - 1 s ..
AFTER_OPEN_S = 0.5                    # .. open + 0.5 s
PREDICT_BATCH = 64
DROP_P = 0.2                          # a held event 'drops' when a judged tick is below this
TAIL_P = 0.9                          # an empty event is a 'tail' error above this
SLIP_P = 0.5
APPROACH_P = 0.5
BAND_RULE = (0.9, 0.2)                # event rule: p_grasp > 0.9 held, < 0.2 empty, else unknown
F1_BANDS = ((0.5, 0.5),)
NS_PER_S = 1e9
CLASSES = ("held", "empty", "loose", "slip")


def takeFolds(takes: np.ndarray) -> list[np.ndarray]:
    """the take groups of the 5 folds (unique takes, shuffled with FOLD_SEED)"""
    tTakes = np.unique(takes)
    np.random.default_rng(FOLD_SEED).shuffle(tTakes)
    return np.array_split(tTakes, FOLDS)


def bandF1(y: np.ndarray, p: np.ndarray, lo: float, hi: float) -> tuple[dict, float]:
    """P/R/F1 (2 decimals) of empty (p >= hi) and held (p < lo) on the decided events, and the decided share (was
    grasp_label_v1_temporal.f1s)"""
    tDecided = (p >= hi) | (p < lo)
    tEmpty = p >= hi
    tOut = {}
    for tName, tPred, tTrue in (("empty", tEmpty, y == 1), ("held", ~tEmpty, y == 0)):
        tTp = (tPred & tTrue & tDecided).sum()
        tPr = tTp / max((tPred & tDecided).sum(), 1)
        tRc = tTp / max((tTrue & tDecided).sum(), 1)
        tOut[tName] = (round(tPr, 2), round(tRc, 2), round(2 * tPr * tRc / max(tPr + tRc, 1e-9), 2))
    return tOut, round(tDecided.mean(), 2)


def foldTrainingSet(gold: list[dict], foldTakes, croppers: dict) -> tuple[np.ndarray, np.ndarray]:
    tX, tY = [], []
    for tRow in gold:
        if tRow["take"] in foldTakes or tRow["cls"] == SLIP:
            continue
        tCrop = croppers[(tRow["take"], tRow["cam"])]
        tClose, tOpen = int(tRow["t_close_ns"]), int(tRow["t_open_ns"])
        tX.append(np.stack([tCrop(tT) for tT in eventFrameTimes(tClose, tOpen)]))
        tY.append(int(tRow["cls"] in NOT_HELD))
        tX.append(np.stack([tCrop(tT) for tT in approachFrameTimes(tClose)]))
        tY.append(1)
    return np.stack(tX), np.array(tY)


def scoreEvent(model, row: dict, crop) -> dict:
    """per-frame P(grasp) of one held-out event from close - 1 s to open + 0.5 s, with the jaw change before each tick"""
    tClose, tOpen = int(row["t_close_ns"]), int(row["t_open_ns"])
    tTs = tickTimes(tClose - BEFORE_CLOSE_S * NS_PER_S, tOpen + AFTER_OPEN_S * NS_PER_S, HZ)
    tFrame = framePGrasp(model, np.stack([crop(tT) for tT in tTs]), PREDICT_BATCH)
    _, _, tEvents = jawV2(row["take"], row["cam"])
    return dict(id=row["id"], cls=row["cls"], te=tClose, tn=tOpen, ts=tTs, fp=tFrame, last=lastChangeTimes(tTs, changeTimes(tEvents)))


def frameLevelCv(gold: list[dict], out: Path) -> list[dict]:
    tCroppers = {}
    for tRow in gold:
        tKey = (tRow["take"], tRow["cam"])
        if tKey not in tCroppers:
            tCroppers[tKey] = JawRoiCropper.forCamera(*tKey)
    tRes = []
    for tFold, tFoldTakes in enumerate(takeFolds(np.array([tRow["take"] for tRow in gold]))):
        tX, tY = foldTrainingSet(gold, tFoldTakes, tCroppers)
        tModel = trainFold(tX, tY)
        print(f"fold {tFold}: trained on {len(tY)}", flush=True)
        for tRow in gold:
            if tRow["take"] in tFoldTakes:
                tRes.append(scoreEvent(tModel, tRow, tCroppers[(tRow["take"], tRow["cam"])]))
    np.save(out / "ts_cv_raw.npy", np.array(tRes, dtype=object), allow_pickle=True)
    return tRes


def windowLine(res: list[dict], windowS: float) -> str:
    tS = {tC: [] for tC in CLASSES}
    tApproach = []
    for tR in res:
        tTs = tR["ts"]
        tP = causalWindowMean(tTs, tR["fp"], tR["last"], windowS)
        tJudged = (tTs >= tR["te"] + WARM_S * NS_PER_S) & (tTs < tR["tn"])
        tAp = (tTs >= tR["te"] - BEFORE_CLOSE_S * NS_PER_S) & (tTs < tR["te"])
        if tJudged.any():
            tS[tR["cls"]].append((tP[tJudged], tTs[tJudged], tR["tn"]))
        if tAp.any():
            tApproach.append(tP[tAp])

    def tMedian(cls: str) -> float:
        return np.median([np.median(tX[0]) for tX in tS[cls]]) if tS[cls] else float("nan")

    tHeld, tEmpty, tSlip = tS["held"], tS["empty"], tS["slip"]
    tFalseDrop = np.mean([tX[0].min() < DROP_P for tX in tHeld])
    tDropTicks = np.mean(np.concatenate([tX[0] < DROP_P for tX in tHeld]))
    tTail = np.mean([np.median(tX[0]) > TAIL_P for tX in tEmpty])
    tTailAny = np.mean([tX[0].max() > TAIL_P for tX in tEmpty])
    tDropped = [tX for tX in tSlip if (tX[0] < SLIP_P).any()]
    tLead = [(tX[2] - tX[1][np.argmax(tX[0] < SLIP_P)]) / NS_PER_S for tX in tDropped]
    tSlipHigh = np.mean([tX[0].max() > TAIL_P for tX in tSlip]) if tSlip else float("nan")
    tAp = np.concatenate(tApproach)
    return (f"W={windowS:.1f}s | held n{len(tHeld)} median {tMedian('held'):.2f} false-drop(any<0.2) {tFalseDrop:.0%} "
            f"ticks<0.2 {tDropTicks:.1%} | empty n{len(tEmpty)} median {tMedian('empty'):.2f} "
            f"tail(median>0.9) {tTail:.0%} any>0.9 {tTailAny:.0%} | loose n{len(tS['loose'])} median {tMedian('loose'):.2f} | "
            f"slip n{len(tSlip)} reached>0.9 {tSlipHigh:.0%} "
            f"dropped<0.5 before open {len(tDropped)}/{len(tSlip)} lead median {np.median(tLead) if tLead else float('nan'):.2f}s | "
            f"open approach median {np.median(tAp):.2f} >0.5 {np.mean(tAp > APPROACH_P):.0%}")


def report(res: list[dict], out: Path) -> str:
    tText = "\n".join(windowLine(res, tW) for tW in WINDOWS)
    print(tText, flush=True)
    (out / "ts_cv_report.txt").write_text(tText + "\n")
    return tText


def eventLevelCv(gold2: list[dict], out: Path, v1CnnCv: Path, log) -> None:
    """event CNN on the v2 crops of the remapped gold, take-grouped 5 folds -> out/cnn_cv_v2.npz + P/R/F1 vs the v1 crops' CV"""
    tCroppers, tX = {}, []
    for tRow in gold2:
        tKey = (tRow["take"], tRow["cam"])
        if tKey not in tCroppers:
            tCroppers[tKey] = JawRoiCropper.forCamera(*tKey)
        tX.append(np.stack([tCroppers[tKey](tT) for tT in eventFrameTimes(tRow["t_close_ns"], tRow["t_open_ns"])]))
    tX = np.stack(tX)
    tY = np.array([tRow["gold"] == "empty" for tRow in gold2], int)
    tTakes = np.array([tRow["take"] for tRow in gold2])
    tP = np.zeros(len(tY))
    for tFoldTakes in takeFolds(tTakes):
        tTest = np.isin(tTakes, tFoldTakes)
        tModel = trainFold(tX[~tTest], tY[~tTest])
        tP[tTest] = predict(tModel, tX[tTest])
    np.savez(out / "cnn_cv_v2.npz", p=tP, y=tY, take=tTakes, ids=np.array([tRow["id"] for tRow in gold2]),
             v1_ids=np.array([tRow["v1_id"] for tRow in gold2]), random=np.array([tRow["random"] for tRow in gold2]),
             split=np.array([tRow["split"] for tRow in gold2]))
    tV1 = np.load(v1CnnCv, allow_pickle=True)
    tV1P = dict(zip(tV1["ids"].tolist(), tV1["p"].tolist()))
    tQ1 = np.array([tV1P[tRow["v1_id"]] for tRow in gold2])
    tRandom = np.array([tRow["random"] for tRow in gold2], bool)
    tHeldAbove, tEmptyBelow = BAND_RULE
    log("=== CNN take-grouped 5-fold CV, same remapped gold events (p = P(empty)) ===")
    for tName, tSel in (("all gold", np.ones(len(tY), bool)), ("random subset", tRandom)):
        for tTag, tQ in (("v1 crops", tQ1), ("v2 crops", tP)):
            (tLo, tHi), = F1_BANDS
            tScores, _ = bandF1(tY[tSel], tQ[tSel], tLo, tHi)
            tPg = 1 - tQ[tSel]
            tHeld, tEmpty = tPg > tHeldAbove, tPg < tEmptyBelow
            tYs = tY[tSel]
            log(f"{tName:13s} {tTag}: @0.5 held P/R/F1 {tScores['held']}  empty P/R/F1 {tScores['empty']}  | rule 0.9/0.2: "
                f"held {tHeld.sum()} prec {1 - (tHeld & (tYs == 1)).sum() / max(tHeld.sum(), 1):.3f}"
                f"  empty {tEmpty.sum()} prec {1 - (tEmpty & (tYs == 0)).sum() / max(tEmpty.sum(), 1):.3f}  "
                f"empty-recall {(tEmpty & (tYs == 1)).sum() / max((tYs == 1).sum(), 1):.2f}  unknown {(~tHeld & ~tEmpty).mean():.0%}")


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tParser.add_argument("out_dir")
    tParser.add_argument("mode", nargs="?", default="", choices=["", "report"], help="report = re-report OUT/ts_cv_raw.npy only")
    tParser.add_argument("--gold", default=str(egopaths.GRASP_LABELS_V2 / GOLD_V2))
    tParser.add_argument("--overrides", default=str(egopaths.GRASP_LABELS_V2 / OVERRIDES_V2))
    return tParser


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    tOut = Path(tArgs.out_dir).expanduser()
    tOut.mkdir(parents=True, exist_ok=True)
    if tArgs.mode == "report":
        report(list(np.load(tOut / "ts_cv_raw.npy", allow_pickle=True)), tOut)
        return 0
    tOverrides = loadOverrides(Path(tArgs.overrides))
    tGold = [dict(tRow, cls=eventClass(tRow, tOverrides)) for tRow in readCsv(Path(tArgs.gold))]
    report(frameLevelCv(tGold, tOut), tOut)
    return 0


if __name__ == "__main__":
    sys.exit(main())
