"""Grasp event segmentation v2 + gold remap (was grasp_v2_events.py).

Tighter jaw hysteresis on the motor g_cont (0 closed .. 1 open) for the GRASP pipeline only (umi_jaw.npz and the training jaw labels
are NOT touched). v1 events came from the umi_jaw.npz state (open > 0.55, closed < 0.35): 202 of 3554 closed runs hid a partial
re-open to ~0.48 for ~1 s. v2: open > 0.30, closed < 0.15, dwell 0.1 s (laundry_jaw.from_motor.hysteresis).

CLI steps: every v2 close event of the flatten demos (non-reject keys of the flat1103n_* / flat1104_* split docs) -> before/after
stats -> human gold (labeling rounds 1-3) remapped by CLOSE TIMESTAMP (|dt| <= 0.3 s) -> the event CNN re-trained on v2 crops, same
take-grouped 5 folds -> P/R/F1 next to v1 (laundry_grasp.cv.eventLevelCv; skip with --no-cv).
Writes OUT_DIR/report.txt, events_v2.csv, gold_v2.csv (+ cnn_cv_v2.npz).

    python -m laundry_grasp.events OUT_DIR [--split-dir DIR] [--v1-events CSV] [--rounds-dir DIR] [--two-close-ids TXT]
        [--v1-cnn-cv NPZ] [--no-cv]
"""
from __future__ import annotations

import argparse
import collections
import csv
import functools
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from laundry_common import paths as egopaths
from laundry_jaw.from_motor import hysteresis

from .demos import HAND_CAMS, camDir, demoWindow, parseDemoKey, splitDocKeys, videoName
from .gold import R2_DIR, TWO_CLOSE_IDS, roundGold

V2_OPEN_ABOVE = 0.30
V2_CLOSED_BELOW = 0.15
V2_DWELL_S = 0.1
REST_EDGE_S = 0.5                     # jaw already closed at the demo start / still closed at its end = resting, not a grasp
MATCH_S = 0.3                         # a gold close maps to the nearest v2 close within this time
NS_PER_S = 1e9
NS_PER_MS = 1e6
REPORT_DEMOS = 343                    # demo count written into the report (fixed number of the 11-05 run, see README)
EVENT_SPLIT_DOCS = ("flat1103n_*_S03G_split.json", "flat1104_*_S03G_split.json")
V1_EVENTS = "grasp_events_v1.csv"
V1_CNN_CV = "grasp_label_v1/cnn_cv.npz"   # under the rounds dir


def segmentJaw(tNs: np.ndarray, gCont: np.ndarray) -> tuple[np.ndarray, list[tuple[int, str]]]:
    """v2 jaw state (1 open / 0 closed) and the [(t_ns, 'open'|'close')] changes of a g_cont series"""
    return hysteresis(tNs, gCont, V2_OPEN_ABOVE, V2_CLOSED_BELOW, V2_DWELL_S)


@functools.lru_cache(maxsize=None)
def jawV2(take: str, cam: str) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str]]]:
    """(jaw t_ns, v2 state, v2 events) of one wrist camera of a take (cached per process; was grasp_v2_events.jaw_v2)"""
    tJaw = np.load(camDir(take, cam) / "umi_jaw.npz")
    tState, tEvents = segmentJaw(tJaw["t_ns"], tJaw["g_cont"])
    return tJaw["t_ns"], tState, tEvents


def closeIntervals(events: list[tuple[int, str]], endNs: int) -> list[tuple[int, int]]:
    """[(t_close, t_open)] per close event: the next open after the close, `endNs` when the jaw never opens again"""
    tOut = []
    for tI, (tClose, tKind) in enumerate(events):
        if tKind != "close":
            continue
        tOut.append((tClose, next((tT for tT, tK in events[tI + 1:] if tK == "open"), endNs)))
    return tOut


def overlapsWindow(tClose: int, tOpen: int, t0: int, t1: int) -> bool:
    return not (tOpen < t0 or tClose > t1)


def isRest(tClose: int, tOpen: int, t0: int, t1: int) -> bool:
    return tClose < t0 + REST_EDGE_S * NS_PER_S or tOpen > t1 - REST_EDGE_S * NS_PER_S


def eventKeys(splitDir: Path) -> list[str]:
    """the flatten demos of the event table: non-reject keys of the 11-03 / 11-04 split docs"""
    return sorted(splitDocKeys(splitDir, EVENT_SPLIT_DOCS, lambda tSplit: tSplit != "reject"))


def eventsV2(keys: Iterable[str]) -> list[dict]:
    """one row per v2 close event overlapping the demo window, per hand (ids numbered per demo and hand from 1)"""
    tRows = []
    for tKey in keys:
        tTake, tDemo = parseDemoKey(tKey)
        tT0, tT1 = demoWindow(tTake, tDemo)
        for tHand, tCam in HAND_CAMS:
            tJawT, _, tEvents = jawV2(tTake, tCam)
            tN = 0
            for tClose, tOpen in closeIntervals(tEvents, int(tJawT[-1])):
                if not overlapsWindow(tClose, tOpen, tT0, tT1):
                    continue
                tN += 1
                tRows.append(dict(id=f"{videoName(tTake, tDemo)}_{tHand}{tN}", demo=tKey, take=tTake, cam=tCam, hand=tHand,
                                  event=f"{tHand}{tN}", t_close_ns=tClose, t_open_ns=tOpen,
                                  closed_s=round((tOpen - tClose) / NS_PER_S, 2), rest=int(isRest(tClose, tOpen, tT0, tT1))))
    return tRows


class ReportLog:
    """print a line and append it to a report file"""

    def __init__(self, path: Path) -> None:
        self.mPath = Path(path)

    def __call__(self, line: str) -> None:
        print(line, flush=True)
        with open(self.mPath, "a") as tFh:
            tFh.write(line + "\n")


def readV1Events(path: Path) -> list[dict]:
    with open(path) as tFh:
        tRows = list(csv.DictReader(tFh))
    for tRow in tRows:
        tRow["t_close_ns"], tRow["t_open_ns"], tRow["rest"] = int(tRow["t_close_ns"]), int(tRow["t_open_ns"]), int(tRow["rest"])
    return tRows


def v1EventId(row: dict) -> str:
    tTake, tDemo = parseDemoKey(row["demo"])
    return f"{videoName(tTake, tDemo)}_{row['event']}"


def remapGold(gold: list[dict], v1Rows: list[dict], byDemoHand: dict, r2Labels: dict) -> tuple[list[dict], int]:
    """gold events (v1 ids) -> v2 event rows + gold columns, matched by close time; (remapped rows, unmapped count)"""
    tV1ById = {v1EventId(tRow): tRow for tRow in v1Rows}
    tOut, tMiss = [], 0
    for tGold in gold:
        tOld = tV1ById.get(tGold["id"])
        if tOld is None:
            tMiss += 1
            continue
        tCand = [tX for tX in byDemoHand[(tOld["demo"], tOld["hand"])]
                 if abs(tX["t_close_ns"] - tOld["t_close_ns"]) <= MATCH_S * NS_PER_S]
        if not tCand:
            tMiss += 1
            continue
        tBest = min(tCand, key=lambda tX: abs(tX["t_close_ns"] - tOld["t_close_ns"]))
        tOut.append(dict(tBest, gold=tGold["gold"], v1_id=tGold["id"], split=int(tOld["n_v2"] > 1),
                         random=int(r2Labels.get(tGold["id"]) == "unknown"),
                         dt_ms=round((tBest["t_close_ns"] - tOld["t_close_ns"]) / NS_PER_MS)))
    return tOut, tMiss


def writeCsv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as tFh:
        tWriter = csv.DictWriter(tFh, fieldnames=list(rows[0]))
        tWriter.writeheader()
        tWriter.writerows(rows)


def segmentationReport(new: list[dict], old: list[dict], log: ReportLog) -> dict:
    """v1 vs v2 event statistics; marks each v1 row with n_v2 (v2 closes inside it); returns {(demo, hand): v2 rows}"""
    tBy = collections.defaultdict(list)
    for tRow in new:
        tBy[(tRow["demo"], tRow["hand"])].append(tRow)
    tSplitCount = 0
    for tRow in old:
        tInside = [tX for tX in tBy[(tRow["demo"], tRow["hand"])]
                   if tRow["t_close_ns"] - MATCH_S * NS_PER_S <= tX["t_close_ns"] < tRow["t_open_ns"]]
        tRow["n_v2"] = len(tInside)
        tSplitCount += len(tInside) > 1
    tOn, tNn = [tR for tR in old if not tR["rest"]], [tR for tR in new if not tR["rest"]]
    log(f"=== segmentation v1 (open>0.55 / closed<0.35) vs v2 (open>{V2_OPEN_ABOVE} / closed<{V2_CLOSED_BELOW}), "
        f"{REPORT_DEMOS} demos ===")
    log(f"events total      v1 {len(old)}  v2 {len(new)}   non-rest v1 {len(tOn)}  v2 {len(tNn)}")
    log(f"events per demo   v1 {len(tOn) / REPORT_DEMOS:.2f}  v2 {len(tNn) / REPORT_DEMOS:.2f}  (non-rest)")
    log(f"v1 events split by a partial re-open: {tSplitCount}  (-> {sum(tR['n_v2'] for tR in old if tR['n_v2'] > 1)} v2 events); "
        f"v1 events with no v2 close: {sum(tR['n_v2'] == 0 for tR in old)}")
    tCd = np.array([tR["closed_s"] for tR in tNn])
    tCo = np.array([float(tR["closed_s"]) for tR in tOn])
    log(f"closed duration s median/p90   v1 {np.median(tCo):.2f}/{np.quantile(tCo, .9):.2f}   "
        f"v2 {np.median(tCd):.2f}/{np.quantile(tCd, .9):.2f}")
    return tBy


def goldReport(gold: list[dict], remapped: list[dict], miss: int, log: ReportLog) -> None:
    log(f"gold held/empty {len(gold)} -> remapped {len(remapped)} ({len(remapped) / len(gold):.1%}), unmapped {miss}; "
        f"of remapped, in a split v1 event {sum(tG['split'] for tG in remapped)}")
    log(f"close-time shift v1->v2 ms median {np.median([tG['dt_ms'] for tG in remapped]):.0f}  "
        f"p90 {np.quantile([abs(tG['dt_ms']) for tG in remapped], .9):.0f}")
    for tName, tSet in (("all", remapped), ("random", [tG for tG in remapped if tG["random"]])):
        tCount = collections.Counter(tG["gold"] for tG in tSet)
        log(f"gold {tName:6s}: held {tCount['held']}  empty {tCount['empty']}  "
            f"(empty {tCount['empty'] / max(len(tSet), 1):.1%})")


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tParser.add_argument("out_dir")
    tParser.add_argument("--split-dir", default=str(egopaths.GRASP_SPLITS), help="per-take split docs (EGO_GRASP_SPLITS)")
    tParser.add_argument("--v1-events", default=str(egopaths.GRASP_LABELS_V1 / V1_EVENTS), help="v1 event table")
    tParser.add_argument("--rounds-dir", default=str(egopaths.GRASP_ROUNDS), help="labeling round folders (EGO_GRASP_ROUNDS)")
    tParser.add_argument("--two-close-ids", default=str(TWO_CLOSE_IDS), help="round-1 clips that showed two closes")
    tParser.add_argument("--v1-cnn-cv", default=None, help=f"v1 event CNN CV (default <rounds-dir>/{V1_CNN_CV})")
    tParser.add_argument("--no-cv", action="store_true", help="stop after the event / gold tables")
    return tParser


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    tOut = Path(tArgs.out_dir).expanduser()
    tOut.mkdir(parents=True, exist_ok=True)
    (tOut / "report.txt").write_text("")
    tLog = ReportLog(tOut / "report.txt")
    tRounds = Path(tArgs.rounds_dir)
    tNew = eventsV2(eventKeys(Path(tArgs.split_dir)))
    tOld = readV1Events(Path(tArgs.v1_events))
    tBy = segmentationReport(tNew, tOld, tLog)
    tGold = roundGold(tRounds, Path(tArgs.two_close_ids))
    with open(tRounds / R2_DIR / "events.csv") as tFh:
        tR2Labels = {f"{tR['video'].rsplit('_', 1)[0]}_{tR['event']}": tR["label"] for tR in csv.DictReader(tFh)}
    tRemapped, tMiss = remapGold(tGold, tOld, tBy, tR2Labels)
    goldReport(tGold, tRemapped, tMiss, tLog)
    writeCsv(tOut / "events_v2.csv", tNew)
    writeCsv(tOut / "gold_v2.csv", tRemapped)
    if not tArgs.no_cv:
        from .cv import eventLevelCv
        eventLevelCv(tRemapped, tOut, Path(tArgs.v1_cnn_cv) if tArgs.v1_cnn_cv else tRounds / V1_CNN_CV, tLog)
    return 0


if __name__ == "__main__":
    sys.exit(main())
