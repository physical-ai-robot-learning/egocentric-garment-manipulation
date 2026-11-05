"""Recorded tag-rig takes as the dataset stage sees them: the free demo windows of a take and the idle-trim table (dataset stage).

A take's meta.json holds t_start_monotonic_ns and the protocol; every protocol entry whose text starts with '자유:' is one demo
(one LeRobot episode). reports/idle_trim.csv (laundry_common.paths.IDLE_TRIM_CSV) trims each demo to its active window.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

FREE_DEMO_PREFIX = "자유:"           # protocol text of a free (task) demo
NS_PER_S = 1e9


@dataclass(frozen=True)
class Demo:
    """one free demo of a take: index among the free demos, window [mT0Ns, mT1Ns] (monotonic ns), ended early by the operator"""
    mIdx: int
    mT0Ns: int
    mT1Ns: int
    mEndedEarly: bool


def demoWindows(take: Path) -> list[Demo]:
    """the free demos of a take, in protocol order (was demos_of in tag_rig_to_lerobot / robotize_ik)"""
    tMeta = json.loads((Path(take) / "meta.json").read_text())
    tStart = int(tMeta["t_start_monotonic_ns"])
    tFree = [tP for tP in tMeta["protocol"] if tP["text"].startswith(FREE_DEMO_PREFIX)]
    return [Demo(tI, tStart + int(tP["t0"] * NS_PER_S), tStart + int((tP["t0"] + tP["dur"]) * NS_PER_S),
                 bool(tP.get("ended_early")))
            for tI, tP in enumerate(tFree)]


def demoKey(take: str, idx: int) -> str:
    """episode key take/demoNN used by every split file and table"""
    return f"{take}/demo{idx:02d}"


def loadIdleTrim(path: Path) -> dict:
    """idle_trim.csv -> {(take, demo): row} (row = the csv row: keep_start_s, keep_end_s, cut_head_s, cut_tail_s, ...)"""
    with open(path) as tFh:
        return {(tRow["take"], int(tRow["demo"])): tRow for tRow in csv.DictReader(tFh)}
