"""Human gold labels of the grasp events (was grasp_label_v1_temporal.gold + the override handling of grasp_render_prob / grasp_ts_cv).

Rounds (labeling page laundry_grasp.gold_ui, one folder each under EGO_GRASP_ROUNDS): round 1 grasp_label_v0_30 (v0 rule sample,
minus the 59 clips that showed two closes, configs/laundry/grasp/r1_twoclose_ids.txt), round 2 grasp_gold_r2 (uniform random sample,
incl. those 59 re-labeled on single-close clips), round 3 grasp_gold_r3 (CNN-mined likely empties + uncertain band). A later round
wins for a repeated id. Round 4 (grasp_gold_r4_loose, re-review of the 12 'loose' events) is recorded in gold_overrides_v2.json,
not here.

gold_overrides_v2.json (11-05 user review of the confident CV errors): 'slip' = grasped then released inside one close (mixed,
excluded from event-level training), 'loose' = arranging the cloth, not a secure grasp (-> not held), 'empty' = relabelled empty.
Keys starting with '_' are notes.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from laundry_common import paths as egopaths

ROUND_DIRS = ("grasp_label_v0_30", "grasp_gold_r2", "grasp_gold_r3")
R1_DIR, R2_DIR, R3_DIR = ROUND_DIRS
TWO_CLOSE_IDS = egopaths.CONFIGS / "grasp" / "r1_twoclose_ids.txt"
GOLD_V2 = "gold_v2.csv"
OVERRIDES_V2 = "gold_overrides_v2.json"
EVENT_CLASSES = ("held", "empty")
NOT_HELD = ("empty", "loose")         # classes the P(not held) head is trained to 1 on
SLIP = "slip"
LOOSE = "loose"


def readCsv(path: Path) -> list[dict]:
    with open(path) as tFh:
        return list(csv.DictReader(tFh))


def roundGold(roundsDir: Path, twoCloseIds: Path = TWO_CLOSE_IDS) -> list[dict]:
    """held / empty gold of rounds 1-3: the gold_selection.json entry of each id + its gold label, in label order"""
    tSelected = {}
    for tDir in ROUND_DIRS:
        with open(roundsDir / tDir / "gold_selection.json") as tFh:
            for tEntry in json.load(tFh):
                tSelected[tEntry["id"]] = tEntry
    tTwoClose = set(Path(twoCloseIds).read_text().split())
    tGold = {tR["id"]: tR["gold"] for tR in readCsv(roundsDir / R1_DIR / "gold_events.csv") if tR["id"] not in tTwoClose}
    for tDir in (R2_DIR, R3_DIR):
        if (roundsDir / tDir / "gold_events.csv").exists():
            tGold.update({tR["id"]: tR["gold"] for tR in readCsv(roundsDir / tDir / "gold_events.csv")})
    return [dict(tSelected[tId], gold=tLabel) for tId, tLabel in tGold.items() if tLabel in EVENT_CLASSES]


def loadOverrides(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def eventClass(row: dict, overrides: dict) -> str:
    """the class an event trains / is scored as: its override, else its gold label"""
    return overrides.get(row["id"], row["gold"])
