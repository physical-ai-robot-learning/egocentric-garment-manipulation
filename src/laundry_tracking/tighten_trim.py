"""Tighter idle trim for the flattening takes (was orbbec/v2b/tighten_trim.py, 10-27, user: drop the waiting start and the still end):
inside the current reports/idle_trim.csv window, keep [first moving - MARGIN, last moving + MARGIN], moving = max over both arms of
the 0.3 s smoothed fused TCP speed (stitch/qa/fus_Cg_g) > V_MOVE, as measured by idle_audit.py (head_s / tail_s per demo).
Only tightens (never widens); backup idle_trim.csv.bak_tight_<time> first.
    python -m laundry_tracking.tighten_trim <idle_audit.csv from idle_audit.py> [MARGIN=0.3]
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time

from laundry_common import paths

DEFAULT_MARGIN_S = 0.3
MIN_CUT_S = 0.05                   # head and tail cuts both below this: row left as is


def tightenTrims(auditCsv: str, marginS: float) -> int:
    """-> number of tightened demos"""
    tCsv = paths.IDLE_TRIM_CSV
    tAudit = {(tRow["take"], int(tRow["demo"])): tRow for tRow in csv.DictReader(open(auditCsv))}
    shutil.copy2(tCsv, tCsv.with_suffix(f".csv.bak_tight_{time.strftime('%m%d_%H%M%S')}"))
    tRows = list(csv.DictReader(open(tCsv)))
    tFields = list(tRows[0].keys())
    tN = 0
    for tRow in tRows:
        tA = tAudit.get((tRow["take"], int(tRow["demo"])))
        if not tA or tA["head_s"] in ("", "nan"):
            continue
        tCutH = max(0.0, float(tA["head_s"]) - marginS)
        tCutT = max(0.0, float(tA["tail_s"]) - marginS)
        if tCutH < MIN_CUT_S and tCutT < MIN_CUT_S:
            continue
        tKs, tKe, tDl = float(tRow["keep_start_s"]) + tCutH, float(tRow["keep_end_s"]) - tCutT, float(tRow["demo_len_s"])
        tRow.update(keep_start_s=round(tKs, 2), keep_end_s=round(tKe, 2), cut_head_s=round(tKs, 1),
                    cut_tail_s=round(tDl - tKe, 1), kept_s=round(tKe - tKs, 1))
        tN += 1
    with open(tCsv, "w", newline="") as tHandle:
        tWriter = csv.DictWriter(tHandle, fieldnames=tFields)
        tWriter.writeheader()
        tWriter.writerows(tRows)
    return tN


def main(argv: list[str] | None = None) -> int:
    tAp = argparse.ArgumentParser(description="tighten idle_trim.csv windows from an idle audit")
    tAp.add_argument("audit", help="idle_audit.csv (take, demo, head_s, tail_s)")
    tAp.add_argument("margin", nargs="?", type=float, default=DEFAULT_MARGIN_S)
    tArgs = tAp.parse_args(argv)
    print("tightened", tightenTrims(tArgs.audit, tArgs.margin), "demos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
