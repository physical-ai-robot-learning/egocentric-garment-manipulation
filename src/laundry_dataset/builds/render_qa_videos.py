#!/usr/bin/env python3
"""11-03 fusion QA videos of converted flatten demos (dataset stage QA; was render_1103.py). User "cp error 있나 확인하게 한번 시각화 후
desktop": for every demo of every pack's quality table (q_<pack>.csv: take, demo, class) except excluded takes, the fusion QA
video
(CT gaps marked by render_fusion_qa, variant fus_Cg_g over fus_R_g) -> <RENDER_OUT>/<take>_d<NN>_<CLASS>.mp4 (NN = demo + 1). An
existing video is kept; two renders run in parallel at nice 10.
render_fusion_qa.py runs from the fusion working tree (laundry_common.paths.FUSION_WORK / orbbec/v2b/fusion) with the recording
workspace interpreter (EGO_DATA_ROOT/.venv/bin/python). Environment: RENDER_OUT (default EGO_DESKTOP/flatten_1103/videos).
usage: python -m laundry_dataset.builds.render_qa_videos PACK [PACK ...] [--work-dir DIR] [--exclude-file FILE]
"""
from __future__ import annotations

import argparse
import csv
import os
import pathlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from laundry_common import paths as egopaths

RENDERER = "render_fusion_qa.py"
FUSION_DIR = egopaths.FUSION_WORK / "orbbec/v2b/fusion"
EGO_PY = egopaths.DATA_ROOT / ".venv/bin/python"
DEFAULT_OUT = egopaths.DESKTOP / "flatten_1103/videos"
RENDER_WORKERS = 2
NICE = "10"
FAIL_TAIL_CHARS = 200


def _render(job, out, work):
    tT, tD, tC, tPack, tQualityFile = job
    out.mkdir(parents=True, exist_ok=True)
    tName = f"{tT}_d{tD + 1:02d}_{tC}.mp4"
    if (out / tName).exists():
        return
    tTmp = out / f".tmp_{tT}_{tD}"
    tTmp.mkdir(exist_ok=True)
    tR = subprocess.run(["nice", "-n", NICE, str(EGO_PY), RENDERER, "--pack", str(work / tPack), "--var", "fus_Cg_g", "--raw-var",
                         "fus_R_g",
                         "--demos", f"{tT}:{tD}", "--out", str(tTmp), "--quality-csv", tQualityFile], cwd=FUSION_DIR,
                        capture_output=True, text=True)
    tVideos = list(tTmp.glob("*.mp4"))
    if tVideos:
        tVideos[0].rename(out / tName)
    else:
        print("FAILED", tT, tD, (tR.stdout + tR.stderr)[-FAIL_TAIL_CHARS:], flush=True)
    for tX in tTmp.iterdir():
        tX.unlink()
    tTmp.rmdir()


def main(argv: list[str] | None = None) -> int:
    tAp = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tAp.add_argument("packs", nargs="+", help="pack names under --work-dir (each with q_<pack>.csv next to it)")
    tAp.add_argument("--work-dir", default=".")
    tAp.add_argument("--exclude-file", default="", help="excluded takes (default <work-dir>/exclude_takes_1103.txt)")
    tA = tAp.parse_args(argv)
    tWork = pathlib.Path(tA.work_dir)
    tOut = pathlib.Path(os.environ.get("RENDER_OUT", str(DEFAULT_OUT)))
    tExclude = set(pathlib.Path(tA.exclude_file or tWork / "exclude_takes_1103.txt").read_text().split())
    tJobs = []
    for tPack in tA.packs:
        tQualityFile = tWork / f"q_{tPack}.csv"
        with open(tQualityFile) as tFh:
            for tR in csv.DictReader(tFh):
                if tR["take"] in tExclude:
                    continue
                tJobs.append((tR["take"], int(tR["demo"]), tR["class"], tPack, str(tQualityFile)))
    with ThreadPoolExecutor(RENDER_WORKERS) as tPool:
        list(tPool.map(lambda j: _render(j, tOut, tWork), tJobs))
    print("RENDER_DONE", len(list(tOut.glob("*.mp4"))), "videos", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
