#!/usr/bin/env python3
"""11-03 user "cp error 있나 확인하게 한번 시각화 후 desktop": fusion QA video (CT gaps marked by render_fusion_qa) for every converted
11-03 flatten demo -> ~/Desktop/flatten_1103/videos/<take>_d<NN>_<CLASS>.mp4.  Excluded takes (user) skipped.  usage: render_1103.py PACK [PACK..]"""
import csv, pathlib, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
S = pathlib.Path("/private/tmp/claude-501/-Users-jeonghwanlee/f6645172-99ad-4391-b7ec-ba0752d58ef5/scratchpad")
import os; OUT = pathlib.Path(os.environ.get("RENDER_OUT", str(pathlib.Path.home() / "Desktop/flatten_1103/videos"))); F = pathlib.Path.home() / "ego_collector/datasets/m3_runs/_imu_consistency/orbbec/v2b/fusion"
PY = str(pathlib.Path.home() / "ego_collector/.venv/bin/python")
EXCL = set(open(S / "exclude_takes_1103.txt").read().split())
jobs = []
for pk in sys.argv[1:]:
    qf = S / f"q_{pk}.csv"
    for r in csv.DictReader(open(qf)):
        if r["take"] in EXCL: continue
        jobs.append((r["take"], int(r["demo"]), r["class"], pk, str(qf)))
def one(j):
    t, d, c, pk, qf = j; OUT.mkdir(parents=True, exist_ok=True); name = f"{t}_d{d + 1:02d}_{c}.mp4"
    if (OUT / name).exists(): return
    tmp = OUT / f".tmp_{t}_{d}"; tmp.mkdir(exist_ok=True)
    r = subprocess.run(["nice", "-n", "10", PY, "render_fusion_qa.py", "--pack", str(S / pk), "--var", "fus_Cg_g", "--raw-var", "fus_R_g", "--demos", f"{t}:{d}",
                        "--out", str(tmp), "--quality-csv", qf], cwd=F, capture_output=True, text=True)
    vids = list(tmp.glob("*.mp4"))
    if vids: vids[0].rename(OUT / name)
    else: print("FAILED", t, d, (r.stdout + r.stderr)[-200:], flush=True)
    for x in tmp.iterdir(): x.unlink()
    tmp.rmdir()
with ThreadPoolExecutor(2) as ex: list(ex.map(one, jobs))
print("RENDER_DONE", len(list(OUT.glob("*.mp4"))), "videos", flush=True)
