#!/usr/bin/env python3
"""11-04 user: GOLD / SILVER / REJECT sets, gated position masks (gap >= 0.3 s), timeline/images never cut.
Per take: <prefix>_<HHMMSS>_S03G_{train,val,test,silver,reject}; train = GOLD minus the clean holdout, val/test = gap-free GOLD holdout (5 + 5 over
the takes), silver = all SILVER (auxiliary robustness eval), reject = REJECT (analysis only, never trained).
usage: flat_gsr_build.py PREFIX DERIVED Q.csv[,Q2.csv] CTGAP.csv[,..] [EXCLUDE_KEYS]  (env NHOLD=5)"""
import json, os, pathlib, subprocess, sys, time, collections
import pandas as pd
S = pathlib.Path(__file__).resolve().parent
L = pathlib.Path.home() / "laundry_dit_training"; TR = pathlib.Path.home() / "ego_collector/datasets/tag_rig"
PY = str(pathlib.Path.home() / "xvla-mac/bin/python")
PREFIX, DERIVED = sys.argv[1], sys.argv[2]
EXCL = set(sys.argv[5].split(",")) if len(sys.argv) > 5 and sys.argv[5] else set()
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
q = pd.concat([pd.read_csv(S / f, dtype={"take": str}) for f in sys.argv[3].split(",")]); q["key"] = [f"{r.take}/demo{int(r.demo):02d}" for r in q.itertuples()]
cls = q.drop_duplicates("key", keep="last").set_index("key")["class"]
_ov = S / "class_overrides_1104.json"                         # 11-04 user per-demo class overrides (e.g. a REJECT fixed by a manual front trim)
if _ov.exists():
    for k, v in json.loads(_ov.read_text()).items():
        if k in cls.index: cls[k] = v["class"]
g = pd.concat([pd.read_csv(S / f, dtype={"take": str}) for f in sys.argv[4].split(",")]); g["key"] = [f"{r.take}/demo{int(r.demo):02d}" for r in g.itertuples()]
gg = g.drop_duplicates("key", keep="last").set_index("key"); gap = gg["gap_frames"]; mgap = gg["max_gap_s"]
GMAX = float(os.environ.get("GOLD_GAP_MAX", "0.25"))   # 11-04 19:5x user: "0.25까지는 사용해도 되는거 같아" (was 0.2)   # 11-04 user: "0.2 이상은 못쓰겠네 train은 0.2만 남겨줘" -> GOLD with a gap >= 0.2 s = goldgap (not trained, not main val/test)
keys = sorted(k for k in cls.index if k not in EXCL)
split = {k: {"GOLD": "train", "SILVER": "silver", "REJECT": "reject"}[cls[k]] for k in keys}
for k in keys:
    if cls[k] == "GOLD" and mgap.get(k, 9.0) >= GMAX: split[k] = "goldgap"
# clean holdout: gap-free GOLD, round-robin over takes (largest first), middle candidate, alternating val / test
NH = int(os.environ.get("NHOLD", "5")); by = collections.defaultdict(list)
for k in keys:
    if cls[k] == "GOLD" and gap.get(k, 1) == 0: by[k.split("/")[0]].append(k)
order = sorted(by, key=lambda t: -len(by[t])); held = {"val": [], "test": []}; r = 0
while len(held["val"]) + len(held["test"]) < 2 * NH and any(len(v) > 1 for v in by.values()):
    t = order[r % len(order)]; r += 1
    if len(by[t]) > 1:
        k = by[t].pop(len(by[t]) // 2); sp = "val" if len(held["val"]) <= len(held["test"]) else "test"; held[sp].append(k); split[k] = sp
if os.environ.get("GOLDGAP_TO", "") == "train":                # 11-04 user: today's goldgap -> TRAIN ("그럼 써야지"), positions masked by gated 0.3 s
    for k in keys:
        if split[k] == "goldgap": split[k] = "train"
elif os.environ.get("GOLDGAP_TO_VALTEST", "1") == "1":         # 11-04 user: "goldgap 은 test나 발로 쓰자" -> goldgap alternates val / test
    gk = sorted(k for k in keys if split[k] == "goldgap")
    for i, k in enumerate(gk): split[k] = "val" if i % 2 == 0 else "test"
    held = {"val": held["val"] + gk[0::2], "test": held["test"] + gk[1::2]}
if os.environ.get("SILVER_NOGAP_TRAIN", "1") == "1":         # 11-04 user: "silver에서 cp gap 없는거는 다 train으로" (0 CoTracker-gap frames in the trimmed window)
    for k in keys:
        if split[k] == "silver" and gap.get(k, 1) == 0: split[k] = "train"
for k in filter(None, os.environ.get("FORCE_TRAIN", "").split(",")):   # 11-04 22:2x user, per demo after watching: "이건 써도 되겠는데? 트레인에 추가 해줘" (e.g. 184244/demo03, GOLD gap 0.4 s)
    if k in split: split[k] = "train"; held = {s: [x for x in v if x != k] for s, v in held.items()}
cnt = collections.Counter(split.values()); log(f"{PREFIX}: {dict(cnt)}  val {held['val']}  test {held['test']}")
(S / f"{PREFIX}_gsr_split.json").write_text(json.dumps({s: sorted(k for k, v in split.items() if v == s) for s in cnt}, indent=1))
env = dict(os.environ, FILL_MASK_S="0.3")


def build(t):
    ks = [k for k in keys if k.startswith(t + "/")]; name = f"{PREFIX}_{t[9:]}_S03G"; sp = S / f"{name}_split.json"
    eps = collections.defaultdict(list)
    for k in ks: eps[split[k]].append(k)
    sp.write_text(json.dumps({"method": "11-04 GOLD/SILVER/REJECT, gated FILL_MASK_S 0.3, clean-GOLD holdout", "episodes": eps}))
    cmd = [PY, "datasets/tag_rig_unsplit.py", "--label", "fused", "--speed", "0.3", "--mask-mode", "gated", "--grasp-fill-drop", "off", "--ik-fail-mm", "30",
           "--render", "0", "--overwrite", "--grip-label", "cont", "--out", f"data/lerobot/{name}", "--takes", str(TR / t), "--derived", DERIVED, "--demos", *ks, "--old-split", str(sp)]
    r = subprocess.run(cmd, cwd=L, capture_output=True, text=True, env=env); (S / f"build_{name}.log").write_text(r.stdout + r.stderr)
    res = {}
    for s in eps:
        d = L / "data/lerobot" / f"{name}_{s}"
        if (d / "meta/info.json").exists(): info = json.loads((d / "meta/info.json").read_text()); res[s] = (info["total_episodes"], info["total_frames"])
    miss = {s: len(v) - res.get(s, (0, 0))[0] for s, v in eps.items() if len(v) != res.get(s, (0, 0))[0]}
    log(f"built {name}: {res}" + (f"  MISSING {miss} -> {(r.stdout + r.stderr)[-300:]}" if miss else "")); return t, res


from concurrent.futures import ThreadPoolExecutor
_only = set(filter(None, os.environ.get("ONLY_TAKES", "").split(",")))   # rebuild just these takes (split json still covers all)
with ThreadPoolExecutor(3) as ex: out = dict(ex.map(build, sorted({k.split('/')[0] for k in keys if not _only or k.split('/')[0] in _only})))
tot = collections.Counter()
for v in out.values():
    for s, (e, f) in v.items(): tot[s] += e
log(f"{PREFIX}_GSR_DONE episodes {dict(tot)}")
