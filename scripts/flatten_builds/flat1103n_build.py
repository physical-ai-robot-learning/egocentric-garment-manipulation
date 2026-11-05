#!/usr/bin/env python3
"""11-03 user: "100데모 ... 5개 val 5개 test ... 90train", "속도는 0.3으로", train on the 2x RTX PRO 6000 (bh-ai-6000).
Per-take LeRobot sets of the FL1103N chain (new camera pose, robot_cam_v006): Cg labels (--label fused), 0.3x, continuous jaw, stitched masks.
VAL 5 / TEST 5 = spread over the 7 takes (round-robin, a middle demo of each take, alternating val/test), rest = train.
Built sets go to gpu-5090 /srv/data/johann/ldt6000/data/lerobot/ (the 6000 pulls them through Ray)."""
import json, pathlib, subprocess, time, collections
S = pathlib.Path("/private/tmp/claude-501/-Users-jeonghwanlee/f6645172-99ad-4391-b7ec-ba0752d58ef5/scratchpad")
L = pathlib.Path.home() / "laundry_dit_training"; TR = pathlib.Path.home() / "ego_collector/datasets/tag_rig"
PY = str(pathlib.Path.home() / "xvla-mac/bin/python"); DERIVED = "data/tag_rig_derived_cgg_1103n"
DST = "bh-ai-5090@gpu-5090:/srv/data/johann/ldt6000/data/lerobot/"
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

keys = collections.defaultdict(list)
for k in open(S / "keys_fl1103n_gs.txt").read().split(): keys[k.split("/")[0]].append(k)
takes = sorted(keys)
for t in takes: keys[t] = sorted(set(keys[t]))
# held-out: round-robin over takes (largest first), the middle demo not yet taken, alternating val / test until 5 + 5
held = {"val": [], "test": []}; used = set(); order = sorted(takes, key=lambda t: -len(keys[t])); r = 0
while len(held["val"]) + len(held["test"]) < 10:
    t = order[r % len(order)]; free = [k for k in keys[t] if k not in used]
    if len(free) > 2:
        k = free[len(free) // 2]; used.add(k); sp = "val" if len(held["val"]) <= len(held["test"]) else "test"; held[sp].append(k)
    r += 1
import os
if os.environ.get("FIXED_SPLIT") == "1": held = json.loads((S / "fl1103n_split.json").read_text())
else: (S / "fl1103n_split.json").write_text(json.dumps(held, indent=1))
VAL, TEST = set(held["val"]), set(held["test"])
assert (VAL | TEST) <= {k for t in takes for k in keys[t]}, "held-out key not in GOLD+SILVER keys"
log(f"VAL {sorted(VAL)}\nTEST {sorted(TEST)}")


def upload(d):
    if os.environ.get("UPLOAD", "1") == "0": return True
    u = subprocess.run(f"/opt/homebrew/bin/rsync -rlt --delete --exclude '._*' {d}/ {DST}{d.name}/", shell=True, capture_output=True, text=True)
    if u.returncode != 0: log(f"UPLOAD FAILED {d.name}: {u.stderr[-200:]}"); return False
    return True


def build(t):
    name = f"flat1103n_{t[9:]}_S03CT"; ks = keys[t]
    sp = S / f"{name}_split.json"
    sp.write_text(json.dumps({"method": "11-03 new-camera flatten: Cg, 0.3x, continuous jaw; VAL/TEST spread over takes",
                              "episodes": {"train": [k for k in ks if k not in VAL | TEST], "val": [k for k in ks if k in VAL], "test": [k for k in ks if k in TEST]}}))
    cmd = [PY, "datasets/tag_rig_unsplit.py", "--label", "fused", "--speed", "0.3", "--mask-mode", "stitched", "--ik-fail-mm", "30", "--render", "0", "--overwrite",
           "--grip-label", "cont", "--out", f"data/lerobot/{name}", "--takes", str(TR / t), "--derived", DERIVED, "--demos", *ks, "--old-split", str(sp)]
    r = subprocess.run(cmd, cwd=L, capture_output=True, text=True); (S / f"build_{name}.log").write_text(r.stdout + r.stderr)
    res = {}
    for s in ("train", "val", "test"):
        d = L / "data/lerobot" / f"{name}_{s}"
        if (d / "meta/info.json").exists():
            info = json.loads((d / "meta/info.json").read_text()); res[s] = (info["total_episodes"], info["total_frames"])
            if not upload(d): return t, None
    if "train" not in res: log(f"BUILD FAILED {name}: {(r.stdout + r.stderr)[-400:]}"); return t, None
    log(f"built+uploaded {name}: {res}"); return t, res


from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(3) as ex: out = dict(ex.map(build, takes))
(S / "fl1103n_build_done.json").write_text(json.dumps(out, indent=1))
tot = collections.Counter()
for v in out.values():
    for s, (e, f) in (v or {}).items(): tot[s + "_eps"] += e; tot[s + "_frames"] += f
log(f"FL1103N_BUILD_DONE failed={[t for t, v in out.items() if not v]} {dict(tot)}")
