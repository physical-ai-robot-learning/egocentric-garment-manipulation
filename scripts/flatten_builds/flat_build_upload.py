#!/usr/bin/env python3
"""11-02 v2 (user "ct gap있는거는 다 test로" -> SILVER with a gap flag = keys_flat1102_test_gap.txt -> TEST split; train/test both uploaded).
11-02 user "데이터 변환된거는 바로바로 올려줘": build each finished flattening take as its own LeRobot set (0.5x, Cg, continuous jaw =
--grip-label cont, all TRAIN, GOLD+SILVER keys from every keys_*_gs.txt incl. the seed rescue) and rsync it to the RunPod volume
(staging pod).  The 10-30 48-demo pass set (cgg_1030f) is built once the same way.  Per-take sets are aggregated on the pod later."""
import glob, json, os, pathlib, subprocess, time, collections
S = pathlib.Path("/private/tmp/claude-501/-Users-jeonghwanlee/f6645172-99ad-4391-b7ec-ba0752d58ef5/scratchpad")
L = pathlib.Path.home() / "laundry_dit_training"; TR = pathlib.Path.home() / "ego_collector/datasets/tag_rig"
PY = str(pathlib.Path.home() / "xvla-mac/bin/python")
SSH = f"ssh -o StrictHostKeyChecking=no -i {pathlib.Path.home()}/.runpod/ssh/runpodctl-ssh-key -p 39064"; HOST = "root@38.80.152.147"
EXPECT = ["20261102_" + t for t in "121342 130406 131156 132540 133624 134434 135136 140013 140903 142422 144657 145344 151622 152215 152909 154442".split()]
DONE_F = S / "flat_build_done.json"; done = json.loads(DONE_F.read_text()) if DONE_F.exists() else {}
log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


TEST = set(open(S / "keys_flat1102_test_gap.txt").read().split())


def upload(d):
    u = subprocess.run(f"/opt/homebrew/bin/rsync -rlt --exclude '._*' -e '{SSH}' {d} {HOST}:/workspace/laundry_dit_training/data/lerobot/", shell=True, capture_output=True, text=True)
    if u.returncode != 0: log(f"UPLOAD FAILED {d.name}: {u.stderr[-200:]}"); return False
    log(f"uploaded {d.name}"); return True


def build(name, takes, derived, keys):
    tr = [k for k in keys if k not in TEST]; te = [k for k in keys if k in TEST]
    sp = S / f"{name}_split.json"; sp.write_text(json.dumps({"method": f"11-02 flatten {name}: Cg, 0.5x, continuous jaw; TEST = SILVER with a CT gap flag", "episodes": {"train": tr, "val": [], "test": te}}))
    cmd = [PY, "datasets/tag_rig_unsplit.py", "--label", "fused", "--speed", "0.5", "--mask-mode", "stitched", "--ik-fail-mm", "30", "--render", "0", "--overwrite",
           "--grip-label", "cont", "--out", f"data/lerobot/{name}", "--takes", *[str(TR / t) for t in takes], "--derived", derived, "--demos", *keys, "--old-split", str(sp)]
    r = subprocess.run(cmd, cwd=L, capture_output=True, text=True)
    (S / f"build_{name}.log").write_text(r.stdout + r.stderr)
    out = L / "data/lerobot" / f"{name}_train"
    if r.returncode != 0 or not (out / "meta/info.json").exists(): log(f"BUILD FAILED {name}: {(r.stdout + r.stderr)[-300:]}"); return None
    info = json.loads((out / "meta/info.json").read_text()); log(f"built {name}: {info['total_episodes']} eps {info['total_frames']} frames")
    ok = upload(out); tst = L / "data/lerobot" / f"{name}_test"
    if tst.exists(): ok = upload(tst) and ok
    return dict(eps=info["total_episodes"], frames=info["total_frames"], test=len(te)) if ok else None


if "flat1030_F48" not in done:     # 11-02 v2: the v1 build was cut off at 34/48 episodes when its loop was stopped -> rebuild from scratch
    keys = open(S / "keys_f58_pass.txt").read().split(); takes = sorted({k.split("/")[0] for k in keys})
    r = build("flat1030_F48_S05C", takes, "data/tag_rig_derived_cgg_1030f", keys)
    if r: done["flat1030_F48"] = r; DONE_F.write_text(json.dumps(done, indent=1))
while True:
    keys = collections.defaultdict(list)
    for f in glob.glob(str(S / "keys_fl1102*_gs.txt")) + [str(S / "keys_f7_gs.txt")]:
        for k in open(f).read().split(): keys[k.split("/")[0]].append(k)
    for t in EXPECT:
        if t in done or t not in keys: continue
        if t in ("20261102_140013",) and not (S / "keys_fl1102r_gs.txt").exists(): continue
        r = build(f"flat1102_{t[9:]}_S05C", [t], "data/tag_rig_derived_cgg_1102", sorted(set(keys[t])))
        if r: done[t] = r; DONE_F.write_text(json.dumps(done, indent=1))
    if all(t in done for t in EXPECT): break
    time.sleep(60)
tot = sum(v["eps"] for v in done.values()); log(f"ALL_UPLOADED {len(done)} sets, {tot} episodes, {sum(v['frames'] for v in done.values())} frames")
