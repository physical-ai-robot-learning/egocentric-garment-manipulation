"""ORB-SLAM3 run dir (allmaps.txt: t map lost x y z qx qy qz qw = final Twc per frame in its map world; online.txt: t state map imuinit nmaps ...)
-> MASt3R-style segs dir (CSV per segment + manifest.json) for umi_fuse_segments.load_segments / umi_slam_eval.py.
tracked frame = final pose exists AND online tracking state == OK (2); RECENTLY_LOST (IMU-only prediction) frames are NOT counted.
--mode map: one segment per continuous stretch of one map (split at map change or gap > --max-gap frames)
--mode win: the map stretches chopped into --win-frame windows every --stride frames (same window layout as the MASt3R segs: 360/315)"""
import argparse, json, csv
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("frames_json"); ap.add_argument("out"); ap.add_argument("--offset-ms", type=float, required=True)
ap.add_argument("--tag", required=True); ap.add_argument("--max-gap", type=int, default=15); ap.add_argument("--mode", default="map", choices=["map", "win"])
ap.add_argument("--win", type=int, default=360); ap.add_argument("--stride", type=int, default=315); ap.add_argument("--allow-rl", action="store_true")
a = ap.parse_args(); run = Path(a.run)
fr = json.loads(Path(a.frames_json).read_text())["frames"]; ft = np.array([f["t_ns"] for f in fr], np.int64); fi = np.array([f["i"] for f in fr]); n = len(fr)
def kidx(t):
    tn = int(round(t * 1e9)) - int(round(a.offset_ms * 1e6)); k = int(np.argmin(np.abs(ft - tn))); return k if abs(ft[k] - tn) < 2e6 else None
state, imu = {}, {}
for line in open(run / "online.txt"):
    v = line.split(); k = kidx(float(v[0]))
    if k is not None: state[k] = int(v[1]); imu[k] = int(v[3])
pose, mp = {}, {}
for line in open(run / "allmaps.txt"):
    v = line.split(); k = kidx(float(v[0]))
    if k is None or int(v[2]) != 0 or int(v[1]) < 0: continue
    if state.get(k) == 2 or (a.allow_rl and state.get(k) == 3): pose[k] = [float(x) for x in v[3:10]]; mp[k] = int(v[1])
stretch, cur = [], []
for k in sorted(pose):
    if cur and (mp[k] != mp[cur[-1]] or k - cur[-1] > a.max_gap): stretch.append(cur); cur = []
    cur.append(k)
if cur: stretch.append(cur)
segs = []
for s in stretch:
    if a.mode == "map": segs.append(s); continue
    st = s[0]
    while st <= s[-1]:
        w = [k for k in s if st <= k < st + a.win]
        if len(w) >= 30: segs.append(w)
        if st + a.win > s[-1]: break
        st += a.stride
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); man = []
for s in segs:
    st, en = s[0], s[-1] + 1; name = f"{st:06d}_{en:06d}.csv"; ss = set(s)
    with open(out / name, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame_idx", "timestamp", "state", "x", "y", "z", "q_x", "q_y", "q_z", "q_w", "match_frac", "map_id", "imu_init"])
        for k in range(st, en):
            if k in ss: w.writerow([k - st, f"{(ft[k] - ft[st]) / 1e9:.6f}", "TRACKING", *[f"{x:.7f}" for x in pose[k]], "nan", mp[k], imu.get(k, 0)])
            else: w.writerow([k - st, f"{(ft[k] - ft[st]) / 1e9:.6f}", "LOST", 0, 0, 0, 0, 0, 0, 1, "nan", -1, imu.get(k, 0)])
    man.append(dict(start=int(fi[st]), end=int(fi[en - 1]) + 1, cut=en - st, good=len(s), map_id=mp[st], csv=name))
(out / "manifest.json").write_text(json.dumps(dict(tag=a.tag, n=n, source=str(run), mode=a.mode, segments=man), indent=1))
ok = np.array([state.get(k) == 2 for k in range(n)]); first = int(np.argmax(ok)) if ok.any() else -1
init = [k for k in sorted(imu) if imu[k] == 1]; t_first = ft[min(state)] if state else ft[0]
print(json.dumps(dict(tag=a.tag, mode=a.mode, online_ok_pct=round(100 * ok.mean(), 1), final_pose_ok_pct=round(100 * len(pose) / n, 1),
      imu_init_s_after_first_fed_frame=None if not init else round((ft[init[0]] - t_first) / 1e9, 1), imu_init_frac_of_ok=round(np.mean([imu.get(k, 0) for k in pose]) if pose else 0, 2),
      maps=len(set(mp.values())), segs=len(man), longest=max([m["good"] for m in man], default=0))))
