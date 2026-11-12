"""11-11 prep for a take recorded WITHOUT AprilTags (user removed the hand-rig tags and 400/401; black L tapes mark the start spots).
Replaces, for such a take, the tag steps of the chain (umi_tcp_traj, tag_rig_to_lerobot --prep-only's assignment, new_take_prep,
tag_rot_extract):
  1. TAG/<take>/tagfree.json  {rig_of_arm: recorder's RIG_OF_HAND (rig "right" = unit in the LEFT hand), anchors: {} (yaw 0 until
     tagfree_yaw.py)} -> fgo_data.load_arm serves tag-free arm dicts for this take
  2. data/tag_rig_derived/<take>/assignment.json  (rig_of_arm, cam_of_arm; read by gate.py / fusion)
  3. idle trim rows APPENDED to reports/idle_trim.csv (same rule as new_take_prep.py with the tag speed term removed: both IMUs
     |gyro| > GYR, 0.3 s smoothing, MARGIN each side), pose_src "tagfree"
  4. start seeds from the L tapes: tagfree_seed.py
    python tagfree_prep.py <take> [<take> ...]"""
import csv, json, shutil, subprocess, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
TAG = Path.home() / "ego_collector/datasets/tag_rig"; DER = Path.home() / "laundry_dit_training/data/tag_rig_derived"; CSV = Path.home() / "laundry_dit_training/reports/idle_trim.csv"
GYR, MARGIN = 0.35, 0.5
RIG_OF_ARM = {"left": "right", "right": "left"}          # visual_teleop/tools/tag_rig_record.py RIG_OF_HAND (10-29)
CAM_OF_ARM = {"left": "wrist_left", "right": "wrist"}


def sm(x, n): return np.convolve(x, np.ones(n) / n, mode="same")


def main(takes):
    have = {r["take"] for r in csv.DictReader(open(CSV))}; shutil.copy2(CSV, CSV.with_suffix(".csv.bak_1111")); rows = []
    for take in takes:
        tk = TAG / take; m = json.loads((tk / "meta.json").read_text()); t0 = int(m["t_start_monotonic_ns"])
        f = tk / "tagfree.json"
        if not f.exists(): f.write_text(json.dumps(dict(rig_of_arm=RIG_OF_ARM, anchors={}, note="11-11 no AprilTags; anchors = per demo x arm "
                                                         "world yaw [deg] of the Mahony gravity frame (tagfree_yaw.py)"), indent=1))
        (DER / take).mkdir(parents=True, exist_ok=True)
        aj = DER / take / "assignment.json"
        if not aj.exists(): aj.write_text(json.dumps(dict(rig_of_arm=RIG_OF_ARM, cam_of_arm=CAM_OF_ARM, rig_labels_swapped=True, tagfree=True), indent=1))
        imu = {a: np.load(tk / f"imu_{RIG_OF_ARM[a]}.npz") for a in ("left", "right")}; end = min(int(imu[a]["host_ns"][-1]) for a in imu)
        for di, P in enumerate(p for p in m["protocol"] if p["text"].startswith("자유:")):
            a0 = t0 + int(P["t0"] * 1e9); a1 = t0 + int((P["t0"] + P["dur"]) * 1e9)
            if a0 >= end: continue
            g = np.arange(a0, min(a1, end), int(0.02e9))
            if len(g) < 50: continue
            act = np.zeros(len(g), bool)
            for a in ("left", "right"): act |= sm(np.interp(g, imu[a]["host_ns"], np.linalg.norm(imu[a]["gyro"], axis=1)), 15) > GYR
            if not act.any(): continue
            idx = np.flatnonzero(act); ks = max(0.0, (g[idx[0]] - a0) / 1e9 - MARGIN); ke = min((g[-1] - a0) / 1e9, (g[idx[-1]] - a0) / 1e9 + MARGIN); dl = (a1 - a0) / 1e9
            if take not in have:
                rows.append(dict(take=take, demo=di, demo_len_s=round(dl, 1), keep_start_s=round(ks, 2), keep_end_s=round(ke, 2), cut_head_s=round(ks, 1),
                                 cut_tail_s=round(dl - ke, 1), kept_s=round(ke - ks, 1), pose_src="tagfree"))
        subprocess.run([sys.executable, str(HERE / "tagfree_seed.py"), take], check=True)
    with open(CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["take", "demo", "demo_len_s", "keep_start_s", "keep_end_s", "cut_head_s", "cut_tail_s", "kept_s", "pose_src"]); [w.writerow(r) for r in rows]
    for r in rows: print(r)


if __name__ == "__main__":
    main(sys.argv[1:])
