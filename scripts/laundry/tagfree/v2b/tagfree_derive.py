"""11-11 tag-free take -> the derived take the LeRobot converter reads (datasets/tag_rig_unsplit.py --derived ... --label fused).
Writes the prep base that tag_rig_to_lerobot --prep-only + umi_tcp_traj would have written for a tagged take, with NO tag data:
  data/tag_rig_derived/<take>/{assignment.json (tagfree_prep), segments.json (umi_tcp_traj.segments(meta)), umi_traj_<arm>.npz}
  umi_traj_<arm>.npz: Orbbec frame times, ntags 0, p_B / R_B_tcp NaN (filled per demo span by the fusion run below), robot_cal = camcal
then runs fusion/make_derived_cgg.py with CGG_VAR=<fusion var> CGG_ROT=1 (position AND rotation = the tag-free fusion run: CoTracker +
AHRS with the L-tape / motion yaw anchors) -> data/<dst>/<take>/.
    python tagfree_derive.py <take> <ctpack dir> [--var fus_Bg_tfL] [--dst tag_rig_derived_tagfree]"""
import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(Path.home() / "ego_collector/scripts")]
import camcal  # noqa
from umi_tcp_traj import segments  # noqa
TAG = Path.home() / "ego_collector/datasets/tag_rig"; DER = Path.home() / "laundry_dit_training/data/tag_rig_derived"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("take"); ap.add_argument("pack"); ap.add_argument("--var", default="fus_Bg_tfL"); ap.add_argument("--dst", default="tag_rig_derived_tagfree")
    a = ap.parse_args(); tk = TAG / a.take; d = DER / a.take; tf = json.loads((tk / "tagfree.json").read_text())
    asg = json.loads((d / "assignment.json").read_text()); meta = json.loads((tk / "meta.json").read_text())
    (d / "segments.json").write_text(json.dumps(segments(meta), ensure_ascii=False, indent=1))
    t = np.load(tk / "t_ns.npy").astype(np.int64); n = len(t); cal = str(camcal.path(a.take).relative_to(Path.home() / "ego_collector"))
    for arm in ("left", "right"):
        nan3 = np.full((n, 3), np.nan); nan33 = np.full((n, 3, 3), np.nan)
        np.savez(d / f"umi_traj_{arm}.npz", t_ns=t, p_C=nan3, R_C_tcp=nan33, ntags=np.zeros(n, np.int64), rot=np.array("imu"), depth=np.array(False),
                 imu_fill=np.array(False), single_new_tag=np.zeros(n, bool), guard_rot_deg=np.full(n, np.nan), p_B=nan3.copy(), R_B_tcp=nan33.copy(),
                 robot_cal=np.array(cal), source_rig_label=np.array(asg["rig_of_arm"][arm]), pose_src=np.array("tagfree stub (11-11): no tags"),
                 yaw_src=np.array(tf.get("yaw_src", "")))
    print(f"[derive] base {d}: segments {len(json.loads((d / 'segments.json').read_text()))}, stubs {n} frames, robot_cal {cal}")
    env = dict(os.environ, CGG_SRC="tag_rig_derived", CGG_DST=a.dst, CGG_VAR=a.var, CGG_ROT="1")
    subprocess.run([sys.executable, str(HERE / "fusion/make_derived_cgg.py"), a.pack], env=env, check=True)
    # the fusion output is cropped to the first / last CoTracker observation, so a demo window can start / end a few frames (right d0:
    # 68) outside it; a tagged take had tag poses there.  Hold the nearest fused pose inside each demo window (robotize_ik needs finite
    # poses) and leave those frames fused_valid = False -> the converter's gated fill mask drops them from the loss when > FILL_MASK_S.
    o = Path.home() / "laundry_dit_training/data" / a.dst / a.take
    for arm in ("left", "right"):
        z = dict(np.load(o / f"umi_traj_{arm}.npz", allow_pickle=True)); t = z["t_ns"].astype(np.int64); nfill = 0
        t0 = int(meta["t_start_monotonic_ns"]); P = [p for p in meta["protocol"] if p["text"].startswith("자유:")]
        for i_, p_ in enumerate(P):      # the whole cue window (robotize_ik / the converter use it, not the idle-trimmed one)
            W = dict(demo=i_, w0=t0 + int((p_["t0"] - 0.5) * 1e9), w1=t0 + int((p_["t0"] + p_["dur"] + 0.5) * 1e9))   # +-0.5 s: interpolation at the edges
            m = np.flatnonzero((t >= W["w0"]) & (t <= W["w1"])); ok = m[np.isfinite(z["p_B"][m, 0]) & np.isfinite(z["R_B_tcp"][m, 0, 0])]
            if not len(ok): print(f"[derive] {arm} d{W['demo']}: no fused pose in the window"); continue
            bad = np.setdiff1d(m, ok); j = ok[np.clip(np.searchsorted(ok, bad), 0, len(ok) - 1)]; j2 = ok[np.clip(np.searchsorted(ok, bad) - 1, 0, len(ok) - 1)]
            near = np.where(np.abs(t[j] - t[bad]) <= np.abs(t[j2] - t[bad]), j, j2); z["p_B"][bad] = z["p_B"][near]; z["R_B_tcp"][bad] = z["R_B_tcp"][near]; nfill += len(bad)
        # robotize_ik converts the WHOLE take's R_B_tcp (Rotation.from_matrix fails on NaN): between demos (never converted) hold too
        ok = np.flatnonzero(np.isfinite(z["p_B"][:, 0]) & np.isfinite(z["R_B_tcp"][:, 0, 0])); bad = np.setdiff1d(np.arange(len(t)), ok)
        if len(ok) and len(bad):
            near = ok[np.clip(np.searchsorted(ok, bad), 0, len(ok) - 1)]; z["p_B"][bad] = z["p_B"][near]; z["R_B_tcp"][bad] = z["R_B_tcp"][near]
        np.savez(o / f"umi_traj_{arm}.npz", **z); print(f"[derive] {arm}: held {nfill} edge frames inside demo windows + {len(bad)} between demos (fused_valid stays False)")


if __name__ == "__main__":
    main()
