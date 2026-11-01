"""10-25 derived root for the in-view training set: ~/laundry_dit_training/data/tag_rig_derived_cgg/<take>/ = the v31speed production
derived take (assignment.json, segments.json, arm-labelled umi_traj_<arm>.npz) with
  p_B          replaced inside each demo's fusion span by stitch/qa/fus_Cg_g (gated CoTracker v2 + tags, production run h-1)
  fused_valid  = gated CoTracker valid (solve3 conf > 0) or >= 1 tag on that Orbbec frame  (tag_rig_unsplit 'measured'; fills > 0.5 s
                 are masked as training targets there)
rotation (R_B_tcp, tag_rig_fuse gyro + tags) and everything else unchanged.
    python make_derived_cgg.py <pack> [<pack> ...]
10-26 new-camera takes have no v31speed derived take: CGG_SRC=tag_rig_derived CGG_DST=tag_rig_derived_cgg_1026 (tag-only prep base)."""
import json, os, shutil, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]; QA = ROOT / "stitch/qa"
LDT = Path.home() / "laundry_dit_training"; SRC = LDT / "data" / os.environ.get("CGG_SRC", "tag_rig_derived_orbbec_v31speed"); DST = LDT / "data" / os.environ.get("CGG_DST", "tag_rig_derived_cgg")


def main(packs):
    loc = {}
    for p in packs:
        for d in Path(p).iterdir():
            if (d / "solve3_left.npz").exists(): loc[d.name] = d
    takes = sorted({k[:15] for k in loc})
    for take in takes:
        o = DST / take; o.mkdir(parents=True, exist_ok=True)
        for f in ("assignment.json", "segments.json"): shutil.copy2(SRC / take / f, o / f)
        for arm in ("left", "right"):
            z = dict(np.load(SRC / take / f"umi_traj_{arm}.npz", allow_pickle=True)); t = z["t_ns"].astype(np.int64)
            p = z["p_B"].copy(); valid = (z["ntags"] >= 1).copy(); n_rep = 0
            for key in sorted(k for k in loc if k.startswith(take)):
                demo = int(key[-2:]); f = QA / "fus_Cg_g" / f"{take}_{arm}_d{demo:02d}_h-1.npz"; s3 = np.load(loc[key] / f"solve3_{arm}.npz")
                if f.exists():
                    r = np.load(f)
                    if len(r["t_ns"]) >= 2:
                        m = (t >= r["t_ns"][0]) & (t <= r["t_ns"][-1]); p[m] = np.stack([np.interp(t[m], r["t_ns"], r["p_B"][:, j]) for j in range(3)], 1); n_rep += int(m.sum())
                k = np.searchsorted(t, s3["t"].astype(np.int64)); ok = (k < len(t)) & (s3["conf"] > 0)
                valid[k[ok]] = True
            z["p_B"] = p; z["fused_valid"] = valid; z["pose_src"] = np.array("fus_Cg_g (gated CoTracker v2 + tags), 10-25")
            np.savez(o / f"umi_traj_{arm}.npz", **z)
            print(take, arm, "replaced", n_rep, "frames, measured", round(float(valid.mean()), 3), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
