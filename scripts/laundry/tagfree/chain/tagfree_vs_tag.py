"""11-11 validation: the full tag-free chain run on a tag-free CLONE of a tagged take (20261105_164113 -> 20261105_164114, tags unused)
vs the tag measurements of the original take.  Per demo x arm, over normalised demo time (5 bins):
  TCP abs error  |p_tf - p_tag|            (p_tag = tag TCP, frames with >= 1 tag)
  body abs error |body_tf - body_tag|      (body = TCP + R o: separates position drift from rotation)
  rotation error angle(R_tf R_tag^T)       (R_tag = per-frame tag PnP rotation, >= 2 tags)
  REL 0.48 s     |R_t^T (p(t+k) - p(t))| difference, tf vs tag (the training label quantity)
    python tagfree_vs_tag.py [src take] [clone take] [var]"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
ROOT = Path.home() / "ego_collector/datasets/m3_runs/_imu_consistency"; sys.path[:0] = [str(ROOT), str(ROOT / "stitch")]
import core as C  # noqa
FD = C.FD
src, clone = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("20261105_164113", "20261105_164114"); var = sys.argv[3] if len(sys.argv) > 3 else "fus_Bg_tfL"
OFF = json.loads((ROOT / "orbbec/v2b/body_offset_by_rig_v1.json").read_text())["offsets_by_rig"]; RIG = {"left": "right", "right": "left"}
QA = ROOT / "stitch/qa" / var; rows = []
for a in ("left", "right"):
    D = FD.load_arm(src, a, with_slam=False); t = D["t"]; o = np.array(OFF[RIG[a]]) / 1000
    gt_ok = D["ntags"] >= 1; rot_ok = (D["n_tagrot"] >= 2) & np.isfinite(D["R_tag"][:, 0, 0])
    for W in FD.trims()[clone]:
        f = QA / f"{clone}_{a}_d{W['demo']:02d}_h-1.npz"
        if not f.exists(): continue
        z = np.load(f); tz = z["t_ns"].astype(np.int64); k = np.flatnonzero((t >= max(W["w0"], tz[0])) & (t <= min(W["w1"], tz[-1])))
        if len(k) < 30: continue
        i = np.clip(np.searchsorted(tz, t[k]), 0, len(tz) - 1); P = z["p_B"][i]; Rt = z["R_B_tcp"][i]
        frac = (t[k] - t[k][0]) / max(t[k][-1] - t[k][0], 1); b = np.minimum((frac * 5).astype(int), 4)
        g = gt_ok[k]; eT = np.linalg.norm(P - D["p_tag"][k], axis=1) * 1000
        bt = P + Rt @ o; bg = D["p_tag"][k] + D["R_fused"][k] @ o; eB = np.linalg.norm(bt - bg, axis=1) * 1000
        r = rot_ok[k]; eR = np.full(len(k), np.nan)
        if r.any(): eR[r] = np.degrees((R.from_matrix(Rt[r]) * R.from_matrix(D["R_tag"][k][r]).inv()).magnitude())
        for bi in range(5):
            m = b == bi
            rows.append(dict(arm=a, demo=W["demo"], bin=bi, tcp=np.nanmedian(eT[m & g]) if (m & g).sum() > 3 else np.nan,
                             body=np.nanmedian(eB[m & g]) if (m & g).sum() > 3 else np.nan, rot=np.nanmedian(eR[m]) if np.isfinite(eR[m]).sum() > 3 else np.nan))
import pandas as pd
df = pd.DataFrame(rows)
print(f"{src} (tags) vs {clone} (tag-free chain, {var}); median over demos per time bin (0 = demo start, 4 = end)")
for a in ("left", "right"):
    d = df[df.arm == a].groupby("bin")[["tcp", "body", "rot"]].median()
    print(f"{a:5s} TCP mm " + " ".join(f"{v:5.0f}" for v in d.tcp) + " | body mm " + " ".join(f"{v:5.0f}" for v in d.body) + " | rot deg " + " ".join(f"{v:5.1f}" for v in d.rot))
print(df.groupby(["arm", "demo"])[["tcp", "body", "rot"]].median().round(1).to_string())
