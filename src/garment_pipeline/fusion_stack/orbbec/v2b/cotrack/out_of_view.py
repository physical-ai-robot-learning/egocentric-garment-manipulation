"""10-25 out-of-view audit per episode (both arms), added to the episode quality table:
  <A>_exit_n / <A>_exit_s   CoTracker gaps >= MIN_GAP_S whose last valid body position projects within EDGE px of the image border
                            (the hand left the Orbbec view), count / total seconds, with the edge of the longest one
  <A>_outproj_s             seconds the fused rig BODY (fus_Cg_g production TCP + R o_rig) projects outside the image (no margin)
  out_of_view               any arm with exit_s > 0 or outproj_s >= MIN_GAP_S
Then the in-view episodes keep their GOLD / SILVER / REJECT class; folders of hard links to the existing renders:
  <out>/{GOLD,SILVER,REJECT,OUT_OF_VIEW}/<take>_dNN.mp4  + episode_quality_inview.csv
    python out_of_view.py <quality.csv> <renders dir> <out dir> <pack> [<pack> ...]"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]; sys.path[:0] = [str(ROOT)]
import core as C  # noqa
import camcal  # noqa
FD = C.FD; EDGE, MIN_GAP_S, W_, H_ = 40, 0.5, 848, 480
T_C_B = np.array(json.loads((FD.EGO / "configs/calibration/robot_cam_v001.json").read_text())["T_C_B"])


def proj(P, I):
    X = P @ T_C_B[:3, :3].T + T_C_B[:3, 3]; return np.stack([I["fx"] * X[:, 0] / X[:, 2] + I["cx"], I["fy"] * X[:, 1] / X[:, 2] + I["cy"]], 1)


def edge_of(u, v):
    return "top" if v < EDGE else "bottom" if v > H_ - EDGE else "left" if u < EDGE else "right" if u > W_ - EDGE else ""


OFFS = {}


def arm_audit(d, take, W, arm, I):
    z = np.load(d / f"solve3_{arm}.npz"); t = z["t"].astype(np.int64); w = (t >= W["w0"]) & (t <= W["w1"]); v = z["conf"] > 0; c = z["c"]
    idx = np.flatnonzero(w); x = np.r_[0, (~v[idx]).astype(int), 0]; n, tot, edges = 0, 0.0, []
    for s, e in zip(np.flatnonzero(np.diff(x) == 1), np.flatnonzero(np.diff(x) == -1)):
        L = (t[idx[e - 1]] - t[idx[s]]) / 1e9
        if L < MIN_GAP_S: continue
        prev = np.flatnonzero(v[:idx[s]])
        if not len(prev): continue
        u, vv = proj(c[prev[-1]][None], I)[0]; ed = edge_of(u, vv)
        if ed: n += 1; tot += L; edges.append((L, ed))
    f = ROOT / f"stitch/qa/fus_Cg_g/{take}_{arm}_d{W['demo']:02d}_h-1.npz"; out_s = 0.0
    if f.exists():
        r = np.load(f); m = (r["t_ns"] >= W["w0"]) & (r["t_ns"] <= W["w1"])
        if m.sum() > 1:
            body = r["p_B"][m] + np.einsum("nij,j->ni", r["R_B_tcp"][m], OFFS[arm])          # rig body = what the Orbbec sees
            uv = proj(body, I); o = (uv[:, 0] < 0) | (uv[:, 0] > W_) | (uv[:, 1] < 0) | (uv[:, 1] > H_)
            out_s = float(o.sum() * np.median(np.diff(r["t_ns"][m])) / 1e9)
    return dict(exit_n=n, exit_s=round(tot, 2), exit_edge=max(edges)[1] if edges else "", outproj_s=round(out_s, 2))


def main(qcsv, renders, out, packs):
    q = pd.read_csv(qcsv, dtype={"take": str}); rows = []
    loc = {}
    for p in packs:
        for d in Path(p).iterdir():
            if (d / "solve3_left.npz").exists(): loc[d.name] = d
    for _, r in q.iterrows():
        global T_C_B; T_C_B = camcal.T_C_B(r["take"])
        key = f"{r['take']}_d{int(r['demo']):02d}"; d = loc[key]; W = {w["demo"]: w for w in FD.trims()[r["take"]]}[int(r["demo"])]
        I = json.loads((FD.TAG / r["take"] / "meta.json").read_text())["intrinsics"]; row = dict(r)
        asg = json.loads((Path.home() / "laundry_dit_training/data/tag_rig_derived" / r["take"] / "assignment.json").read_text())
        off = json.loads((HERE.parent / "body_offset_by_rig_v1.json").read_text())["offsets_by_rig"]
        for a in ("left", "right"): OFFS[a] = np.array(off[asg["rig_of_arm"][a]]) / 1000
        for a in ("left", "right"):
            for k, v in arm_audit(d, r["take"], W, a, I).items(): row[f"{a[0].upper()}_{k}"] = v
        row["out_of_view"] = bool(any(row[f"{A}_exit_s"] > 0 or row[f"{A}_outproj_s"] >= MIN_GAP_S for A in "LR")); rows.append(row)
    df = pd.DataFrame(rows); df.to_csv(Path(out) / "episode_quality_inview.csv", index=False)
    for c in ("GOLD", "SILVER", "REJECT", "OUT_OF_VIEW"): (Path(out) / c).mkdir(parents=True, exist_ok=True)
    for _, r in df.iterrows():
        key = f"{r['take']}_d{int(r['demo']):02d}.mp4"; src = Path(renders) / r["class"] / key; dst = Path(out) / ("OUT_OF_VIEW" if r["out_of_view"] else r["class"]) / key
        if src.exists() and not dst.exists(): os.link(src, dst)
    for c in ("GOLD", "SILVER", "REJECT"):
        df[(~df["out_of_view"]) & (df["class"] == c)].to_csv(Path(out) / c / f"{c.lower()}_list.csv", index=False)
    df[df["out_of_view"]].to_csv(Path(out) / "OUT_OF_VIEW" / "out_of_view_list.csv", index=False)
    x = df[df["in_130"]]
    print("all:", df.groupby(["out_of_view", "class"]).size().unstack(fill_value=0).to_string())
    print("in_130:", x.groupby(["out_of_view", "class"]).size().unstack(fill_value=0).to_string())
    o = df[df["out_of_view"]]; ed = pd.concat([o["L_exit_edge"], o["R_exit_edge"]]); print("exit edges:", ed[ed != ""].value_counts().to_dict())
    print("out-of-view arms: L", int((o["L_exit_s"] > 0).sum() + 0), "R", int((o["R_exit_s"] > 0).sum()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:])
