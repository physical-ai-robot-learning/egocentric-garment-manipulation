#!/usr/bin/env python3
"""[2026-11-05] Per-frame p_grasp side file for an aggregated tag-rig LeRobot set (FLAT250-PGRASP), no dataset rebuild.
Episode i of the set = key i of meta/split.json (aggregate order) -- VERIFIED per episode: the source time of frame 0 of both wrist
cams (take wrist t_ns[src_index]) must sit at the demo window start (grasp_label_v0.demo_window, keep_start) within TOL_S, and every
src_index must be inside that take's wrist recording.  Frame value = the latest 15 Hz p_grasp tick at or before that frame's wrist
source time (zero-order hold, causal; the policy sees the same wrist frame).
-> <set>/meta/pgrasp_<version>.npy  float32 (total_frames, 2) [L, R] by global index  + pgrasp_<version>.json
usage: grasp_pgrasp_sidefile.py SET_DIR PGRASP_DIR [--check-only]"""
import glob, json, pathlib, sys
import numpy as np, pandas as pd
sys.argv, ARGV = sys.argv[:1] + ["/tmp/_unused", "--none"], sys.argv
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import grasp_label_v0 as G  # noqa: E402
SET, PG = pathlib.Path(ARGV[1]).expanduser(), pathlib.Path(ARGV[2]).expanduser(); CHECK = "--check-only" in ARGV
TOL_S, CAMS = 0.25, (("L", "wrist_left", 1), ("R", "wrist", 2))       # src_index columns: (global, left_wrist, right_wrist)


def main():
    df = pd.concat([pd.read_parquet(f, columns=["index", "episode_index", "frame_index", "src_index"]) for f in sorted(glob.glob(str(SET / "data/*/*.parquet")))])
    df = df.sort_values("index"); info = json.loads((SET / "meta/info.json").read_text()); N = info["total_frames"]
    assert len(df) == N and (df["index"].values == np.arange(N)).all(), "global index not contiguous"
    sp = json.loads((SET / "meta/split.json").read_text())["episodes"]; keys = [k for v in sp.values() for k in v]
    eps = sorted(df["episode_index"].unique()); assert len(eps) == len(keys), (len(eps), len(keys))
    P = np.full((N, 2), np.nan, np.float32); version = None; worst = 0.0
    for e, k in zip(eps, keys):
        take, dn = k.split("/"); d = int(dn[4:]); t0, t1 = G.demo_window(take, d)
        sub = df[df["episode_index"] == e]; si = np.stack(sub["src_index"].values)
        assert (sub["frame_index"].values == np.arange(len(sub))).all(), f"{k}: frame_index not 0..n-1"
        z = None if CHECK else np.load(PG / "demos" / f"{take}_{dn}.npz")
        if z is not None: version = str(z["version"])
        for j, (h, cam, col) in enumerate(CAMS):
            tw = np.load(G.TR / take / cam / "t_ns.npy"); assert si[:, col].max() < len(tw), f"{k} {cam}: src_index beyond the recording"
            tf = tw[si[:, col]]; off = abs(tf[0] - t0) / 1e9; worst = max(worst, off)
            assert off <= TOL_S, f"{k} {cam}: frame 0 at {off:.2f} s from the demo window start -> episode/key order mismatch?"
            if z is not None:
                ts, p = z["t_ns"], z[f"{h}_p"]; i = np.searchsorted(ts, tf, side="right") - 1
                assert (i >= 0).all() and (tf <= ts[-1] + 1e9 / 15).all(), f"{k}: frames outside the p_grasp series"
                P[sub["index"].values, j] = p[i]
    print(f"{SET.name}: {len(eps)} episodes, {N} frames, order verified (worst frame-0 offset {worst:.3f} s)", flush=True)
    if CHECK: return
    assert not np.isnan(P).any()
    np.save(SET / f"meta/pgrasp_{version}.npy", P)
    (SET / f"meta/pgrasp_{version}.json").write_text(json.dumps(dict(version=version, shape=list(P.shape), columns=["L", "R"], source=str(PG),
        rule="value = latest 15 Hz p_grasp tick <= wrist source time of the frame (src_index)", mean=P.mean(0).round(3).tolist()), indent=1))
    print(f"  wrote meta/pgrasp_{version}.npy  mean L {P[:, 0].mean():.3f} R {P[:, 1].mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
