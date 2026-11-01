"""10-24 CoTracker3 benchmark pack v2: as make_pack.py (initial seeds at the first tracked frame) PLUS reserve candidate queries every
RES_S seconds per arm: dark above-table pixels within RES_R of that arm's v31speed centre at that frame (tag-free; may be wrong -- the
solver's probation decides), up to N_RES per arm per time, farthest-point spread.  query_kind 0 = initial seed, 1 = reserve.
--- v1 ---
10-24 CoTracker3 benchmark pack: per (take, demo) the Orbbec frames of the idle-trim window + per-arm seed queries at the first frame.
Seeds (tag-free): the v31speed tracker centre at the first tracked frame of the window (start-marker seeded), dark above-table pixels whose
3D point lies within SEED_R of it, up to N_SEED per arm, spread by farthest-point sampling.  Output <pack>/<take>_dNN/{frames/*.jpg, meta.json}.
    python make_pack.py <pack_dir> take:demo ..."""
import json, shutil, sys
from pathlib import Path
import cv2, numpy as np
HERE = Path(__file__).resolve().parent
from garment_pipeline.fusion_stack import core as C  # noqa
from garment_pipeline.fusion_stack import camcal  # noqa
FD = C.FD
SEED_R, N_SEED, DARK, H_MIN = 0.06, 40, 40, 0.02
RES_S, RES_R, N_RES = 1.0, 0.08, 20
TBC = np.array(json.loads((FD.EGO / "configs/calibration/robot_cam_v001.json").read_text())["T_B_C"]); TCB = np.linalg.inv(TBC)
pack = Path(sys.argv[1]); pack.mkdir(parents=True, exist_ok=True)
for spec in sys.argv[2:]:
    take, d = spec.split(":"); demo = int(d); TBC = camcal.T_B_C(take); TCB = np.linalg.inv(TBC); tk = FD.TAG / take; I = json.loads((tk / "meta.json").read_text())["intrinsics"]
    W = {x["demo"]: x for x in FD.trims()[take]}[demo]; z = np.load(HERE.parent / f"out/trackv31speed_{take}.npz"); t = z["t_ns"]; z_tab = float(z["z_table"])
    ks = np.flatnonzero((t >= W["w0"]) & (t <= W["w1"])); out = pack / f"{take}_d{demo:02d}"; (out / "frames").mkdir(parents=True, exist_ok=True)
    for i, k in enumerate(ks): shutil.copy(tk / f"color_{k:05d}.jpg", out / "frames" / f"{i:05d}.jpg")
    queries = []; seeds = {}; kinds = []; arms_q = []
    def pick(j, a, R, N):
        k = ks[j]; c = z["c_" + a][k]
        if not np.isfinite(c[0]): return np.zeros((0, 2))
        g = cv2.imread(str(tk / f"color_{k:05d}.jpg"), cv2.IMREAD_GRAYSCALE); dpt = cv2.imread(str(tk / f"depth_{k:05d}.png"), -1).astype(np.float32) / 1000
        vv, uu = np.mgrid[0:dpt.shape[0], 0:dpt.shape[1]]; X = np.stack([(uu - I["cx"]) / I["fx"] * dpt, (vv - I["cy"]) / I["fy"] * dpt, dpt], -1) @ TBC[:3, :3].T + TBC[:3, 3]
        m = (dpt > 0.2) & (g < DARK) & (X[..., 2] - z_tab > H_MIN) & (np.linalg.norm(X - c, axis=-1) < R)
        cand = np.argwhere(m)[:, ::-1].astype(float)
        if not len(cand): return cand
        sel = [int(np.argmin(np.linalg.norm(cand - cand.mean(0), axis=1)))]
        while len(sel) < min(N, len(cand)):
            dmin = np.min(np.linalg.norm(cand[:, None] - cand[sel][None], axis=2), axis=1); sel.append(int(np.argmax(dmin)))
        return cand[sel]
    for a in ("left", "right"):
        j0 = next(i for i, k in enumerate(ks) if np.isfinite(z["c_" + a][k, 0]))
        tres = t[ks[j0]] + np.arange(1, 1000) * RES_S * 1e9; tres = tres[tres < t[ks[-1]]]
        for tr_ in tres:
            j = int(np.argmin(np.abs(t[ks] - tr_))); q = pick(j, a, RES_R, N_RES)
            queries += [[float(j), float(x), float(y)] for x, y in q]; kinds += [1] * len(q); arms_q += [a] * len(q)
    for a in ("left", "right"):
        j = next(i for i, k in enumerate(ks) if np.isfinite(z["c_" + a][k, 0])); k = ks[j]; c = z["c_" + a][k]
        g = cv2.imread(str(tk / f"color_{k:05d}.jpg"), cv2.IMREAD_GRAYSCALE); dpt = cv2.imread(str(tk / f"depth_{k:05d}.png"), -1).astype(np.float32) / 1000
        vv, uu = np.mgrid[0:dpt.shape[0], 0:dpt.shape[1]]; X = np.stack([(uu - I["cx"]) / I["fx"] * dpt, (vv - I["cy"]) / I["fy"] * dpt, dpt], -1) @ TBC[:3, :3].T + TBC[:3, 3]
        m = (dpt > 0.2) & (g < DARK) & (X[..., 2] - z_tab > H_MIN) & (np.linalg.norm(X - c, axis=-1) < SEED_R)
        cand = np.argwhere(m)[:, ::-1].astype(float)                                   # (x, y)
        sel = [int(np.argmin(np.linalg.norm(cand - cand.mean(0), axis=1)))]
        while len(sel) < min(N_SEED, len(cand)):
            dmin = np.min(np.linalg.norm(cand[:, None] - cand[sel][None], axis=2), axis=1); sel.append(int(np.argmax(dmin)))
        q = cand[sel]; seeds[a] = dict(frame=int(j), k=int(k), n=len(q), centre=c.tolist())
        queries += [[float(j), float(x), float(y)] for x, y in q]; kinds += [0] * len(q); arms_q += [a] * len(q)
    (out / "meta.json").write_text(json.dumps(dict(take=take, demo=demo, ks=ks.tolist(), t_ns=t[ks].tolist(), seeds=seeds, queries=queries,
                                                   arm_of_query=arms_q, query_kind=kinds), indent=0))
    print(take, demo, len(ks), "frames", {a: seeds[a]["n"] for a in seeds}, "reserve", kinds.count(1))
