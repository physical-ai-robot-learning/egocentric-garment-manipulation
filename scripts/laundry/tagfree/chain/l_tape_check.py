"""11-12: L-tape positions (robot base) from mid-demo frames (hands away) of a take, with a given T_B_C json -> camera-drift check."""
import json, sys
from pathlib import Path
import cv2, numpy as np
HERE = Path.home() / "ego_collector/datasets/m3_runs/_imu_consistency/orbbec/v2b"; sys.path[:0] = [str(HERE), str(HERE.parent.parent)]
import tagfree_seed as S  # noqa
take, calf = sys.argv[1], sys.argv[2]
tk = S.TAG / take; m = json.loads((tk / "meta.json").read_text()); I = m["intrinsics"]; t = np.load(tk / "t_ns.npy")
T_B_C = np.array(json.loads(Path(calf).read_text())["T_B_C"])
t0 = int(m["t_start_monotonic_ns"]); D = [p for p in m["protocol"] if p["text"].startswith("자유:")]
ks = []
for p in D[:6]:
    a, b = t0 + int((p["t0"] + 0.3 * p["dur"]) * 1e9), t0 + int((p["t0"] + 0.7 * p["dur"]) * 1e9)
    ks += list(np.flatnonzero((t >= a) & (t <= b))[::8])
G = np.median([cv2.imread(str(tk / f"color_{k:05d}.jpg"), 0) for k in ks], 0).astype(np.uint8)
Ds = np.stack([cv2.imread(str(tk / f"depth_{k:05d}.png"), -1).astype(float) for k in ks]); Ds[Ds == 0] = np.nan; Dm = np.nan_to_num(np.nanmedian(Ds, 0)).astype(np.uint16)
P, ok = S.lift(Dm, I, T_B_C); n, c0 = S.plane(P, ok); h = (P - c0) @ n; h[~ok] = np.nan
on = ((G < S.DARK) & (np.abs(np.nan_to_num(h, nan=9)) < S.ON_H)).astype(np.uint8)
nc, lab, st, cen = cv2.connectedComponentsWithStats(on); out = {}
for s in ("left", "right"):
    best = None
    for i in range(1, nc):
        if not (S.L_AREA[0] <= st[i][4] <= S.L_AREA[1]) or max(st[i][2], st[i][3]) > 80: continue
        pix = np.argwhere(lab == i); Q = P[pix[:, 0], pix[:, 1]]; Q = Q[np.isfinite(Q[:, 0])]
        if len(Q) < 10 or not 0.025 <= np.ptp(Q[:, :2], 0).max() <= 0.08: continue
        d = np.linalg.norm(Q.mean(0)[:2] - S.PREV[s])
        if d < 0.08 and (best is None or d < best[0]): best = (d, Q.mean(0), cen[i])
    out[s] = None if best is None else dict(xyz_mm=(best[1] * 1000).round(1).tolist(), px=best[2].round(1).tolist())
print(json.dumps(dict(take=take, cal=Path(calf).name, frames=len(ks), plane_n=n.round(4).tolist(), plane_z_mm=round(float(c0 @ n / n[2] * 1000) if abs(n[2]) > 0.5 else float('nan'), 1), L=out)))
