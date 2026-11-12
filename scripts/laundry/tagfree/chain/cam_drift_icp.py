"""11-12: Orbbec camera motion between two takes from the static depth scene: median depth of mid-demo frames (hands / cloth move,
the room does not) -> camera-frame clouds -> trimmed point-to-plane ICP (own implementation; open3d segfaults on this Mac).
Output T (4x4) with X_ref_cam = T @ X_cur_cam, i.e. T_B_C(cur) = T_B_C(ref) @ T."""
import json, sys
from pathlib import Path
import cv2, numpy as np
from scipy.spatial import cKDTree
TAG = Path.home() / "ego_collector/datasets/tag_rig"


def cloud(take, n_demo=8, step=6):
    tk = TAG / take; m = json.loads((tk / "meta.json").read_text()); I = m["intrinsics"]; t = np.load(tk / "t_ns.npy")
    t0 = int(m["t_start_monotonic_ns"]); ks = []
    for p in [p for p in m["protocol"] if p["text"].startswith("자유:")][:n_demo]:
        if p["dur"] > 600: continue
        ks += list(np.flatnonzero((t >= t0 + (p["t0"] + 0.2 * p["dur"]) * 1e9) & (t <= t0 + (p["t0"] + 0.8 * p["dur"]) * 1e9))[::step])
    D = np.stack([cv2.imread(str(tk / f"depth_{k:05d}.png"), -1).astype(float) for k in ks]); D[D == 0] = np.nan
    med = np.nanmedian(D, 0); spread = np.nanpercentile(D, 90, 0) - np.nanpercentile(D, 10, 0)
    h, w = med.shape; v, u = np.mgrid[0:h, 0:w]
    z = med / 1000.0
    ok = np.isfinite(z) & (z > 0.4) & (z < 3.5) & (spread < 15)          # static pixels only (moving cloth / hands vary > 15 mm)
    X = np.stack([(u - I["cx"]) * z / I["fx"], (v - I["cy"]) * z / I["fy"], z], -1)[ok]
    return X[::7], len(ks)


def normals(X, k=12):
    tr = cKDTree(X); _, nn = tr.query(X, k); Q = X[nn] - X[nn].mean(1, keepdims=True)
    _, _, vt = np.linalg.svd(Q, full_matrices=False); return vt[:, -1]


def icp(src, dst, T=np.eye(4), it=40, trim=0.7):
    nd = normals(dst); tr = cKDTree(dst)
    for _ in range(it):
        s = src @ T[:3, :3].T + T[:3, 3]; d, j = tr.query(s); keep = d < np.quantile(d, trim)
        p, q, n = s[keep], dst[j[keep]], nd[j[keep]]
        A = np.hstack([np.cross(p, n), n]); b = ((q - p) * n).sum(1)
        x = np.linalg.lstsq(A, b, rcond=None)[0]; R, _ = cv2.Rodrigues(x[:3]); dT = np.eye(4); dT[:3, :3] = R; dT[:3, 3] = x[3:]
        T = dT @ T
        if np.linalg.norm(x) < 1e-6: break
    s = src @ T[:3, :3].T + T[:3, 3]; d, _ = tr.query(s)
    return T, float(np.median(d) * 1000), float(np.mean(d < 0.01))


ref, cur = sys.argv[1], sys.argv[2]
Xr, nr = cloud(ref); Xc, nc = cloud(cur)
print(f"ref {ref} {len(Xr)} pts ({nr} frames) | cur {cur} {len(Xc)} pts ({nc} frames)")
T, med_mm, frac = icp(Xc, Xr)
rv, _ = cv2.Rodrigues(T[:3, :3])
print(json.dumps(dict(ref=ref, cur=cur, rot_deg=np.degrees(rv.ravel()).round(3).tolist(), trans_mm=(T[:3, 3] * 1000).round(1).tolist(),
                      median_resid_mm=round(med_mm, 2), inlier_1cm=round(frac, 3), T=T.round(6).tolist())))
