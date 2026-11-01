"""2026-10-26 re-register the fixed Orbbec after it was MOVED, without the robot: T_B_C from the fixed table tags 400 / 401
(configs/calibration/start_anchors_v001.json: tag centres + rotations in the robot base, 6 cm, measured 10-21 with robot_cam_v001)
plus the table plane from depth.  Joint least squares over T_C_B (6 DoF):
  - reprojection of the 4 corners of every visible anchor tag in every selected frame (sigma PX_SIG px)
  - table depth points (RANSAC plane inliers of the first selected frame, subsampled): base z = TABLE_Z (sigma Z_SIG m)
Use a STATIC stretch of the take (camera untouched, table empty enough that both tags are seen).  Prints the residuals, the change vs
robot_cam_v001, how high above the table the image top edge reaches at the workspace centre, and writes --out (same keys as v001).
    .venv/bin/python scripts/pose/robot_cam_from_anchors.py datasets/tag_rig/<take> --frames 2400:2828:4 [--out configs/calibration/robot_cam_v002.json]"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[2]
PX_SIG, Z_SIG, TABLE_Z = 0.7, 0.002, 0.001


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("take"); ap.add_argument("--frames", required=True); ap.add_argument("--out", default=""); ap.add_argument("--meta", default="", help="meta.json with the intrinsics (a take still recording has none yet)")
    ap.add_argument("--anchors", default=str(ROOT / "configs/calibration/start_anchors_v001.json")); a = ap.parse_args()
    E = Path(a.take); I = json.loads(Path(a.meta or E / "meta.json").read_text())["intrinsics"]; K = np.array([[I["fx"], 0, I["cx"]], [0, I["fy"], I["cy"]], [0, 0, 1.0]])
    A = json.loads(Path(a.anchors).read_text()); h = A["tag_size_m"] / 2; obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)
    W = {int(k): (np.array(v["R_B"]), np.array(v["p_B"])) for k, v in A["tags"].items()}
    prm = cv2.aruco.DetectorParameters(); prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), prm)
    s0, s1, st = (int(x) for x in a.frames.split(":")); P3, P2, seen = [], [], {k: 0 for k in W}
    for k in range(s0, s1, st):
        g = cv2.imread(str(E / f"color_{k:05d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if g is None: continue
        c, ids, _ = det.detectMarkers(g)
        if ids is None: continue
        for cc, i in zip(c, ids.ravel()):
            if int(i) in W:
                RB, pB = W[int(i)]; P3.append(obj @ RB.T + pB); P2.append(cc.reshape(4, 2)); seen[int(i)] += 1
    if not P3: sys.exit(f"anchor tags seen {seen} -- none")
    if min(seen.values()) == 0: print(f"WARNING only one anchor tag seen {seen}: yaw / x-y rest on a single small tag (provisional)")
    P3, P2 = np.concatenate(P3), np.concatenate(P2).astype(float)
    ok, rv, tv = cv2.solvePnP(P3, P2, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    # table plane points (camera frame) from the first selected depth frame: RANSAC plane, inliers subsampled
    d = cv2.imread(str(E / f"depth_{s0:05d}.png"), -1).astype(np.float32) / 1000; H_, W_ = d.shape; vv, uu = np.mgrid[0:H_:4, 0:W_:4]; z = d[::4, ::4]
    X = np.stack([(uu - I["cx"]) / I["fx"] * z, (vv - I["cy"]) / I["fy"] * z, z], -1).reshape(-1, 3); X = X[(X[:, 2] > 0.3) & (X[:, 2] < 2.5)]
    Rc = R.from_rotvec(rv.ravel()).as_matrix(); XB = (X - tv.ravel()) @ Rc            # camera -> base with the PnP init
    near = np.abs(XB[:, 2] - TABLE_Z) < 0.03; rng = np.random.default_rng(0); best = None
    for _ in range(300):
        s = X[near][rng.choice(near.sum(), 3, replace=False)]; n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-9: continue
        n /= np.linalg.norm(n); inl = np.abs((X - s[0]) @ n) < 0.006
        if best is None or inl.sum() > best.sum(): best = inl
    T = X[best]; T = T[rng.choice(len(T), min(3000, len(T)), replace=False)]

    def res(x):
        Rm = R.from_rotvec(x[:3]).as_matrix(); t = x[3:]
        pc = P3 @ Rm.T + t; pr = pc[:, :2] / pc[:, 2:3] * [I["fx"], I["fy"]] + [I["cx"], I["cy"]]
        zb = ((T - t) @ Rm)[:, 2] - TABLE_Z
        return np.r_[((pr - P2) / PX_SIG).ravel(), zb / Z_SIG]
    x0 = np.r_[rv.ravel(), tv.ravel()]; sol = least_squares(res, x0, loss="huber", f_scale=2.0)
    Rm = R.from_rotvec(sol.x[:3]).as_matrix(); t = sol.x[3:]; T_C_B = np.eye(4); T_C_B[:3, :3] = Rm; T_C_B[:3, 3] = t; T_B_C = np.linalg.inv(T_C_B)
    r = res(sol.x); n2 = len(P2) * 2; rp = np.abs(r[:n2]) * PX_SIG; rz = np.abs(r[n2:]) * Z_SIG * 1000
    old = np.array(json.loads((ROOT / "configs/calibration/robot_cam_v001.json").read_text())["T_B_C"])
    dR = np.degrees(R.from_matrix(old[:3, :3].T @ T_B_C[:3, :3]).magnitude()); dt = np.linalg.norm(old[:3, 3] - T_B_C[:3, 3]) * 1000
    print(f"anchor detections {seen}, {len(P2) // 4} tag views; table points {len(T)}")
    print(f"reprojection |px| p50/p95 {np.median(rp):.2f}/{np.percentile(rp, 95):.2f}; table plane |z| p50/p95 {np.median(rz):.1f}/{np.percentile(rz, 95):.1f} mm")
    print(f"camera in base: {np.round(T_B_C[:3, 3], 3)} m (v001 {np.round(old[:3, 3], 3)}), moved {dt:.0f} mm, rotated {dR:.1f} deg")
    for x_, y_ in ((0.40, 0.0), (0.35, 0.25), (0.35, -0.25)):        # how high above the table the image TOP edge reaches over a workspace point
        hs = np.linspace(0, 1.5, 301); pts = np.stack([np.full_like(hs, x_), np.full_like(hs, y_), hs], 1) @ T_C_B[:3, :3].T + T_C_B[:3, 3]
        v = pts[:, 1] / pts[:, 2] * I["fy"] + I["cy"]; u = pts[:, 0] / pts[:, 2] * I["fx"] + I["cx"]; inside = (v >= 0) & (v < 480) & (u >= 0) & (u < 848)
        top = hs[inside].max() if inside.any() else float("nan"); print(f"  above base point ({x_:.2f}, {y_:+.2f}): visible up to z = {top:.2f} m")
    if a.out:
        Path(a.out).write_text(json.dumps(dict(T_B_C=T_B_C.tolist(), T_C_B=T_C_B.tolist(), method="fixed tags 400/401 (start_anchors_v001) + table depth plane",
                                               take=str(E), frames=a.frames, reproj_px_p50=float(np.median(rp)), plane_mm_p50=float(np.median(rz))), indent=1))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
