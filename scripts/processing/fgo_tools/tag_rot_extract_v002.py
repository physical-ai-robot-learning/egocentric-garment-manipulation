"""10-24 copy of tag_rot_extract.py using the v002 rig config (adds tags 220 / 221; for takes recorded with them, e.g. 20261023_155004). 10-23 FGO: raw per-frame AprilTag rig rotation (PnP, no IMU) in the robot base for both rigs of a take -> _fgo_1023/tagrot/<take>.npz
(R_B_tcp_tag_<rig>, ntags_<rig>, rmse_<rig>; NaN where not seen). Used as the rotation reference / rotation factors."""
import json, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path.home() / "ego_collector"; sys.path.insert(0, str(ROOT / "scripts"))
from tag_rig_pose import rig_pose  # noqa
take = Path(sys.argv[1]); meta = json.loads((take / "meta.json").read_text()); I = meta["intrinsics"]
K = np.array([[I["fx"], 0, I["cx"]], [0, I["fy"], I["cy"]], [0, 0, 1.0]]); cfg = json.loads((ROOT / "configs/calibration/umi_tag_rig_v002.json").read_text())
sys.path.insert(0, str(ROOT / "datasets/m3_runs/_imu_consistency")); import camcal  # noqa
TBC = camcal.T_B_C(take.name)                                     # 10-26: per-take camera calibration
rigs = {"right": "UMI 오른손", "left": "UMI 왼손"}; Trt = {s: {int(k): np.array(v) for k, v in cfg[n]["T_rig_tag"].items()} for s, n in rigs.items()}
prm = cv2.aruco.DetectorParameters(); prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11), prm)
tc = np.load(take / "t_ns.npy"); n = len(tc); out = dict(t_ns=tc)
for s in rigs: out[f"R_{s}"] = np.full((n, 3, 3), np.nan); out[f"n_{s}"] = np.zeros(n, int); out[f"rmse_{s}"] = np.full(n, np.nan)
for k in range(n):
    im = cv2.imread(str(take / f"color_{k:05d}.jpg"), cv2.IMREAD_GRAYSCALE)
    if im is None: continue
    cs, ids, _ = det.detectMarkers(im)
    for s in rigs:
        r = rig_pose(cs, ids, Trt[s], K, 0.048)
        if r is None or r["rmse_px"] > 2.0: continue
        out[f"R_{s}"][k] = TBC[:3, :3] @ r["T"][:3, :3]; out[f"n_{s}"][k] = len(r["tags"]); out[f"rmse_{s}"][k] = r["rmse_px"]
np.savez(Path.home() / f"ego_collector/datasets/m3_runs/_fgo_1023/tagrot/{take.name}.npz", **out); print(take.name, n, {s: int((out[f'n_{s}'] > 0).sum()) for s in rigs})
