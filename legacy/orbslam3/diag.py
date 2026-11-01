import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path.home() / "ego_collector/scripts"))
from umi_fuse_segments import load_segments, sim3_resid
import umi_joint_handeye as JH
from scipy.spatial.transform import Rotation as R
take, man, frames, hand = sys.argv[1:5]
z = np.load(Path(take) / f"umi_traj_{hand}.npz"); tT = z["t_ns"].astype(np.int64); tag = z["ntags"] > 0; tt0, pb0, Rb0 = tT[tag], z["p_B"][tag], R.from_matrix(z["R_B_tcp"][tag])
dt = int(20e6)
for sg in load_segments(man, str(Path(frames) / "frames.json"), Path(man).parent):
    ia = np.where((tt0 + dt >= sg["t"][0]) & (tt0 + dt <= sg["t"][-1]))[0]
    if len(ia) < 10: print(sg["start"], len(sg["t"]), "anchors", len(ia)); continue
    pc, Rc = JH.seg_obs(sg, tt0[ia], dt); r, s = sim3_resid(pc, pb0[ia])
    # rotation consistency: relative rotation magnitudes SLAM vs tag
    rs = np.degrees(np.linalg.norm((Rc[0].inv() * Rc).as_rotvec(), axis=1)); rb = np.degrees(np.linalg.norm((Rb0[ia[0]].inv() * Rb0[ia]).as_rotvec(), axis=1))
    print(sg["start"], len(sg["t"]), "anchors", len(ia), "res_med_mm", round(np.median(r) * 1000, 1), "scale", round(s, 3), "rot_slam/tag max", round(rs.max()), round(rb.max()), "corr", round(np.corrcoef(rs, rb)[0, 1], 2), "travel", round(np.ptp(pb0[ia], 0).max(), 2))
