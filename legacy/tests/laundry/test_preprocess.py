import json

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from laundry_collector.preprocess import slam
from laundry_collector.preprocess.renderer import project
from laundry_collector.preprocess.virtual_gripper import GripperConfig, run
from laundry_collector.preprocess.world_align import FixedScale, HandToWorldTransformer, WorldAligner, inv_se3, pose_mat, relative_actions
from tests.laundry.synth import hand_local, place, pose


def test_slam_stage_and_collect(recorded_episode, tmp_path):
    d = slam.stage(recorded_episode, "left", tmp_path / "stage")
    assert (d / "raw_video.mp4").exists() and "KannalaBrandt8" in (d / "orbslam_setting.yaml").read_text()
    fm = pd.read_csv(d / "frame_map.csv"); n = len(fm)
    # fake runner output in the 4090 format
    rows = [dict(frame_idx=i, timestamp=i / 30, state="INIT" if i == 0 else "TRACKING", is_lost="false", is_keyframe="false",
                 x=0.01 * i, y=0, z=0, q_x=0, q_y=0, q_z=0, q_w=1, map_updated=0, kf_ref=0, sim3_scale=1.0, match_frac=0.9) for i in range(n)]
    pd.DataFrame(rows).to_csv(d / "ss1.csv", index=False)
    out = slam.collect(recorded_episode, "left", tmp_path / "stage", pull=False)
    a = np.load(out); assert a.shape == (n, 11) and a[:, 2].all()


def test_world_alignment_recovers_metric_world():
    rng = np.random.default_rng(1); n = 60; s_true = 0.37
    T_wc_true = pose_mat(np.column_stack([np.linspace(0.3, 0.5, n), np.full(n, 0.2), np.full(n, 0.25)]),
                         R.from_rotvec(rng.normal(0, 0.2, (n, 3))).as_quat())
    T_ct = pose((0.1, 0.2, 0), (0.0, 0.05, 0.12))
    init = np.zeros(n, bool); init[:10] = True
    T_wc_true[:10] = T_wc_true[5]                                      # still during the hold
    T_w_tcp0 = T_wc_true[5] @ T_ct
    G = pose((0.3, -1.0, 0.5), (1.0, -2.0, 0.5))                        # arbitrary SLAM world
    T_slam = np.einsum("ij,njk->nik", G, T_wc_true); T_slam[:, :3, 3] /= s_true   # non-metric
    world = dict(init_markers=dict(X_L=dict(position=T_w_tcp0[:3, 3].tolist(), R_world_tcp=T_w_tcp0[:3, :3].tolist())))
    # the SLAM world's own scale also scales G's translation; express G in SLAM units consistently
    T_slam = np.einsum("ij,njk->nik", np.diag([1, 1, 1, 1.0]), T_slam)
    res = WorldAligner(world, {"left": T_ct}, FixedScale(s_true)).align("left", T_slam, np.ones(n, bool), init)
    rel_true = np.einsum("ij,njk->nik", inv_se3(T_wc_true[5]), T_wc_true); rel = np.einsum("ij,njk->nik", inv_se3(res.T_world_camera[5]), res.T_world_camera)
    assert np.allclose(rel[:, :3, 3], rel_true[:, :3, 3], atol=1e-9) and np.allclose(res.T_world_camera @ T_ct, np.einsum("nij,jk->nik", res.T_world_camera, T_ct))
    assert np.allclose(res.T_world_camera[5] @ T_ct, T_w_tcp0, atol=1e-9)
    W = HandToWorldTransformer.transform(res.T_world_camera, np.broadcast_to(T_ct, (n, 4, 4)))
    A = relative_actions(W, ks=[1, 2]); assert A.shape == (n - 2, 2, 4, 4) and np.allclose(A[0, 0], inv_se3(W[0]) @ W[1])


def test_virtual_gripper_render_and_labels_share_one_pose(recorded_episode, tmp_path):
    ep = recorded_episode; n = len(pd.read_csv(ep / "right_wrist" / "timestamps.csv"))
    t = np.arange(n) / 30.0
    kp = np.stack([place(hand_local(0.02 + 0.05 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * ti)), "right"),
                         pose((0.3 * np.sin(ti), 0.2, 0.1 * ti), (0.01 * np.sin(ti), -0.06, 0.05))) for ti in t])
    kp[20:25] = np.nan                                                   # a hand-pose dropout
    np.savez(tmp_path / "kp.npz", keypoints_3d=kp, confidence=np.ones((n, 21)))
    rep = run(ep, "right", tmp_path / "kp.npz", GripperConfig(pose_mode="hand"), debug=True)
    out = ep / "processed" / "right"
    for f in ("wrist_raw.mp4", "wrist_virtual_overlay.mp4", "wrist_debug.mp4", "gripper_pose.csv", "gripper_pose.npy", "T_camera_gripper.npy", "gripper_qa.json"):
        assert (out / f).exists(), f
    T = np.load(out / "T_camera_gripper.npy"); df = pd.read_csv(out / "gripper_pose.csv")
    assert np.isfinite(T).all() and len(T) == n and (~df.hand_pose_valid[20:25]).all()
    assert np.allclose(df[["x", "y", "z"]].to_numpy(), T[:, :3, 3])     # labels == the rendered pose, same array
    intr = json.loads((ep / "calibration" / "right_intrinsics.json").read_text())
    i = 60; mid = 0.5 * (kp[i, 4] + kp[i, 8])
    assert np.linalg.norm(project(T[i, :3, 3][None], intr) - project(mid[None], intr)) < 3.0   # TCP on the pinch (smoothed)
    assert rep["hand_pose_valid"] > 0.9 and rep["frames_total"] == n


def test_camera_fixed_mode_constant_pose_and_pinch_opening(recorded_episode, tmp_path):
    ep = recorded_episode; n = len(pd.read_csv(ep / "left_wrist" / "timestamps.csv"))
    aps = 0.02 + 0.06 * (np.arange(n) % 60) / 60
    kp = np.stack([place(hand_local(a, "left"), pose((0.2, 0.1, 0.0), (0.0, -0.05, 0.08))) for a in aps])
    np.savez(tmp_path / "kp.npz", keypoints_3d=kp, confidence=np.ones((n, 21)))
    cfg = GripperConfig(view="fisheye")                                 # camera_fixed on the raw fisheye frame
    run(ep, "left", tmp_path / "kp.npz", cfg)
    T = np.load(ep / "processed" / "left" / "T_camera_gripper.npy"); df = pd.read_csv(ep / "processed" / "left" / "gripper_pose.csv")
    assert np.allclose(T, cfg.mount_T()) and np.allclose(df[["x", "y", "z"]].to_numpy(), T[:, :3, 3])
    assert df.jaw_width_m.max() - df.jaw_width_m.min() > 0.03         # opening follows the pinch


def test_robot_view_mode_uses_hand_eye_tcp_and_writes_robot_view(recorded_episode, tmp_path):
    from laundry_collector.preprocess.robot_view import T_camera_tcp
    ep = recorded_episode; n = len(pd.read_csv(ep / "left_wrist" / "timestamps.csv"))
    kp = np.stack([place(hand_local(0.04, "left"), pose((0.2, 0.1, 0.0), (0.0, -0.05, 0.08)))] * n)
    np.savez(tmp_path / "kp.npz", keypoints_3d=kp, confidence=np.ones((n, 21)))
    run(ep, "left", tmp_path / "kp.npz", GripperConfig())                # default: camera_fixed + robot_c922
    out = ep / "processed" / "left"
    assert np.allclose(np.load(out / "T_camera_gripper.npy"), T_camera_tcp())
    import av
    with av.open(str(out / "wrist_robotview_gripper.mp4")) as c:
        s = c.streams.video[0]; assert (s.codec_context.width, s.codec_context.height) == (640, 480)
