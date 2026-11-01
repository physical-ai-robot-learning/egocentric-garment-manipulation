"""[2026-10-23] Shared core of the tag-rig ACT real-robot deploy (deploy/act_tagrig_deploy.py) and its parity test
(deploy/act_tagrig_parity.py).  Everything that decides WHAT the policy sees and HOW its output becomes a robot target lives here,
so the parity test exercises the exact deploy code path.

Contract (datasets/tag_rig_to_lerobot.py, training/tagrig_common.py, checkpoint act_tagrig_v1_1m):
  images   observation.images.{global, left_wrist, right_wrist}, RGB float [0, 1] CHW, 384x216
           global = Orbbec colour 848x480 (OrbbecCapture, JPEG q95)  -> INTER_AREA 384x216 (slight 1.767 -> 1.778 stretch, as trained)
           wrist  = Arducam fisheye 1920x1080 -> INTER_AREA 1280x720 (JPEG q92, umi_jaw record) -> INTER_AREA 384x216
  state    30D = proprio_rows_lr: [inv(T_L0) T_L | inv(T_R0) T_R | gL | gR | inv(T_L) T_R | 1]; T_X0 = pose at the EPISODE start
           (deploy: the measured pose of the first cycle), g binary 1 = open (deploy: the last commanded binary jaw)
  action   (32, 20) rows k = 1..32: [inv(T(t)) T(t+k) L pose9 | R pose9 | gL | gR], rot6d = first two ROWS
  frame    T = FK(q) of <side>_tcp in the reBot base B (PinkIkSolver: URDF + tcp/joint/base overrides) = the robotize_ik label frame
           (R_B_E = R_B_tcp @ R_U_E).  Left arm = base +y.
Policy loading / pre / post-processing = evaluation/tagrig_offline_eval.load (the checkpoint's own processors).

[2026-11-01] The code is in tagrig_core/: policy.py (images, state, decode, RTC weights, policy runners) and geometry.py (arms, FK,
capsule collision, camera model, plot). This module stays the one the deploy code imports as `core`; it re-exports every name.
"""
from __future__ import annotations

from egogarment import paths as egopaths  # noqa: F401
from contracts.action_contract import T_to_pose9, inv_T, make_T, pose9_to_T  # noqa: F401  (core.inv_T, core.T_to_pose9, ...)
from garment_pipeline.eval_deploy.tagrig_core.policy import (
    LDT, CKPT, IMG_WH, CAMS, TASK, SIDES, SL, GI, DT, prep_global, prep_wrist, apply_align, make_item, make_item2,
    build_state, decode, rtc_weights, PolicyRunner, XvlaRunner, retime, LabelRunner, ActRunner, DIT_CKPT)
from garment_pipeline.eval_deploy.tagrig_core.geometry import (
    ROBOT_CAM, ROBOT_CAM_EXT, ROBOTIZE_ARGS, make_arms, fk, CAPSULES, FINGER_R, TABLE_R_WRIST, Collision, seg_dist,
    link_gap, rot_interp, rot_angle_deg, project, robot_cam, cv_plot)
