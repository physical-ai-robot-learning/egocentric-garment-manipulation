"""[2026-10-23] Input / output contract of the tag-rig laundry policies on the bimanual reBot B601-DM (deploy stage).

Contract (laundry_dataset tag_rig_to_lerobot, laundry_training, checkpoint act_tagrig_v1_1m):
  images   observation.images.{global, left_wrist, right_wrist}, RGB float [0, 1] CHW, 384x216
           global = Orbbec colour 848x480 (OrbbecCapture, JPEG q95) -> INTER_AREA 384x216 (slight 1.767 -> 1.778 stretch, as trained)
           wrist  = Arducam fisheye 1920x1080 -> INTER_AREA 1280x720 (JPEG q92, umi_jaw record) -> INTER_AREA 384x216
  state    30D = proprio_rows_lr: [inv(T_L0) T_L | inv(T_R0) T_R | gL | gR | inv(T_L) T_R | 1]; T_X0 = pose at the EPISODE start
           (deploy: the measured pose of the first cycle), g binary 1 = open (deploy: the last commanded binary jaw)
  action   (32, 20) rows k = 1..32: [inv(T(t)) T(t+k) L pose9 | R pose9 | gL | gR], rot6d = first two ROWS
  frame    T = FK(q) of <side>_tcp in the reBot base B (PinkIkSolver: URDF + tcp/joint/base overrides) = the robotize_ik label frame
           (R_B_E = R_B_tcp @ R_U_E).  Left arm = base +y.
Policy loading / pre / post-processing = core.checkpoint.loadPolicy (the checkpoint's own processors).
"""

from __future__ import annotations

from laundry_common import paths as egopaths

TRAIN_ROOT = egopaths.TRAIN_ROOT  # was LDT: the training workspace (checkpoints, reports, LeRobot sets)
ACT_CKPT = TRAIN_ROOT / "checkpoints_pod/act_tagrig_v1_1m/checkpoints/0500000"
DIT_CKPT = TRAIN_ROOT / "checkpoints_pod/dit_tagrig_v1/checkpoints/last"
IMG_WH = (384, 216)  # policy image size (w, h)
WRIST_MID_WH = (1280, 720)  # umi_jaw record frame size the wrist path passes through
WRIST_JPEG_QUALITY = 92  # umi_jaw record JPEG quality (take files)
CAMS = ("global", "left_wrist", "right_wrist")
TASK = "laundry folding"
SIDES = ("left", "right")
POSE_SLICE = {"left": slice(0, 9), "right": slice(9, 18)}  # pose9 columns of each arm in a chunk row / state
GRIP_INDEX = {"left": 18, "right": 19}  # jaw column of each arm
GRIP_COLUMNS = slice(18, 20)
CHUNK_ROWS = 32  # rows k = 1..32 of a chunk
ACTION_DIM = 20
ROW_DT_S = 0.05  # training row period (20 Hz)
NS_PER_S = 1e9
