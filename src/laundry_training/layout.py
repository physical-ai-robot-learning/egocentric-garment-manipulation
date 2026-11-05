"""Index layout of the contract action / proprio vectors as the tag-rig trainers use them (training stage).

action (20)  [L xyz | L rot6d | R xyz | R rot6d | gL | gR]  (laundry_shared.contracts.action_contract)
state  (30)  [L pose9 | R pose9 | gL gR | T_LR pose9 | T_LR valid]  (laundry_shared.contracts.observation_contract)
"""
from __future__ import annotations

ACTION_DIM = 20
GRIP_A = [18, 19]                    # gripper dims of the action: absolute, never standardised
STATE_UNSCALED = [18, 19, 29]        # gripper dims + T_LR_valid of the state: never standardised
GRIP_STATE_DIMS = (18, 19)           # jaw state hidden by GRIP_STATE_MASK
# loss / metric groups of the action dims
GROUPS = {"pos": [0, 1, 2, 9, 10, 11], "rot": list(range(3, 9)) + list(range(12, 18)), "grip": [18, 19]}
# per-arm slices of the action: position, rotation; and the anchor_ok columns [pos_L, rot_L, pos_R, rot_R] that gate them
POSE9_L, POSE9_R, GRIPS = slice(0, 9), slice(9, 18), slice(18, 20)   # per-arm pose9 and both grippers
POS_L, ROT_L, POS_R, ROT_R = slice(0, 3), slice(3, 9), slice(9, 12), slice(12, 18)
ANCHOR_POS_L, ANCHOR_ROT_L, ANCHOR_POS_R, ANCHOR_ROT_R = 0, 1, 2, 3
# quantiles stored in every LeRobot stats dict the trainers write (keys q01 .. q99)
STAT_QUANTILES = (1, 10, 50, 90, 99)
