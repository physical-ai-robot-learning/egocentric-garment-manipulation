"""Dataset stage of the laundry pipeline: tag-rig takes -> robot-feasible labels (IK robotization) -> LeRobot datasets.

Interface used by other packages:
    Arm(side, args)   one reBot arm (Pink IK with the bridge limits, cornerLift / level, solve, links) -- robot_arm.Arm

Modules: takes (demo windows, idle-trim table), calibration (gripper alignment, jaw calibration gate), robot_arm (Arm),
robotize_ik
(IK labels, CLI), arm_assignment (arm / camera assignment + prep), export_common (shared LeRobot pieces), tag_rig_to_lerobot (v1
builder, source meta), unsplit_build / unsplit_render / tag_rig_unsplit (current builder), idle_trim (idle_trim.csv), ik_qa_stats
(IK QA tables), builds/ (11-02 .. 11-04 flatten build scripts).
"""
from __future__ import annotations

from .robot_arm import Arm

__all__ = ["Arm"]
