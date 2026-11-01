# retarget

Human → robot mapping for the open-loop replay track: UMI gripper frame ↔ reBot end-effector alignment and the
robotization of a two-hand demo into a reBot-feasible trajectory.

**Status:** current for calibration (`umi_align.py`); replay track only for `umi_robotize_bi.py` (training labels are
built by `scripts/processing/labels_dataset/robotize_ik.py`)

## Inputs → Outputs
`umi_traj_{right,left}.npz` from `scripts/pose/umi_tcp_traj.py` → `umi_traj_<arm>_robot.npz` (robotized path, per-frame
quality); an align take → `configs/calibration/umi_gripper_align[_left]_v001.json` (R_U_E).

## Entry points
```bash
python scripts/retarget/umi_align.py datasets/tag_rig/<align take> [--write]          # needs visual_teleop (src/) + Pink IK
python scripts/retarget/umi_robotize_bi.py datasets/tag_rig/<take> --clear 0 --sep 0.06 --min-reach 0
```

## Layout
- `umi_align.py` — UMI TCP frame → reBot EEF frame rotation
- `umi_robotize_bi.py` — side keeping, table clearance, reach warps for both hands

## Provenance
Moved from flat `scripts/` in the 11-01 layout refactor.
