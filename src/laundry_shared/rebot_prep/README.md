# rebot_prep

Importable reBot preprocessing code shared by several entry scripts. Package name follows the company refactor plan
(`rebot_prep.parity` = tag-rig pose modules).

**Status:** current

## Inputs → Outputs
Orbbec tag detections + depth + IMU of a take → rig poses `T_C_rig`, fused with IMU orientation (`tag_rig_fuse.fuse_take`).

## Entry points
Library: `from rebot_prep.parity.tag_rig_pose import rig_pose`, `from rebot_prep.parity.tag_rig_fuse import fuse_take`.
CLIs: `PYTHONPATH=src python -m rebot_prep.parity.tag_rig_corners <take>`, `... -m rebot_prep.parity.tag_rig_imu <take> ...`.

## Layout
- `parity/tag_rig_pose.py` — multi-corner PnP, tag sizes, depth lookup, guard-tag rules
- `parity/tag_rig_fuse.py` — tags + IMU orientation → rig pose per frame
- `parity/tag_rig_corners.py` — table-corner check of the rig TCP
- `parity/tag_rig_imu.py` — rig angular velocity vs gyro

## Provenance
Moved from `scripts/tag_rig_{pose,fuse,corners,imu}.py` in the 11-01 layout refactor; every importer was switched to
`rebot_prep.parity.*` (no wrapper scripts left behind).
