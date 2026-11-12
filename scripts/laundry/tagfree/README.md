# tagfree — pose chain without AprilTags (production snapshot, 2026-11-12)

On 11-11 every AprilTag was removed (hand rigs and table start tags); each demo starts with the hand units resting on two
black L tapes. These are the production files that process such takes, **copied as they ran on 11-12, not refactored**
into `src/laundry_tracking` yet (the chain was still being changed that day). Status and numbers:
[docs/status_2026-11-12.md](../../../docs/status_2026-11-12.md) §4.

| file | what | maintained counterpart |
|---|---|---|
| `v2b/tagfree_seed.py` | start seed per demo from the L tapes + resting units in depth → `start_seed_<take>.json` | — (new) |
| `v2b/tagfree_prep.py` | per-take `tagfree.json`, arm assignment, IMU-only idle trim, seeds | `laundry_tracking.new_take_prep` |
| `v2b/tagfree_yaw.py` | yaw from the IMU (Mahony) anchored by the L heading at rest; `--calibrate-L` | — (new) |
| `v2b/tagfree_derive.py` | derived take (fusion variant `fus_Bg_tfL`) for the label / dataset steps | `laundry_tracking.fusion.make_derived` |
| `v2b/cotrack/{make_pack2,solve2,gate}.py` | CoTracker pack (starts 0.8 s before the demo cue), point solver (rest-centre seed), gate (tag-free per-arm checks) | `laundry_tracking.cotrack.{make_pack,point_solver,gate}` |
| `v2b/v32_common.py`, `v2b/imu_motion.py` | arm rotations (pre-roll for tag-free seeds, rest centre), AHRS | `laundry_tracking.arm_rotation`, `laundry_tracking.imu_motion` |
| `v2b/camcal.py` | per-take camera ↔ robot calibration segments (adds `robot_cam_v007` from 11-12 12:40) | `laundry_tracking.calib` |
| `chain/tagfree_take_chain*.sh`, `chain/tagfree_from3.sh` | one take → per-take S03G LeRobot sets (prep → CoTracker → solve → gate → fusion → jaw → IK → grade → split → build) | README §5 stages 3–5 |
| `chain/tagfree_split.py` | GOLD / SILVER / REJECT → train / val / test / silver / reject (same rules as 11-04) | `laundry_dataset.builds.split_gold_silver_reject` |
| `chain/render_tagfree.py`, `chain/gather_*.sh`, `chain/start_sheet.py` | QA video per demo, clips per class, start-frame contact sheet | `laundry_tracking.fusion.render_qa` |
| `chain/cam_drift_icp.py`, `chain/l_tape_check.py` | camera motion between takes from the static depth scene (ICP); L-tape positions as a drift check | — (new) |
| `chain/a0_tagfree_test.py`, `chain/tagfree_vs_tag.py` | A0 marker-free label test on tagged data; tag-free chain on a tagged take vs its tags | — (validation) |

The scripts import the production tree (`~/ego_collector/datasets/m3_runs/_imu_consistency/orbbec/v2b`) and have machine
paths; they are here so the method is in the repository, not to be run from it.
