# Files moved in from egocentric-cube-stacking (2026-11-01)

These laundry-era files existed only in the cube repository (its 10-20 snapshot of the laundry work). They are kept here as records; current code lives in `src/` and `scripts/`. Mapping and md5 below.

# cube -> garment staging (egocentric-cube-stacking, branch refactor-v3, 2026-11-01)

Compared against `/Users/jeonghwanlee/physical-ai-vla-repos/egocentric-laundry-folding` (HEAD c905354; its content is what
egocentric-garment-manipulation now contains). Nothing was written into the garment repo.

- **staged**: exists only in cube -> copied here under its cube path, then removed from cube.
- **removed-identical**: byte-identical file exists in laundry-folding (path given) -> removed from cube.
- **removed-older**: laundry-folding holds a later version of the same file (laundry repo starts 2026-10-21; the cube copies are the 2026-10-20
  snapshot; diffs are path updates for the new layout and later edits) -> removed from cube.

Totals: staged 44 (incl. the README section), removed-identical 135, removed-older 33.

## Staged (only in cube)

| cube path | md5 | why laundry-era |
|---|---|---|
| `docs/UPDATE_2026-10-20.md` | `e10a3888e17884a3398dfd1064529b1c` | post-v5 update note: T-shirt flattening (laundry project), servo-less HandUMI, visual gripper, UMI->robot calibration |
| `docs/UPDATE_2026-10-20_ko.md` | `0b4d165af8b3319ed29be25d372245a7` | post-v5 update note: T-shirt flattening (laundry project), servo-less HandUMI, visual gripper, UMI->robot calibration |
| `docs/laundry/README.md` | `6dedb8c893353a142326a8d16e1f8080` | README of scripts/rebot/laundry/ (UMIalign pose + visual gripper) (only a same-name README.md with different content exists) |
| `docs/parity/README.md` | `be662cc134abe8918167a66ff2cd961d` | README of scripts/rebot/parity/ (UMI TCP calibration, UMI->reBot replay parity) (only a same-name README.md with different content exists) |
| `docs/provenance/umi_laundry_MANIFEST.md` | `6d59ee9e2a5472a257efe87a6e27e6e1` | provenance record of the former pipeline/umi_laundry/ (= scripts/rebot/laundry/) |
| `docs/provenance/umi_robot_parity_MANIFEST.md` | `72f4df22cde7db10e3ce32d40631a6c4` | provenance record of the former pipeline/umi_robot_parity/ (= scripts/rebot/parity/) |
| `scripts/rebot/laundry/drivers/abl_m3.sh` | `30089628e5951ae0c479978b2d985d50` | laundry run driver (UMIalign/UMIpilot/Hfold MASt3R, Orbbec, smoother, export chains, 10-13..10-15) |
| `scripts/rebot/laundry/drivers/after_sf1000k_m3_1014.sh` | `986a26acef880adb7e8f3915d6ad3593` | laundry run driver (UMIalign/UMIpilot/Hfold MASt3R, Orbbec, smoother, export chains, 10-13..10-15) |
| `scripts/rebot/laundry/drivers/canon_110829.sh` | `5ba4b49efa6ebd8ed051dee3249d8360` | laundry run driver (UMIalign/UMIpilot/Hfold MASt3R, Orbbec, smoother, export chains, 10-13..10-15) |
| `scripts/rebot/laundry/drivers/export_1015.sh` | `39cc4aaf07f2397ca209d8338ea95b2b` | laundry run driver (UMIalign/UMIpilot/Hfold MASt3R, Orbbec, smoother, export chains, 10-13..10-15) |
| `scripts/rebot/laundry/drivers/export_rest_1014.sh` | `57f45eeb636c1458385ac5e048e14886` | laundry run driver (UMIalign/UMIpilot/Hfold MASt3R, Orbbec, smoother, export chains, 10-13..10-15) |
| `scripts/rebot/laundry/drivers/gate_chain_93.sh` | `88e3c8e60e089908fe18ab76f387069c` | laundry run driver (UMIalign/UMIpilot/Hfold MASt3R, Orbbec, smoother, export chains, 10-13..10-15) |
| `scripts/rebot/laundry/drivers/hfold_m3_all.sh` | `7c0d86ac913f4e762dd8a1c69758d0fd` | laundry run driver (UMIalign/UMIpilot/Hfold MASt3R, Orbbec, smoother, export chains, 10-13..10-15) |
| `scripts/rebot/laundry/drivers/m3_chain_1013.sh` | `b62394adfeea972fb7656eafa6b8e493` | laundry run driver (UMIalign/UMIpilot/Hfold MASt3R, Orbbec, smoother, export chains, 10-13..10-15) |
| `scripts/rebot/laundry/drivers/m3_then_resume_1013.sh` | `f8cbc38ed6542d8d0a3ea009db1ba6b9` | laundry run driver (UMIalign/UMIpilot/Hfold MASt3R, Orbbec, smoother, export chains, 10-13..10-15) |
| `scripts/rebot/laundry/drivers/orbbec_112743.sh` | `ce037bb6de52b819327fee78bfe46cfa` | laundry run driver (UMIalign/UMIpilot/Hfold MASt3R, Orbbec, smoother, export chains, 10-13..10-15) |
| `scripts/rebot/laundry/drivers/orbbec_resume_112920.sh` | `262f06fec4c4da9174a1fee9e9fa3450` | laundry run driver (UMIalign/UMIpilot/Hfold MASt3R, Orbbec, smoother, export chains, 10-13..10-15) |
| `scripts/rebot/laundry/drivers/pilot2_m3.sh` | `b27d489e16a8b371de51191df354dd59` | laundry run driver (UMIalign/UMIpilot/Hfold MASt3R, Orbbec, smoother, export chains, 10-13..10-15) |
| `scripts/rebot/laundry/drivers/restart_m3.sh` | `a7ca29d60d8856d4f45e767af713d733` | laundry run driver (UMIalign/UMIpilot/Hfold MASt3R, Orbbec, smoother, export chains, 10-13..10-15) |
| `scripts/rebot/laundry/drivers/rp_after_1015.sh` | `dc8e15697931d8c798bd66bb6b72cb51` | laundry run driver (UMIalign/UMIpilot/Hfold MASt3R, Orbbec, smoother, export chains, 10-13..10-15) |
| `scripts/rebot/laundry/drivers/smooth_112743.sh` | `db1113c1498c7aca13ed9c02d9c41f0a` | laundry run driver (UMIalign/UMIpilot/Hfold MASt3R, Orbbec, smoother, export chains, 10-13..10-15) |
| `scripts/rebot/laundry/drivers/smooth_112920.sh` | `e7b594b0a0ccbbc9dd79118452924f4b` | laundry run driver (UMIalign/UMIpilot/Hfold MASt3R, Orbbec, smoother, export chains, 10-13..10-15) |
| `scripts/rebot/laundry/drivers/smooth_rcb3_93.sh` | `22db31c9ddcd99d9c8006aba50ed2553` | laundry run driver (UMIalign/UMIpilot/Hfold MASt3R, Orbbec, smoother, export chains, 10-13..10-15) |
| `scripts/rebot/laundry/drivers/smooth_rcb3fix_93.sh` | `a035d691dd9cf724d59e65908472d913` | laundry run driver (UMIalign/UMIpilot/Hfold MASt3R, Orbbec, smoother, export chains, 10-13..10-15) |
| `scripts/rebot/laundry/drivers/stream_1015.sh` | `78c159d8d034daf0583d26cafe551aec` | laundry run driver (UMIalign/UMIpilot/Hfold MASt3R, Orbbec, smoother, export chains, 10-13..10-15) |
| `scripts/rebot/laundry/drivers/track_all_1013.sh` | `3c9fd14748ac0119c522fb37bf86b41b` | laundry run driver (UMIalign/UMIpilot/Hfold MASt3R, Orbbec, smoother, export chains, 10-13..10-15) |
| `scripts/rebot/laundry/drivers/track_chain_1013.sh` | `46524825a8c1b805afd910bafee9323a` | laundry run driver (UMIalign/UMIpilot/Hfold MASt3R, Orbbec, smoother, export chains, 10-13..10-15) |
| `scripts/rebot/laundry/drivers/umialign_metrics.py` | `a6ab48db3bf39c5a6726fe1d6db08707` | laundry run driver (UMIalign/UMIpilot/Hfold MASt3R, Orbbec, smoother, export chains, 10-13..10-15) |
| `scripts/rebot/laundry/drivers/umipilot_m3.sh` | `6683e51699eee4ff212ee4b1c7abd9c8` | laundry run driver (UMIalign/UMIpilot/Hfold MASt3R, Orbbec, smoother, export chains, 10-13..10-15) |
| `scripts/rebot/parity/kalibr/patch_kalibr.py` | `e72f4d00e89a87f93a8fff7255ca968e` | Kalibr stereo-IMU run bundle for the UMI wrist cameras (10-16, UMI->robot parity work) |
| `scripts/rebot/parity/kalibr/remote_kalibr.sh` | `968705ab9861ab04cece30e4f71a4071` | Kalibr stereo-IMU run bundle for the UMI wrist cameras (10-16, UMI->robot parity work) |
| `scripts/rebot/parity/kalibr/run_kalibr.sh` | `befd69f32d037fa2bb9c2300b62817d2` | Kalibr stereo-IMU run bundle for the UMI wrist cameras (10-16, UMI->robot parity work) |
| `scripts/umi_teleop_eval/umi_body_kpt_dataset.py` | `1afd0e33b8bf26c436c1ebec9401be32` | UMI keypoint teleop evaluation; imports visual_teleop (laundry-era teleop, 10-14..10-16) |
| `scripts/umi_teleop_eval/umi_kpt_dataset.py` | `b5acff7e537b86284928ae8c41d4defc` | UMI keypoint teleop evaluation; imports visual_teleop (laundry-era teleop, 10-14..10-16) |
| `scripts/umi_teleop_eval/umi_kpt_eval.py` | `6d72290dc67cc81dd20bb9ee0dc4b64f` | UMI keypoint teleop evaluation; imports visual_teleop (laundry-era teleop, 10-14..10-16) |
| `tests/experimental/vteleop/conftest.py` | `e395d43536b261636b95f18e414ab3d9` | tests of src/visual_teleop (older copy; garment owns the newer package); collection already failed in cube (ImportError gripper_cmd_from_g) (only a same-name conftest.py with different content exists) |
| `tests/experimental/vteleop/test_controller_safety.py` | `20e8f8c25ffcfbb246c3ac9abf74980b` | tests of src/visual_teleop (older copy; garment owns the newer package); collection already failed in cube (ImportError gripper_cmd_from_g) |
| `tests/experimental/vteleop/test_frames_and_retarget.py` | `0353113fb876070e1daaf4faade91b6c` | tests of src/visual_teleop (older copy; garment owns the newer package); collection already failed in cube (ImportError gripper_cmd_from_g) |
| `tests/experimental/vteleop/test_imu_fusion.py` | `91c3a65572193eaab2fa62e40260ca72` | tests of src/visual_teleop (older copy; garment owns the newer package); collection already failed in cube (ImportError gripper_cmd_from_g) |
| `tests/experimental/vteleop/test_kpt_translation.py` | `badfee0e1f0e6b27a56fe8d0ef41c137` | tests of src/visual_teleop (older copy; garment owns the newer package); collection already failed in cube (ImportError gripper_cmd_from_g) |
| `tests/experimental/vteleop/test_recording.py` | `791a7b744e7d22e08f18c38ee846bc81` | tests of src/visual_teleop (older copy; garment owns the newer package); collection already failed in cube (ImportError gripper_cmd_from_g) |
| `tests/experimental/vteleop/test_residual_monitor.py` | `760f86e4f7b91cf1f99de16f2cef7b0c` | tests of src/visual_teleop (older copy; garment owns the newer package); collection already failed in cube (ImportError gripper_cmd_from_g) |
| `tests/experimental/vteleop/test_tracker6dof.py` | `af22d68cf1b521aa6761b79fe6f2d32b` | tests of src/visual_teleop (older copy; garment owns the newer package); collection already failed in cube (ImportError gripper_cmd_from_g) |
| `README.md` lines 488-507, section "Update 2026-10-20" -> `README.md.section_update_2026-10-20.md` | `c5efd88a12b4c01df9aebdfc1c8efefd` | README section on the laundry work (servo-less HandUMI, visual gripper, UMIalign, UMI->robot), replaced in cube by a pointer to garment |

## Removed, newer version in laundry-folding

| cube path | md5 (cube) | laundry-folding counterpart | diff |
|---|---|---|---|
| `scripts/rebot/parity/scripts/replay_parity.py` | `af160f5a97f5c5c47cb737940e7d7d94` | `scripts/evaluation/replay_parity.py` | difflines=207, cube_only_lines=12; lines cube 180 / laundry 363 |
| `scripts/rebot/parity/scripts/robot_cam_solve.py` | `61a0f7903609d99832846f3189592714` | `scripts/pose/robot_cam_solve.py` | difflines=39, cube_only_lines=9; lines cube 222 / laundry 243 |
| `scripts/rebot/parity/scripts/tag_rig_corners.py` | `9598ab33bd301624a0e6b963e5b3bb4d` | `src/rebot_prep/parity/tag_rig_corners.py` | difflines=10, cube_only_lines=5; lines cube 109 / laundry 109 |
| `scripts/rebot/parity/scripts/tag_rig_fuse.py` | `5d4cfb904d5138dbf8ce2d3657c1a359` | `src/rebot_prep/parity/tag_rig_fuse.py` | difflines=75, cube_only_lines=20; lines cube 63 / laundry 98 |
| `scripts/rebot/parity/scripts/tag_rig_imu.py` | `0115f9f551ac69802d79438e733b00ab` | `src/rebot_prep/parity/tag_rig_imu.py` | difflines=14, cube_only_lines=6; lines cube 94 / laundry 96 |
| `scripts/rebot/parity/scripts/tag_rig_pivot.py` | `6ae696c9f3eaf200a8c622dc789c33cc` | `scripts/pose/tag_rig_pivot.py` | difflines=4, cube_only_lines=2; lines cube 65 / laundry 65 |
| `scripts/rebot/parity/scripts/tag_rig_pivot_joint.py` | `d0481beddde3402072994b471e35ac54` | `scripts/pose/tag_rig_pivot_joint.py` | difflines=10, cube_only_lines=5; lines cube 76 / laundry 76 |
| `scripts/rebot/parity/scripts/tag_rig_pose.py` | `7c90b74cc9650e0bebeb5f239366d2db` | `src/rebot_prep/parity/tag_rig_pose.py` | difflines=47, cube_only_lines=4; lines cube 58 / laundry 97 |
| `scripts/rebot/parity/scripts/tag_rig_solve.py` | `af95186c88c42f3232be89ea5b409018` | `scripts/pose/tag_rig_solve.py` | difflines=13, cube_only_lines=5; lines cube 115 / laundry 118 |
| `scripts/rebot/parity/scripts/tag_rig_touch_tcp.py` | `8123a3bf652461bde0c1c598da48b9cd` | `scripts/pose/tag_rig_touch_tcp.py` | difflines=14, cube_only_lines=7; lines cube 131 / laundry 131 |
| `scripts/rebot/parity/scripts/umi_tcp_traj.py` | `e1da0078482fccfa151c6cd669f65c87` | `scripts/pose/umi_tcp_traj.py` | difflines=57, cube_only_lines=16; lines cube 58 / laundry 83 |
| `scripts/rebot/parity/tools/ready_pose.py` | `50c4738ad5df248ce9bb9190e9036a4c` | `src/visual_teleop/tools/ready_pose.py` | difflines=3, cube_only_lines=1; lines cube 76 / laundry 77 |
| `scripts/rebot/parity/tools/robot_cam_poses.py` | `5ab8150710ac4af30a2aba01d29f8aa1` | `src/visual_teleop/tools/robot_cam_poses.py` | difflines=10, cube_only_lines=4; lines cube 113 / laundry 115 |
| `scripts/rebot/parity/tools/tag_rig_record.py` | `2242b3a2211657430c1a903c56ec828d` | `src/visual_teleop/tools/tag_rig_record.py` | difflines=295, cube_only_lines=17; lines cube 188 / laundry 449 |
| `scripts/rebot/parity/tools/umi_replay.py` | `0f67b19856ccda60902b142acd270332` | `src/visual_teleop/tools/umi_replay.py` | difflines=161, cube_only_lines=36; lines cube 123 / laundry 212 |
| `src/visual_teleop/README.md` | `7a5ea630b5979dc5f8948ee80ca4a81d` | `src/visual_teleop/README.md` | difflines=51, cube_only_lines=19; lines cube 100 / laundry 113 |
| `src/visual_teleop/config/teleop_fastumi.yaml` | `79cff406503fba971702c8335fabb9ff` | `src/visual_teleop/config/teleop_fastumi.yaml` | difflines=4, cube_only_lines=2; lines cube 7 / laundry 7 |
| `src/visual_teleop/config/teleop_kpt.yaml` | `3bc8f61a275ad1524ab5c056db60c132` | `src/visual_teleop/config/teleop_kpt.yaml` | difflines=4, cube_only_lines=2; lines cube 6 / laundry 6 |
| `src/visual_teleop/config/teleop_rebot.yaml` | `837699f472b75fef73bb1204486eaee8` | `src/visual_teleop/config/teleop_rebot.yaml` | difflines=4, cube_only_lines=2; lines cube 202 / laundry 202 |
| `src/visual_teleop/config/teleop_tracker.yaml` | `0ee78b7662fc9641bc5da3fceb6143c4` | `src/visual_teleop/config/teleop_tracker.yaml` | difflines=4, cube_only_lines=2; lines cube 29 / laundry 29 |
| `src/visual_teleop/config/teleop_tracker_axis_x.yaml` | `b35df9f279e10a1ba1d9d5908ff5add5` | `src/visual_teleop/config/teleop_tracker_axis_x.yaml` | difflines=4, cube_only_lines=2; lines cube 31 / laundry 31 |
| `src/visual_teleop/robot/ik_pink.py` | `56f859420debcef0f449223aeabf13e8` | `src/visual_teleop/robot/ik_pink.py` | difflines=32, cube_only_lines=3; lines cube 105 / laundry 131 |
| `src/visual_teleop/tools/orbbec_tag_check.py` | `84c611a84924ef59fc3af6334a97c0b4` | `src/visual_teleop/tools/orbbec_tag_check.py` | difflines=2, cube_only_lines=1; lines cube 62 / laundry 62 |
| `src/visual_teleop/tools/orient_probe.py` | `dc25bdf3141840904377f58c95591a97` | `src/visual_teleop/tools/orient_probe.py` | difflines=2, cube_only_lines=1; lines cube 84 / laundry 84 |
| `src/visual_teleop/tools/ready_pose.py` | `50c4738ad5df248ce9bb9190e9036a4c` | `src/visual_teleop/tools/ready_pose.py` | difflines=3, cube_only_lines=1; lines cube 76 / laundry 77 |
| `src/visual_teleop/tools/robot_cam_poses.py` | `828307419b57775b50f9e64cb8bc8a1d` | `src/visual_teleop/tools/robot_cam_poses.py` | difflines=8, cube_only_lines=3; lines cube 113 / laundry 115 |
| `src/visual_teleop/tools/robot_moves.py` | `6c8560365b15696ff374f668aa8e36d4` | `src/visual_teleop/tools/robot_moves.py` | difflines=3, cube_only_lines=1; lines cube 41 / laundry 42 |
| `src/visual_teleop/tools/stereo_check.py` | `dc81cef640d4b25e65a6ab921306c87a` | `src/visual_teleop/tools/stereo_check.py` | difflines=1, cube_only_lines=0; lines cube 129 / laundry 130 |
| `src/visual_teleop/tools/tag_rig_record.py` | `61df2bf537f44bfc122d443ecf945eb1` | `src/visual_teleop/tools/tag_rig_record.py` | difflines=293, cube_only_lines=16; lines cube 188 / laundry 449 |
| `src/visual_teleop/tools/trackcam_calib.py` | `9d399386b9b1e7becb15e0f56aab6eab` | `src/visual_teleop/tools/trackcam_calib.py` | difflines=2, cube_only_lines=1; lines cube 151 / laundry 151 |
| `src/visual_teleop/tools/umi_replay.py` | `e45660712ec93ec66a57573ef89555a2` | `src/visual_teleop/tools/umi_replay.py` | difflines=159, cube_only_lines=35; lines cube 123 / laundry 212 |
| `src/visual_teleop/vision/fingertip_detector.py` | `63dfb4313b94ea83e369d58cd02a4a10` | `src/visual_teleop/vision/fingertip_detector.py` | difflines=2, cube_only_lines=1; lines cube 140 / laundry 140 |
| `src/visual_teleop/vision/stereo_vio_tracker.py` | `81ae19837edea18d9d7e24b08a873647` | `src/visual_teleop/vision/stereo_vio_tracker.py` | difflines=2, cube_only_lines=1; lines cube 541 / laundry 541 |

## Removed, identical in laundry-folding

| cube path | md5 | laundry-folding path |
|---|---|---|
| `scripts/rebot/laundry/orbbec_tracking/handumi_body_track.py` | `4b9769f880bdd0f067ba4e323636da4a` | `legacy/scripts/handumi_body_track.py` |
| `scripts/rebot/laundry/orbbec_tracking/handumi_canonical_eval.py` | `1c229a6f7b5a60f763b6bbf77a516dff` | `legacy/scripts/handumi_canonical_eval.py` |
| `scripts/rebot/laundry/orbbec_tracking/handumi_canonical_track.py` | `41b6f5b64798cc57e9eec53ae7d9bb20` | `legacy/scripts/handumi_canonical_track.py` |
| `scripts/rebot/laundry/umi_fusion_scripts/detect_wrist_flip.py` | `3285057b0b69725b2c09d9d721e76ac3` | `legacy/scripts/umi_fusion/detect_wrist_flip.py` |
| `scripts/rebot/laundry/umi_fusion_scripts/imu_bridge_validate.py` | `05b085d301a53b286147efef7913ca54` | `legacy/scripts/umi_fusion/imu_bridge_validate.py` |
| `scripts/rebot/laundry/umi_fusion_scripts/imu_cam_extrinsic_check.py` | `5919af9e3256c60db212858629242408` | `legacy/scripts/umi_fusion/imu_cam_extrinsic_check.py` |
| `scripts/rebot/laundry/umi_fusion_scripts/jaw_from_video_eval.py` | `3b79d6e34d80dff7959a477b14999d19` | `legacy/scripts/umi_fusion/jaw_from_video_eval.py` |
| `scripts/rebot/laundry/umi_fusion_scripts/jaw_gap_video.py` | `c5e02d1839e3d4dd778650547434b022` | `legacy/scripts/umi_fusion/jaw_gap_video.py` |
| `scripts/rebot/laundry/umi_fusion_scripts/jaw_redx.py` | `d73997fa9e2f30b6683b8eb6beb0b417` | `legacy/scripts/umi_fusion/jaw_redx.py` |
| `scripts/rebot/laundry/umi_fusion_scripts/jaw_track.py` | `30fc77940f62b1195d2fd3234d6fe668` | `legacy/scripts/umi_fusion/jaw_track.py` |
| `scripts/rebot/laundry/umi_fusion_scripts/m3_runner/run_perframe_resident.py` | `7e7d94c01da4cca02e81fbe1b15339a0` | `legacy/scripts/umi_fusion/m3_runner/run_perframe_resident.py` |
| `scripts/rebot/laundry/umi_fusion_scripts/m3_runner/run_perframe_restart.py` | `5d83a13ef5dd471671d4677cbff458c5` | `legacy/scripts/umi_fusion/m3_runner/run_perframe_restart.py` |
| `scripts/rebot/laundry/umi_fusion_scripts/make_gripper_tags.py` | `8c4bb0d24be7c95bcbf823304a395216` | `legacy/scripts/umi_fusion/make_gripper_tags.py` |
| `scripts/rebot/laundry/umi_fusion_scripts/orbbec_tag_check.py` | `0fecae672424981a2a45f2f58e5b6979` | `legacy/scripts/umi_fusion/orbbec_tag_check.py` |
| `scripts/rebot/laundry/umi_fusion_scripts/orbbec_tag_gate.py` | `a471d4d62dc6a96ebf4936c4de0ef921` | `legacy/scripts/umi_fusion/orbbec_tag_gate.py` |
| `scripts/rebot/laundry/umi_fusion_scripts/raw_qa.py` | `cfece48de7f4cc0269bfda488d6e32eb` | `legacy/scripts/umi_fusion/raw_qa.py` |
| `scripts/rebot/laundry/umi_fusion_scripts/space_trim_points.py` | `98c05ed9a572170c99fe3866e1ad61a8` | `legacy/scripts/umi_fusion/space_trim_points.py` |
| `scripts/rebot/laundry/umi_fusion_scripts/tag_vs_imu.py` | `dd43aa45f3fb5821038c09fd03f912a4` | `legacy/scripts/umi_fusion/tag_vs_imu.py` |
| `scripts/rebot/laundry/umi_fusion_scripts/tcp_axes_overlay.py` | `41ef4c4770ff81a40dac3e3e61e03a78` | `legacy/scripts/umi_fusion/tcp_axes_overlay.py` |
| `scripts/rebot/laundry/umi_fusion_scripts/viz_today.py` | `c632d770d7f94dc855beaae82f04bfe6` | `legacy/scripts/umi_fusion/viz_today.py` |
| `scripts/rebot/parity/configs/robot_cam_v001.json` | `fc928894fcb34f93cb3649bd67293026` | `configs/calibration/robot_cam_v001.json` |
| `scripts/rebot/parity/configs/umi_tag_rig_v001.json` | `6342d784b9c7ba0dccab1d9643670c7b` | `configs/calibration/umi_tag_rig_v001.json` |
| `scripts/rebot/parity/print/aprilgrid_3x4_50mm_id300.pdf` | `b9f012380a10c11b1a44834e9e1a0681` | `hardware/umihand/tags/aprilgrid_3x4_50mm_id300.pdf` |
| `scripts/rebot/parity/print/robot_gripper_tags_left_230-232_50mm.pdf` | `72bc391e093710ffdbad2294768bb550` | `hardware/umihand/tags/robot_gripper_tags_left_230-232_50mm.pdf` |
| `scripts/rebot/parity/print/robot_gripper_tags_right_220-222_50mm.pdf` | `90b684a6106b13c7a57b31a6ddf77704` | `hardware/umihand/tags/robot_gripper_tags_right_220-222_50mm.pdf` |
| `scripts/rebot/parity/print/umi_anchor_tags_200-202_50mm.pdf` | `bb181f3375b89cfed7715ffa506225e8` | `hardware/umihand/tags/umi_anchor_tags_200-202_50mm.pdf` |
| `scripts/rebot/parity/print/umi_anchor_tags_left_210-212_50mm.pdf` | `15fc67aada38e30ad1a62923ef43b930` | `hardware/umihand/tags/umi_anchor_tags_left_210-212_50mm.pdf` |
| `scripts/rebot/parity/scripts/kalibr_bag_export.py` | `5cfea1240ad0ef15352cfd867ed6201b` | `legacy/scripts/kalibr_bag_export.py` |
| `scripts/rebot/parity/scripts/kalibr_to_camimu.py` | `ec1e1ca71ab0b41473c8d95d09f51f0a` | `legacy/scripts/kalibr_to_camimu.py` |
| `scripts/umi_teleop_eval/dpvo_eval_metrics.py` | `30df2a5dd256512aa09b991ae3afb3bf` | `legacy/scripts/dpvo_eval_metrics.py` |
| `scripts/umi_teleop_eval/finger_distill_2kpt.py` | `4b4002d1c84d5456613f4ceae702af20` | `legacy/scripts/stereo_vio_and_finger/finger_distill_2kpt.py` |
| `scripts/umi_teleop_eval/finger_distill_label.py` | `a483e8dfa6615bb35bfdf5dfe798c51f` | `legacy/scripts/stereo_vio_and_finger/finger_distill_label.py` |
| `scripts/umi_teleop_eval/finger_distill_record.py` | `06edf617390db4572c8d09b9e0a7d9d7` | `legacy/scripts/stereo_vio_and_finger/finger_distill_record.py` |
| `scripts/umi_teleop_eval/finger_distill_train.py` | `c16e1a45be69bc0669e0aa1351cf58da` | `legacy/scripts/stereo_vio_and_finger/finger_distill_train.py` |
| `scripts/umi_teleop_eval/finger_grip_teleop.py` | `05eecbcc7b1b821aa05fba7a488dcda0` | `legacy/scripts/stereo_vio_and_finger/finger_grip_teleop.py` |
| `scripts/umi_teleop_eval/fisheye_calib_robust.py` | `5cf42f1a2bce17b1bd3de4074afa0a7e` | `legacy/scripts/stereo_vio_and_finger/fisheye_calib_robust.py` |
| `scripts/umi_teleop_eval/intrinsics_compare.py` | `0ebadd0ede39a1d50edd29514542caa2` | `legacy/scripts/stereo_vio_and_finger/intrinsics_compare.py` |
| `scripts/umi_teleop_eval/m3_imu_eval_metrics.py` | `e5230b5f2e2efbc5830095646be1273a` | `legacy/scripts/m3_imu_eval_metrics.py` |
| `scripts/umi_teleop_eval/stereo_basalt_eval.py` | `67c8d160fdd18e0da294da11e54d99cd` | `legacy/scripts/stereo_vio_and_finger/stereo_basalt_eval.py` |
| `scripts/umi_teleop_eval/stereo_calib.py` | `cc4a1d85aa5ff6f5bd47beac5915b73a` | `legacy/scripts/stereo_vio_and_finger/stereo_calib.py` |
| `scripts/umi_teleop_eval/stereo_calib_full.py` | `51ed9ffd12b7532bd6a307ccb7ef977a` | `legacy/scripts/stereo_vio_and_finger/stereo_calib_full.py` |
| `scripts/umi_teleop_eval/stereo_openvins_eval.py` | `9838fbcb23c3d09aa3bba8b494cdefa3` | `legacy/scripts/stereo_vio_and_finger/stereo_openvins_eval.py` |
| `scripts/umi_teleop_eval/trackcam_basalt_calib.py` | `98678f559b9a6cf5400d755459eb4ac5` | `legacy/scripts/stereo_vio_and_finger/trackcam_basalt_calib.py` |
| `scripts/umi_teleop_eval/trackcam_basalt_eval.py` | `4c1f94bd553021ef46566caa624c7388` | `legacy/scripts/stereo_vio_and_finger/trackcam_basalt_eval.py` |
| `scripts/umi_teleop_eval/trackcam_board_lever.py` | `d8e0b91bff3e9a51d4bfa323d2d1a238` | `legacy/scripts/stereo_vio_and_finger/trackcam_board_lever.py` |
| `scripts/umi_teleop_eval/trackcam_imu_offline.py` | `c7bb2ef020328de70757f5c32f3c4371` | `legacy/scripts/stereo_vio_and_finger/trackcam_imu_offline.py` |
| `scripts/umi_teleop_eval/trackcam_m3_vs_board.py` | `26df0177850896a3c4475a17019f027c` | `legacy/scripts/stereo_vio_and_finger/trackcam_m3_vs_board.py` |
| `scripts/umi_teleop_eval/trackcam_orb3_export.py` | `292593e6c94e2fee99721566e2ff8759` | `legacy/scripts/stereo_vio_and_finger/trackcam_orb3_export.py` |
| `scripts/umi_teleop_eval/trackcam_tap_test.py` | `3a629e301ed0e452f0fbb1b600f9b0a5` | `legacy/scripts/stereo_vio_and_finger/trackcam_tap_test.py` |
| `scripts/umi_teleop_eval/trackcam_vio_eval.py` | `256605bf8a3d53ecac65c23a1d259413` | `legacy/scripts/stereo_vio_and_finger/trackcam_vio_eval.py` |
| `scripts/umi_teleop_eval/trackcam_vio_vs_board.py` | `0910e452928bd29351444a6dd6fe95cf` | `legacy/scripts/stereo_vio_and_finger/trackcam_vio_vs_board.py` |
| `scripts/umi_teleop_eval/vio_verify.py` | `84a4453e87595ce08c38246bbad3eeb9` | `legacy/scripts/stereo_vio_and_finger/vio_verify.py` |
| `src/visual_teleop/RUNBOOK_RIGHT_MVP.md` | `4891391570778033e566c280bd9a5922` | `src/visual_teleop/RUNBOOK_RIGHT_MVP.md` |
| `src/visual_teleop/__init__.py` | `d41d8cd98f00b204e9800998ecf8427e` | `legacy/laundry_collector/preprocess/__init__.py` |
| `src/visual_teleop/assets/umi_tag36h11_id0_30mm.png` | `661fff39b3e7c1c676a43ceeec73e4b7` | `src/visual_teleop/assets/umi_tag36h11_id0_30mm.png` |
| `src/visual_teleop/build.py` | `e651f885e38a0201960e4dac95ba2d8e` | `src/visual_teleop/build.py` |
| `src/visual_teleop/config.py` | `37cefe4a119384dd952bb37a518b59d6` | `src/visual_teleop/config.py` |
| `src/visual_teleop/config/calibration.yaml` | `fc34104cf1486c7b1b7edb72006865d0` | `src/visual_teleop/config/calibration.yaml` |
| `src/visual_teleop/config/calibration_log.jsonl` | `f8b376eb090daa7af99ee324f63f41d3` | `src/visual_teleop/config/calibration_log.jsonl` |
| `src/visual_teleop/fusion/__init__.py` | `d41d8cd98f00b204e9800998ecf8427e` | `legacy/laundry_collector/preprocess/__init__.py` |
| `src/visual_teleop/fusion/filters.py` | `82d5e4288dc7687150beac08ddcac68a` | `src/visual_teleop/fusion/filters.py` |
| `src/visual_teleop/fusion/wrist_pose_fusion.py` | `c479f99f2b3706e8940b961fc95c6010` | `src/visual_teleop/fusion/wrist_pose_fusion.py` |
| `src/visual_teleop/gripper/__init__.py` | `d41d8cd98f00b204e9800998ecf8427e` | `legacy/laundry_collector/preprocess/__init__.py` |
| `src/visual_teleop/gripper/gripper_mapper.py` | `98b65c92506b2d2dc31acf526f55a0c5` | `src/visual_teleop/gripper/gripper_mapper.py` |
| `src/visual_teleop/gripper/pinch_detector.py` | `98219afa276086d8ba7b121a3ffcec27` | `src/visual_teleop/gripper/pinch_detector.py` |
| `src/visual_teleop/imu/__init__.py` | `d41d8cd98f00b204e9800998ecf8427e` | `legacy/laundry_collector/preprocess/__init__.py` |
| `src/visual_teleop/imu/ahrs.py` | `d42dde3aa6eeef6c554d59e7cec08a65` | `src/visual_teleop/imu/ahrs.py` |
| `src/visual_teleop/imu/imu_reader.py` | `77d165d05d7b9939420120b49bf45319` | `src/visual_teleop/imu/imu_reader.py` |
| `src/visual_teleop/imu/time_sync.py` | `e91bb765d2757a0909d008ad1c45ccf3` | `src/visual_teleop/imu/time_sync.py` |
| `src/visual_teleop/link.py` | `33e90e89f372b7ce5c76b2ffa0b914f4` | `src/visual_teleop/link.py` |
| `src/visual_teleop/recording/__init__.py` | `d41d8cd98f00b204e9800998ecf8427e` | `legacy/laundry_collector/preprocess/__init__.py` |
| `src/visual_teleop/recording/episode_qa.py` | `edb02efd6bfc1425e7d154edcf4f06ae` | `src/visual_teleop/recording/episode_qa.py` |
| `src/visual_teleop/recording/export_lerobot.py` | `4ea76395f2ebea0c5d7ddc3db1120bd8` | `src/visual_teleop/recording/export_lerobot.py` |
| `src/visual_teleop/recording/recorder.py` | `bc40780dc8abb458b2712a4048bfd5b9` | `src/visual_teleop/recording/recorder.py` |
| `src/visual_teleop/reports/calibration_report.md` | `0c2d973067a718bba2f023931afc5d41` | `src/visual_teleop/reports/calibration_report.md` |
| `src/visual_teleop/reports/imu_bringup_20261014.json` | `3796d92a1a0dd1785fdab88849d50ba9` | `src/visual_teleop/reports/imu_bringup_20261014.json` |
| `src/visual_teleop/reports/imu_bringup_xiao_20261014.json` | `ee30229a885fed162fc2e30fe34ed75e` | `src/visual_teleop/reports/imu_bringup_xiao_20261014.json` |
| `src/visual_teleop/reports/latency_report.md` | `4aede0f6b9250fa1eae76442d79c97e8` | `src/visual_teleop/reports/latency_report.md` |
| `src/visual_teleop/reports/teleop_validation.md` | `49af0956dbc1665bf552a6cc166ec2db` | `src/visual_teleop/reports/teleop_validation.md` |
| `src/visual_teleop/retargeting/__init__.py` | `d41d8cd98f00b204e9800998ecf8427e` | `legacy/laundry_collector/preprocess/__init__.py` |
| `src/visual_teleop/retargeting/frame_adapter.py` | `3151fd7c6e12432b4de99c280c749849` | `src/visual_teleop/retargeting/frame_adapter.py` |
| `src/visual_teleop/retargeting/motion_scaling.py` | `04d782d964efcf6af10d2e92cb034318` | `src/visual_teleop/retargeting/motion_scaling.py` |
| `src/visual_teleop/retargeting/relative_se3.py` | `24cf9d714b0e38c75a445b6eeb8d9eff` | `src/visual_teleop/retargeting/relative_se3.py` |
| `src/visual_teleop/robot/__init__.py` | `d41d8cd98f00b204e9800998ecf8427e` | `legacy/laundry_collector/preprocess/__init__.py` |
| `src/visual_teleop/robot/bridge.py` | `b8e8c34d40a47c6ada072f22c4d7216a` | `src/visual_teleop/robot/bridge.py` |
| `src/visual_teleop/robot/dual_arm_controller.py` | `72b216b42d3998c62330aab8270da754` | `src/visual_teleop/robot/dual_arm_controller.py` |
| `src/visual_teleop/robot/ik.py` | `1739074ffff014634a817a61e236efa2` | `src/visual_teleop/robot/ik.py` |
| `src/visual_teleop/safety/__init__.py` | `d41d8cd98f00b204e9800998ecf8427e` | `legacy/laundry_collector/preprocess/__init__.py` |
| `src/visual_teleop/safety/clutch.py` | `c3d9c507836e59260c559e6096b080a2` | `src/visual_teleop/safety/clutch.py` |
| `src/visual_teleop/safety/collision.py` | `e9406d3a716b59f71a948b6fb06461ee` | `src/visual_teleop/safety/collision.py` |
| `src/visual_teleop/safety/deadman.py` | `d6215571bf6e52a7549d0258022f52aa` | `src/visual_teleop/safety/deadman.py` |
| `src/visual_teleop/safety/limits.py` | `2627c44c250880bdf4b783683e6c9e07` | `src/visual_teleop/safety/limits.py` |
| `src/visual_teleop/tools/__init__.py` | `d41d8cd98f00b204e9800998ecf8427e` | `legacy/laundry_collector/preprocess/__init__.py` |
| `src/visual_teleop/tools/benchmark.py` | `6dbca48482ecb7a77deaf00b6aadedf3` | `src/visual_teleop/tools/benchmark.py` |
| `src/visual_teleop/tools/c922_live.py` | `cc289cc86c8b223115da568c20eaf066` | `src/visual_teleop/tools/c922_live.py` |
| `src/visual_teleop/tools/calib_right.py` | `b182397e1c3ca94d4ec372b78f03a084` | `src/visual_teleop/tools/calib_right.py` |
| `src/visual_teleop/tools/calib_tracker.py` | `34b8d319f58845ce0ed6123a909e4ae4` | `src/visual_teleop/tools/calib_tracker.py` |
| `src/visual_teleop/tools/cam_preview.py` | `b942c4596dc698d16c94ddb39a5e6412` | `src/visual_teleop/tools/cam_preview.py` |
| `src/visual_teleop/tools/fingertip_node.py` | `1807a9331d96e38c1ae6f4fa1f177046` | `src/visual_teleop/tools/fingertip_node.py` |
| `src/visual_teleop/tools/imu_bringup.py` | `99f62c277c3b199d6517dac92f9da1bb` | `src/visual_teleop/tools/imu_bringup.py` |
| `src/visual_teleop/tools/jaw_capture.py` | `ce5db74b49d12febc3ac63371e6d90a4` | `src/visual_teleop/tools/jaw_capture.py` |
| `src/visual_teleop/tools/jaw_finetune.py` | `0eb1d13d46b87e5e1c51e0d2ee96939c` | `src/visual_teleop/tools/jaw_finetune.py` |
| `src/visual_teleop/tools/jaw_train.py` | `75fb269ec6c7c335dc6c83cdbb8bb47d` | `src/visual_teleop/tools/jaw_train.py` |
| `src/visual_teleop/tools/joint_step_test.py` | `30c719955bc06ed2b1570f33d1e371ee` | `src/visual_teleop/tools/joint_step_test.py` |
| `src/visual_teleop/tools/kpt_click.py` | `c862d5225f3018e6bf9d17a3e7128bb7` | `src/visual_teleop/tools/kpt_click.py` |
| `src/visual_teleop/tools/live_teleop.py` | `471e8de7087130df66bdedf91783d155` | `src/visual_teleop/tools/live_teleop.py` |
| `src/visual_teleop/tools/mock_tracker.py` | `48843346dba659b217f0cc69199f8197` | `src/visual_teleop/tools/mock_tracker.py` |
| `src/visual_teleop/tools/orbbec_res_probe.py` | `9c10dc1d12c510d99f423459252414e9` | `src/visual_teleop/tools/orbbec_res_probe.py` |
| `src/visual_teleop/tools/pose_report.py` | `a73fb3c8ec0f08205dfba28c7192c97d` | `src/visual_teleop/tools/pose_report.py` |
| `src/visual_teleop/tools/sensor_hub.py` | `5ebe70ce7682d302a92bf4130122aa8a` | `src/visual_teleop/tools/sensor_hub.py` |
| `src/visual_teleop/tools/stage0_replay.py` | `8886a78d0ce3474c98846453f01f9f68` | `src/visual_teleop/tools/stage0_replay.py` |
| `src/visual_teleop/tools/stereo_imu_record.py` | `9538ca7f6075dda4f76bfe62ab226736` | `src/visual_teleop/tools/stereo_imu_record.py` |
| `src/visual_teleop/tools/stereo_preview.py` | `9266643a0673ebc49cf89413b153082e` | `src/visual_teleop/tools/stereo_preview.py` |
| `src/visual_teleop/tools/stereo_tip_calib.py` | `2bbc43093024f8268390dd6869725e96` | `src/visual_teleop/tools/stereo_tip_calib.py` |
| `src/visual_teleop/tools/tcp_check.py` | `c68b285963e8355ec6cd05fba4ccab84` | `src/visual_teleop/tools/tcp_check.py` |
| `src/visual_teleop/tools/teacher_record.py` | `5378be1849d0c946909517fd803aed1e` | `src/visual_teleop/tools/teacher_record.py` |
| `src/visual_teleop/tools/trackcam_imu_calib.py` | `7e2e7bb8cc69245f212b12b9a52522e3` | `src/visual_teleop/tools/trackcam_imu_calib.py` |
| `src/visual_teleop/tools/tracker_guide.py` | `f3740a75b4e70c66941760e9e827e997` | `src/visual_teleop/tools/tracker_guide.py` |
| `src/visual_teleop/tools/wrist_frontend_check.py` | `9d80fd1735c1cc2ddb84efca3833078c` | `src/visual_teleop/tools/wrist_frontend_check.py` |
| `src/visual_teleop/ui/__init__.py` | `d41d8cd98f00b204e9800998ecf8427e` | `legacy/laundry_collector/preprocess/__init__.py` |
| `src/visual_teleop/ui/teleop_dashboard.py` | `cb91e3e77b140439a14c69309dc3d717` | `src/visual_teleop/ui/teleop_dashboard.py` |
| `src/visual_teleop/vision/__init__.py` | `d41d8cd98f00b204e9800998ecf8427e` | `legacy/laundry_collector/preprocess/__init__.py` |
| `src/visual_teleop/vision/airslam_live.py` | `cf5c973dc8dbbefad7ea47f3790bb03e` | `src/visual_teleop/vision/airslam_live.py` |
| `src/visual_teleop/vision/confidence.py` | `1a1bfa96fbd62a9c6d8e274a2facf629` | `src/visual_teleop/vision/confidence.py` |
| `src/visual_teleop/vision/kpt_translation.py` | `21c399249ae40b0a80f7704e1d4fe549` | `src/visual_teleop/vision/kpt_translation.py` |
| `src/visual_teleop/vision/red_blobs.py` | `0c8ea4fe51ac62c789338bf4ba927388` | `src/visual_teleop/vision/red_blobs.py` |
| `src/visual_teleop/vision/residual_monitor.py` | `e1a724aa853d7b2fec2380ace8cba8cb` | `src/visual_teleop/vision/residual_monitor.py` |
| `src/visual_teleop/vision/rgbd_pose.py` | `6a23724979190633663dfce8bdb47b08` | `src/visual_teleop/vision/rgbd_pose.py` |
| `src/visual_teleop/vision/tag_camera.py` | `8248e898912393bdaf8e49856ba0eb14` | `src/visual_teleop/vision/tag_camera.py` |
| `src/visual_teleop/vision/tag_pose.py` | `24175791db5fe2a135aa22500e063b23` | `src/visual_teleop/vision/tag_pose.py` |
| `src/visual_teleop/vision/tape_pose.py` | `ddfb56ef09b8d68d737772bd386d0941` | `src/visual_teleop/vision/tape_pose.py` |
| `src/visual_teleop/vision/tracker6dof.py` | `bf40d7461eded06f6176f78e42f5814a` | `src/visual_teleop/vision/tracker6dof.py` |
| `src/visual_teleop/vision/umi_kpt.py` | `1a00019b241bce5b25e6881e6138d5d7` | `src/visual_teleop/vision/umi_kpt.py` |
| `src/visual_teleop/vision/umi_pose.py` | `06513cc32a99649d897a486e348dff08` | `src/visual_teleop/vision/umi_pose.py` |
| `tests/experimental/vteleop/__init__.py` | `d41d8cd98f00b204e9800998ecf8427e` | `legacy/laundry_collector/preprocess/__init__.py` |

## Kept in cube (cube work, or unsure) — listed for review

| cube path | why kept |
|---|---|
| `firmware/xiao_imu/` | laundry-era XIAO IMU firmware (identical copy in laundry-folding `hardware/umihand/firmware/xiao_imu/`); kept because `firmware/` is the shared HandUMI firmware folder and `firmware/teensy_imu/PROTOCOL.md` is claim evidence |
| `configs/calibration/{stereo_trackcam*,trackcam_*,jaw_cls_v001.npz,umi_tag_rig_v001.json,robot_cam_v001.json,imu_noise_xiao_right_v001.yaml}` | laundry-era calibrations (some identical in laundry-folding, stereo_trackcam*/trackcam_imu*/jaw_cls only in cube); kept as part of the versioned calibration archive; no remaining cube code reads them except `src/ego_cart20/sources/jaw_visual.py` (jaw_cls) |
| `configs/handumi/tasks.yaml` (UMIpilot / UMI100 / UMIalign entries), `configs/handumi/hardware_handumi_rgbd*.yaml` | collector task/hardware profiles of `handumi_collector` (cube package) used for the laundry takes |
| `src/ego_cart20/sources/{hfold_export,jaw_visual}.py`, `src/ego_cart20/umi_fusion/`, `src/ego_cart20/robotize/` | part of the `ego_cart20` package (cube Gen-2); some modules served the laundry data (T-shirt spec, visual jaw, robotize imports ~/laundry_system) |
| `src/handumi_collector` UMI100 metadata panel (garment_id / initial_state) | collector feature, cube package |
| `hardware/README.md` section "Changes after 2026-10-12" | state of the shared HandUMI units; dead links repointed to garment |
| `docs/experimental/ego_teleop/VISUAL_TELEOP_MVP_GAP.md` | ego_teleop design doc (mentions laundry stages) |
| `scripts/orbbec_alone_test.py`, `scripts/orbbec_combo_test.py`, `scripts/imu_bridge_test.py` | Orbbec / IMU bench tools from the 10-12 era; hardware diagnostics shared with the collector |
