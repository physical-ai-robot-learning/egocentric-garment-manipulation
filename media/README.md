# Media

Clips and photos of the real setup, cut from the project's own recordings by the author. Take / run numbers in the
file names are recording timestamps (HHMMSS). "human time" = shown at the demonstrator's speed (33 s demo), "robot time" =
shown at the replay speed (94 s at 0.35×).

| file | what | used in |
|---|---|---|
| `human_vs_robot_replay_large.gif` | human demonstration vs. robot replay of the converted trajectory, same fixed camera, human speed; the bar shows the robot jaw commands | [README](../README.md) (top) |
| `replay3_200335_robot_time_94s_large.gif` | human demo (left) vs. two-arm replay (right), run 200335, same fixed camera, one clock, robot speed; the jaws catch the shirt edge and drag it | [README §7](../README.md#7-results) |
| `replay2_195745_human_time_33s_large.gif` | replay run 195745 (take 171057 demo 2), human vs. robot, human speed; the shirt lay 15–20 cm farther away, so the jaws closed on the empty table | [docs/details.md §5](../docs/details.md) |
| `replay1_195154_shirt_far_partial_human_time_33s_v2.{gif,mp4}` | replay run 195154, shirt farther than in the demo, partial run, human speed | – |
| `replay2_195745_shirt_far_human_time_33s_v2.{mp4}`, `replay2_195745_shirt_far_robot_time_94s_v2.{gif,mp4}` | replay run 195745, shirt farther than in the demo, at human and at robot speed | – |
| `replay3_200335_shirt_moved_human_time_33s_v2.{gif,mp4}`, `replay3_200335_shirt_moved_robot_time_94s_v2.mp4` | replay run 200335, shirt moved, at human and at robot speed | – |
| `trajectory_reconstruction_large.gif` | reconstructed jaw-tip paths of both hands (replay era), coloured by source (markers + depth / MASt3R-SLAM filler / IMU), projected into the camera and in 3D | [README §5](../README.md#pose-reconstruction-and-validation-stages-24), [docs/details.md](../docs/details.md) |
| `flatten_gold_trajectory_sources.gif` | one GOLD flattening demonstration (Oct 26), jaw-tip paths coloured by pose source (marker / point tracking / IMU) | [README §8](../README.md#from-replay-to-policy-training-data-october-2627), [docs/details.md](../docs/details.md) |
| `flatten_fusion_qa_gold.gif` | per-demo fusion QA video of a GOLD demo: fused jaw tips, hidden-marker estimate, marker, raw CoTracker points | [docs/details.md](../docs/details.md) |
| `flatten_fusion_qa_silver_redborder.gif` | QA video of a SILVER demo; the frame turns red while the hidden-marker estimate is more than 10 cm off | [docs/details.md](../docs/details.md) |
| `public_data_softfold_iphumi_primeumi.gif` | one episode each of the pretraining sources: Soft-Fold, iPhUMI, PrimeBot UMI | [docs/details.md](../docs/details.md) |
| `deploy_dit_dryrun_ui.gif` | dry run of a DiT checkpoint in the deploy UI (no command sent): predicted 32-step jaw-tip paths, top view, wrist cameras, live checks | [docs/details.md](../docs/details.md) |
| `pipeline.png` | pipeline overview figure (record → pose → … ) | [docs/details.md](../docs/details.md) |
| `umi_hands.jpg` | the two hand-worn grippers (HandUMI build) with their fiducial markers, photo from 10-21 | [README §9](../README.md#9-hardware-and-attribution) |
| `umi_wrist_fisheye_right.jpg` | a frame from the right UMI wrist fisheye camera: the jaws, the cloth and a table tag in view | – |
| `start_markers.jpg` | (a) the two start markers (IDs 400 / 401) with their measured robot coordinates, (b) a take starting with the grippers there | [README §5](../README.md#stage-0--one-world-frame-for-human-and-robot) |
| `world_frame_start_markers.jpg` | the same two panels without the IDs and coordinates | (copy in `report/figures/`) |
| `orbbec_global_view_demo2_start.jpg` | the fixed Orbbec camera's view at the start of a demo: both grippers on the start markers, shirt laid out, both arms in view | – |

Raw recordings are not included. `media/umi_hand_current.jpg` of the former `egocentric-laundry-folding` repository was
byte-identical to `umi_hands.jpg` and was not kept.
| `pinching_cg_gold_trajectory_sources.gif` / `.mp4` | three GOLD left-hand pinching demonstrations (11-01), Cg-fused jaw-tip paths coloured by pose source; rendered with `garment_pipeline.fusion_stack.orbbec.v2b.fusion.render_traj_sources` |
