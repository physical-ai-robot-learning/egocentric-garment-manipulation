"""garment_pipeline: the offline pipeline that turned the T-shirt flattening demonstrations into fused trajectories,
quality grades, training labels, a LeRobot dataset, a DiT policy and its deploy / evaluation tools.

Subpackages (one per stage group; see README.md in this folder for the 12-stage map and what each stage needs):

* ``garment_pipeline.fusion_stack``    CoTracker + tag + IMU pose fusion, gates, trims, quality grades, QA renders
* ``garment_pipeline.fgo_tools``       data loaders and factor-graph helpers shared by every stage
* ``garment_pipeline.labels_dataset``  robotize / IK labels, IK QA, idle trim, LeRobot export
* ``garment_pipeline.training``        DiT trainer (jaw BCE head, masks, per-episode weights) and its configs
* ``garment_pipeline.eval_deploy``     offline policy evaluation, receding-horizon deploy core + UI, X-VLA server, tools

Entry points run as modules, e.g. ``python -m garment_pipeline.labels_dataset.tag_rig_unsplit --help``. File locations
come from ``egogarment.paths`` (environment variables ``EGO_DATA_ROOT``, ``EGO_TRAIN_ROOT``, ``EGO_CKPT_ROOT``, ...).
"""
