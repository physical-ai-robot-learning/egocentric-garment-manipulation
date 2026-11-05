"""laundry_tracking: the fixed-camera pose stage of the laundry pipeline (recorded takes -> metric UMI TCP trajectories).

Orbbec tracker -> CoTracker packs / point solver / gates (cotrack) -> trajectory fusion (fusion) -> derived roots for the dataset
builder; plus the SLAM<->IMU consistency and whole-demo stitch of the wrist-SLAM branch, the factor-graph loaders (fgo) and the
calibration tools (calib).  Working files live under laundry_common.paths.FUSION_WORK (see workdirs.py).
"""
