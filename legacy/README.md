# legacy/ — earlier and dropped approaches

Kept for reference; not used by the current pipeline. Why each was dropped is described in the technical report.

| folder | what | period | status |
|---|---|---|---|
| [`from_cube/`](from_cube/README.md) | laundry-era files that existed only in the cube repository (UMIalign drivers, parity/Kalibr helpers, visual-teleop tests, 10-20 update notes) | 10-13 → 10-20 | records, moved 11-01 |
| `laundry_collector/`, `tests/` | bare-hand collector + virtual gripper renderer | 10-08 | mock-tested, parked |
| `configs/laundry/` | config files of `laundry_collector/` (read as `legacy/configs/laundry` by its `config.py`; moved from `configs/laundry/` 11-01) | 10-08 | parked with the collector |
| `configs/handumi/*.bak*` | backup copies of HandUMI collector / task configs taken before edits (`.bak_pre_<change>`); the live files stay in `configs/handumi/` (moved here in refactor v3, 11-01) | – | backups only |
| `umialign_pipeline/` | MASt3R + Orbbec body tracking + IMU smoother and exporter (UMIalign data) | 10-13 … 10-16 | labels only 4–6 cm accurate → replaced by the AprilTag rig |
| `scripts/umi_fusion/` | red-tape jaw detection, QA, Orbbec tag check/gate, MASt3R restart runner (old) | 10-12 … 10-15 | tape removed 10-20 |
| `scripts/handumi_*` | Orbbec body / canonical tracking of the UMI | 10-12 | replaced by tags |
| `scripts/kalibr_*`, `dpvo_eval_metrics.py`, `m3_imu_eval_metrics.py` | camera–IMU calibration export, VIO/IMU evaluation | 10-15 … 10-16 | stereo dropped 10-16 |
| `scripts/stereo_vio_and_finger/` | stereo trackcam / Basalt / OpenVINS / finger-pinch experiments | 10-13 … 10-16 | dropped |
| `scripts/mast3r_gapfill/` | MASt3R wrist-SLAM gap filling: frame prep, hand-eye, segment fusion, SLAM-vs-tag eval, `umi_pipeline.sh` | 10-17 … 10-23 | breaks at grasps (19–48 % usable) → replaced by CoTracker + marker fusion; moved here 10-27 |
| `slam/mast3r/` | MASt3R-SLAM restart runner (Ray / GPU) | 10-13 … 10-23 | same; moved here 10-27 |
| `orbslam3/` | ORB-SLAM3 mono-inertial evaluation (our scripts only, no GPL source) | 10-21 | scale diverged → rejected |
