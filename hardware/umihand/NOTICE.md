# NOTICE

This repository contains material derived from **HandUMI hardware**, https://github.com/murobotics-ai/handumi-hw,
Copyright 2026 BrikHMP18, licensed under the Apache License, Version 2.0 (`LICENSE`), base commit
`e58de33b9a88bd208e1a40e52ea575eaf68963db`.

- `cad/STEP/` and `bom/` are unmodified copies of the upstream files.
- The STL meshes in `print/{left,right,tips}/stl/` are the upstream meshes **translated** so all coordinates are positive;
  the geometry is unchanged. The 3MF plate projects were generated from them by Beyond Honeycomb (2026-08-26).
- `mods/` contains parts **modified** by Beyond Honeycomb in 2026-09 from the upstream `left_controller_support`,
  `camera_mount` and `servo_controller_cover` (see `mods/README.md`).
- `firmware/` (wrist IMU) is original Beyond Honeycomb work, not part of upstream.
