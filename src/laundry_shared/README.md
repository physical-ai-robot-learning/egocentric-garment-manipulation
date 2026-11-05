# laundry_shared

Copies of the shared modules the laundry packages import (recorder, tag-rig pose, IK client, contracts), kept in their
original code style so that they stay comparable with the source they came from.

**Status:** current, imported by the pipeline; not restyled on purpose. Only the files the pipeline imports are kept.

## Layout
| path | what |
|---|---|
| `visual_teleop/` | recorder (`tools/tag_rig_record`, wrist-camera jaw capture `tools/umi_jaw`), reBot IK / limits / safety, replay planner |
| `rebot_prep/parity/` | marker-rig pose code (tag-rig corners and fusion) |
| `handumi_collector/`, `ego_collector/`, `ego_teleop/` | device, camera, tracking and transform helpers used by the recorder and calibration |
| `contracts/`, `xvla_core.py`, `umi_to_lerobot.py` | action / observation contracts, X-VLA helpers, LeRobot export helper |

Each sub-package has its own README. Recording entry point:
```bash
sudo .venv/bin/python -m laundry_shared.visual_teleop.tools.tag_rig_record --proto task --wrist
```

## Provenance
Copied in the 11-04 refactor from `src/{visual_teleop,handumi_collector,ego_collector,ego_teleop,rebot_prep,contracts}`,
`xvla_core.py`, `umi_to_lerobot.py` (97 files the laundry code imports).
