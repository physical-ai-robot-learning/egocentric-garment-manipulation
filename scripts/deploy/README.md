# deploy

Robot side: the reBot MIT-mode HTTP bridge and its gripper / joint zero tools.

**Status:** current

## Inputs → Outputs
Joint / gripper commands over HTTP (port 8021) → two reBot B601-DM arms over CAN; `/status` reports the measured state.

## Entry points
```bash
PYTHONPATH=src python -m visual_teleop.tools.bridge_start --sides left,right --go   # start / restart with configs/robot/rebot_mit_deploy_v001.json
python scripts/deploy/robot_bridge/check_grippers.py <ports…>                       # run with the vendor plugin's Python
```
The policy deploy loop itself is `src/garment_pipeline/eval_deploy/act_tagrig_deploy.py`.

## Layout
- `robot_bridge/` — `robot_service_mit.py` and the zero / read tools ([README](robot_bridge/README.md))

## Provenance
Moved from `robot_bridge/` in the 11-01 layout refactor. Needs the vendor reBot LeRobot plugin, which is not included.
