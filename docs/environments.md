# Environments

The project uses four Python environments. Only the first is fully specified by `pyproject.toml`; the training and
deploy environments are recorded as pinned requirement files (versions read from the environments that produced the
reported results on 2026-11-02; a fresh install from them has not been tested).

| environment | where it ran | what it runs | how to build |
|---|---|---|---|
| core | any machine, Python ≥ 3.10 | tests, `src/egogarment` reference implementation, claims check; with the `pipeline` extra the CPU stages of `src/garment_pipeline` | `pip install -e ".[test]"` (+ `.[pipeline]`, `.[vision]`) |
| training | Linux, RTX 4090, CUDA 12.8, Python 3.13 | DiT fine-tunes (`src/garment_pipeline/training`), offline evaluation | `pip install -r envs/requirements-train.txt --extra-index-url https://download.pytorch.org/whl/cu128` then `pip install -e .` |
| deploy | macOS arm64 (MPS), Python 3.13 | tag-rig UI and receding-horizon deploy (`src/garment_pipeline/eval_deploy`), height calibration | `pip install -r envs/requirements-deploy.txt` then `pip install -e .` |
| robot bridge | the robot Mac, Python 3.12 | `scripts/deploy/robot_bridge/robot_service_mit.py` (HTTP bridge to the reBot motors) | not reproducible from this repository: it imports the robot-PC driver packages (`lerobot_plugin_rebot`, the in-house feed-forward and GUI adapter packages, `usb_arm_ports`), which are not included; it was run with LeRobot 0.5.1, motorbridge 0.5.6, Flask 3.1.3, NumPy 2.2.6 |

Stages that ran on rented GPUs (CoTracker3 point tracking, the jaw classifier) used separate cloud images whose exact
package versions were not recorded.

The X-VLA policies served during some robot runs used the X-VLA serving setup described in
`src/garment_pipeline/eval_deploy/` (`xvla_server.py`); its environment is not pinned here.
