# Environments

The project uses four Python environments. Only the first is fully specified by `pyproject.toml`; the training and
deploy environments are recorded as pinned requirement files (versions read from the environments that produced the
reported results on 2026-11-02; a fresh install from them has not been tested).

| environment | where it ran | what it runs | how to build |
|---|---|---|---|
| core | any machine, Python ≥ 3.12 | tests, `src/laundry_reference` reference implementation, claims check, the CPU stages of the pipeline packages | `uv sync` or `pip install -e ".[test]"` (+ dependency groups `vision`, `robot`, `collector` in `pyproject.toml`) |
| training | Linux, RTX 4090, CUDA 12.8, Python 3.13 | DiT fine-tunes (`src/laundry_training`), offline evaluation | `pip install -r configs/laundry/envs/requirements-train.txt --extra-index-url https://download.pytorch.org/whl/cu128` then `pip install -e .` |
| deploy | macOS arm64 (MPS), Python 3.13 | tag-rig UI and receding-horizon deploy (`src/laundry_deploy`), height calibration | `pip install -r configs/laundry/envs/requirements-deploy.txt` then `pip install -e .` |
| robot bridge | the robot Mac, Python 3.12 | `python -m laundry_deploy.robot_service.mit_bridge` (HTTP bridge to the reBot motors; the runs used the same code as `robot_service_mit.py` of the old layout) | not reproducible from this repository: it imports the robot-PC driver packages (`lerobot_plugin_rebot`, the in-house feed-forward and GUI adapter packages, `usb_arm_ports`), which are not included; it was run with LeRobot 0.5.1, motorbridge 0.5.6, Flask 3.1.3, NumPy 2.2.6 |

Stages that ran on rented GPUs (CoTracker3 point tracking, the jaw classifier) used separate cloud images whose exact
package versions were not recorded.

The X-VLA policies served during some robot runs used the X-VLA serving setup described in
`src/laundry_deploy/xvla_server.py`; its environment is not pinned here.
