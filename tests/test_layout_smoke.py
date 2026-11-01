"""Smoke checks that need no hardware, GPU or data: packages import, resource paths resolve, and the
self-tests that ship inside the scripts still pass from their new locations."""
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("module", ["visual_teleop", "visual_teleop.config", "visual_teleop.robot.ik_pink", "rebot_prep.parity"])
def test_import(module):
    importlib.import_module(module)


def test_visual_teleop_config_files_exist():
    from visual_teleop import config
    assert config.DEFAULT_PATH.is_file() and config.CALIB_PATH.is_file()


@pytest.mark.parametrize("name", ["tcp_override_v001.json", "joint_offset_v001.json", "base_override_v001.json", "rebot_mit_deploy_v001.json"])
def test_robot_configs_resolve_from_package_root(name):
    # src/visual_teleop/robot/ik_pink.py and tools/bridge_start.py read these as parents[3] / "configs/robot/..."
    root = Path(importlib.import_module("visual_teleop.robot.ik_pink").__file__).resolve().parents[3]
    assert root == REPO and (root / "configs/robot" / name).is_file()


def _selftest(script):
    r = subprocess.run([sys.executable, str(REPO / script), "--synthetic"], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "PASS" in r.stdout


def test_replay_parity_synthetic():
    _selftest("scripts/evaluation/replay_parity.py")


def test_robot_cam_solve_synthetic():
    pytest.importorskip("cv2")
    _selftest("scripts/pose/robot_cam_solve.py")
