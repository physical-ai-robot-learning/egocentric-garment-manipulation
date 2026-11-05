"""Every runnable laundry_tracking module answers --help (python -m <module> --help exits 0) without touching data."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_SRC = Path(__file__).resolve().parents[2]
CLI_MODULES = (
    "laundry_tracking.new_take_prep", "laundry_tracking.tag_rotations", "laundry_tracking.orbbec_tracker", "laundry_tracking.orbbec_candidate",
    "laundry_tracking.edge_gap_trim", "laundry_tracking.tighten_trim", "laundry_tracking.whole_demo_stitch",
    "laundry_tracking.cotrack.make_pack", "laundry_tracking.cotrack.run_cotracker", "laundry_tracking.cotrack.point_solver",
    "laundry_tracking.cotrack.gate", "laundry_tracking.cotrack.episode_quality", "laundry_tracking.cotrack.out_of_view",
    "laundry_tracking.fusion.trajectory_fusion", "laundry_tracking.fusion.fusion_metrics", "laundry_tracking.fusion.make_derived",
    "laundry_tracking.fusion.render_qa", "laundry_tracking.fusion.render_sources",
    "laundry_tracking.calib.robot_cam_from_anchors", "laundry_tracking.calib.robot_cam_solve", "laundry_tracking.calib.tag_rig_pivot",
    "laundry_tracking.calib.tag_rig_pivot_joint", "laundry_tracking.calib.tag_rig_solve", "laundry_tracking.calib.tag_rig_touch_tcp",
    "laundry_tracking.calib.umi_side_check", "laundry_tracking.calib.umi_tcp_traj",
)


@pytest.mark.parametrize("module", CLI_MODULES)
def test_module_help_exits_zero(module):
    pytest.importorskip("cv2")
    tProc = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, text=True, timeout=120,
                           env={**os.environ, "PYTHONPATH": str(REPO_SRC)})
    assert tProc.returncode == 0, tProc.stderr[-2000:]
    assert "usage:" in tProc.stdout


def test_anchor_pnp_camera_height_picks_the_solution_above_the_table():
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    from laundry_tracking.calib.robot_cam_from_anchors import _cameraHeight
    # camera 1.1 m above the base origin looking down (optical axis -z_base): t = -R c
    tRm = R.from_euler("x", 180, degrees=True).as_matrix()
    tC = np.array([0.4, 0.0, 1.1])
    assert abs(_cameraHeight(R.from_matrix(tRm).as_rotvec(), -tRm @ tC) - 1.1) < 1e-9
