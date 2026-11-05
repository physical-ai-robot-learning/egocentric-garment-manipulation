import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from laundry_collector.preprocess.gripper_frame import GripperConfig, gripper_from_keypoints
from laundry_collector.preprocess.smoothing import smooth_trajectory
from tests.laundry.synth import hand_local, place, pose

CFG = GripperConfig()


@pytest.mark.parametrize("side", ["right", "left"])
@pytest.mark.parametrize("rv", [(0, 0, 0), (0.4, -0.3, 0.2), (2.0, 0.5, -1.0)])
@pytest.mark.parametrize("mode", ["palm_forward", "thumb_index"])
def test_same_physical_convention_both_hands(side, rv, mode):
    T = pose(rv); k = place(hand_local(0.04, side), T)
    g = gripper_from_keypoints(k, None, side, GripperConfig(orientation_source=mode))
    assert g["valid"] and g["rot_valid"]
    Rg, Rh = g["R"], T[:3, :3]                                         # hand-local axes: x radial(R hand), y fingers, z out of palm
    assert np.isclose(np.linalg.det(Rg), 1) and np.allclose(Rg.T @ Rg, np.eye(3), atol=1e-9)
    assert np.dot(Rg[:, 2], Rh[:, 2]) > 0.8                            # +z out of the palm, both hands
    assert np.dot(Rg[:, 1], Rh[:, 1]) > 0.99                           # +y towards the fingers, both hands
    if mode == "thumb_index":
        assert abs(np.dot(Rg[:, 0], (k[8] - k[4]) / np.linalg.norm(k[8] - k[4]))) > 1 - 1e-9   # jaw axis exactly along thumb-index
    else:
        assert np.allclose(Rg[:, 2], Rh[:, 2], atol=1e-9)              # palm normal exact


@pytest.mark.parametrize("side", ["right", "left"])
def test_jaw_tips_on_fingertips_with_1to1_mapping(side):
    cfg = GripperConfig(human_pinch_min_m=0.0, human_pinch_max_m=0.1, robot_jaw_min_m=0.0, robot_jaw_max_m=0.1, orientation_source="thumb_index")
    k = place(hand_local(0.05, side), pose((0.3, 0.2, -0.1)))
    g = gripper_from_keypoints(k, None, side, cfg)
    tips = [g["p"] + s_ * 0.5 * g["jaw_m"] * g["R"][:, 0] for s_ in (-1, 1)]
    d = sorted(min(np.linalg.norm(t - k[j]) for t in tips) for j in (4, 8))
    assert d[-1] < 1e-9


def test_tcp_is_midpoint_and_jaw_mapping():
    k = place(hand_local(0.0575), pose())
    g = gripper_from_keypoints(k, None, "right", CFG)
    assert np.allclose(g["p"], 0.5 * (k[4] + k[8]))
    assert np.isclose(g["pinch_m"], 0.0575) and np.isclose(g["jaw_n"], 0.5) and np.isclose(g["jaw_m"], 0.04)
    assert gripper_from_keypoints(place(hand_local(0.2), pose()), None, "right", CFG)["jaw_n"] == 1.0


def test_degenerate_cases_never_nan():
    k = place(hand_local(0.004), pose())                                # pinch closed: axis undefined
    g = gripper_from_keypoints(k, None, "right", CFG)
    assert g["valid"] and not g["rot_valid"] and g["reason"] == "pinch_closed_axis_undefined"
    k2 = k.copy(); k2[[5, 17]] = k2[0]                                  # palm collapsed
    assert gripper_from_keypoints(k2, None, "right", CFG)["reason"] == "palm_degenerate"
    k3 = k.copy(); k3[4] = np.nan
    assert not gripper_from_keypoints(k3, None, "right", CFG)["valid"]
    conf = np.ones(21); conf[8] = 0.1
    assert gripper_from_keypoints(place(hand_local(), pose()), conf, "right", CFG)["reason"] == "low_confidence"


def test_smoothing_fills_gaps_without_nan_and_reduces_jitter():
    n = 120; rng = np.random.default_rng(0)
    p = np.column_stack([np.linspace(0, 0.1, n), np.zeros(n), np.full(n, 0.2)]) + rng.normal(0, 0.003, (n, 3))
    q = R.from_rotvec(np.column_stack([np.linspace(0, 1, n), np.zeros(n), np.zeros(n)]) + rng.normal(0, 0.03, (n, 3))).as_quat()
    jaw = np.full(n, 0.04) + rng.normal(0, 0.003, n)
    pv = np.ones(n, bool); pv[40:50] = False; rv = pv.copy(); rv[0:5] = False
    ps, qs, js = smooth_trajectory(p, q, jaw, pv, rv)
    assert np.isfinite(ps).all() and np.isfinite(qs).all() and np.isfinite(js).all()
    assert np.allclose(np.linalg.norm(qs, axis=1), 1)
    assert np.abs(np.diff(ps, axis=0)).mean() < np.abs(np.diff(p, axis=0)).mean() / 2
