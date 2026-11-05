import json

import numpy as np
import pytest
import yaml

from laundry_collector.config import CAL_DIR, load_config


@pytest.fixture
def mock_cfg(tmp_path):
    cfg = load_config(mock=True)
    cfg.collector.dataset_root = str(tmp_path / "ds")
    p = cfg.collector.protocol; p.init_hold_s, p.end_hold_s, p.min_manip_s = 1.0, 0.5, 0.3
    return cfg


@pytest.fixture
def recorded_episode(mock_cfg):
    from laundry_collector.collect import headless
    msg = headless(mock_cfg, 1.0, keep=True)
    ep = sorted(mock_cfg.dataset_dir.glob("episode_*"))[-1]
    # mock cameras have no lens calibration: give both wrists the real fisheye intrinsics scaled to the mock 320x240
    for side in ("left", "right"):
        d = yaml.safe_load(sorted(CAL_DIR.glob(f"fisheye_{side}_v*.yaml"))[-1].read_text())
        K = np.asarray(d["K"], float); w, h = d["image_size"] if isinstance(d["image_size"], list) else (1920, 1080)
        K[0] *= 320 / w; K[1] *= 240 / h
        (ep / "calibration" / f"{side}_intrinsics.json").write_text(json.dumps(dict(d, K=K.tolist(), source="test-scaled", calibrated=True)))
    return ep
