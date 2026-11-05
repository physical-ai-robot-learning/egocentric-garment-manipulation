import json

import pandas as pd

from laundry_collector.collect import headless
from laundry_collector.qa import build_report


def test_episode_layout_metadata_and_sync(recorded_episode):
    ep = recorded_episode
    assert (ep / ".complete").exists() and not (ep / ".incomplete").exists()
    for s in ("global", "left_wrist", "right_wrist"):
        ts = pd.read_csv(ep / s / "timestamps.csv")
        assert {"video_frame", "frame_index", "camera_timestamp_ns", "host_timestamp_ns"} <= set(ts.columns)
        assert ts.host_timestamp_ns.is_monotonic_increasing and len(ts) > 40
    g = pd.read_csv(ep / "global" / "timestamps.csv")
    assert all((ep / "global" / "depth" / f).exists() for f in g.depth_file)
    for f in ("global_intrinsics.json", "left_intrinsics.json", "right_intrinsics.json", "world_config.json", "devices.json"):
        assert (ep / "calibration" / f).exists()
    md = json.loads((ep / "metadata.json").read_text())
    assert md["success"] is True and md["world_frame"] == "table" and md["collector_version"] == "v1"
    assert [p["phase"] for p in md["phases"]] == ["initialization", "manipulation", "end"]
    rep = json.loads((ep / "sync_report.json").read_text())
    assert rep["verdict"] == "PASS", rep["flags"]
    assert set(rep["skew_ms"]) == {"global-left", "global-right"}


def test_drops_detected_and_cancel_moves_to_discarded(mock_cfg):
    headless(mock_cfg, 1.0, keep=False, drop_every=10)
    d = sorted((mock_cfg.dataset_dir / "_discarded").glob("episode_*"))
    assert d, "cancelled episode must be kept under _discarded/"
    rep = build_report(d[-1], fps={"global": 30, "left_wrist": 30, "right_wrist": 30})
    assert all(rep["streams"][s]["dropped_by_index"] > 0 for s in rep["streams"])
    assert any("dropped" in f for f in rep["flags"])
