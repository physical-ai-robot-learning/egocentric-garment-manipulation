"""MASt3R-SLAM preprocessing command (spec §13).

  python -m laundry_collector.preprocess.preprocess_slam --episode <dir> --side left|right|both stage
  python -m laundry_collector.preprocess.preprocess_slam --episode <dir> --side both submit     # 4090 via Ray (stage first)
  python -m laundry_collector.preprocess.preprocess_slam --episode <dir> --side both collect    # -> <episode>/processed/
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import slam

ap = argparse.ArgumentParser()
ap.add_argument("--episode", required=True, type=Path, nargs="+")
ap.add_argument("--side", default="both", choices=["left", "right", "both"])
ap.add_argument("--stage-root", type=Path, default=Path("~/datasets/laundry_slam_stage").expanduser())
ap.add_argument("action", choices=["stage", "submit", "collect"])
a = ap.parse_args()
sides = ["left", "right"] if a.side == "both" else [a.side]
if a.action == "stage":
    for ep in a.episode:
        for s in sides:
            print("staged", slam.stage(ep, s, a.stage_root))
elif a.action == "submit":
    dirs = [a.stage_root / slam.tag_of(ep, s) for ep in a.episode for s in sides]
    missing = [d for d in dirs if not (d / "raw_video.mp4").exists()]
    if missing:
        raise SystemExit(f"stage first: {missing}")
    print("submitted", slam.submit(dirs, f"laundry-slam-{time.strftime('%m%d%H%M%S')}"))
else:
    for ep in a.episode:
        for s in sides:
            print("collected", slam.collect(ep, s, a.stage_root))
