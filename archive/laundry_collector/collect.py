"""Entry point.

  python -m laundry_collector.collect                     # UI, hardware profile laundry_v1
  python -m laundry_collector.collect --mock              # UI on synthetic cameras
  python -m laundry_collector.collect --mock --headless-smoke 2   # one scripted episode (init 2 s ... ), no UI
"""
from __future__ import annotations

import argparse
import logging
import time

from .config import load_config
from .session import CollectorSession, State


def headless(cfg, seconds: float, keep: bool = True, drop_every: int = 0) -> str:
    s = CollectorSession(cfg, mock_drop_every=drop_every)
    s.open()
    try:
        time.sleep(0.5)
        assert s.start(), s.message
        t0 = time.time()
        while s.state != State.REVIEW:
            s.tick(); time.sleep(0.02)
            if s.state == State.MANIPULATION and time.time() - t0 > cfg.collector.protocol.init_hold_s + seconds:
                s.stop()
        print(s.last_report_txt)
        if keep:
            s.save(dict(success=True, garment_id="shirt_01", initial_state="flat", notes="headless smoke"))
        else:
            s.cancel()
        print(s.message)
        return s.message
    finally:
        s.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="laundry_v1")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--dataset", default=None, help="override collector.yaml dataset (e.g. laundry_bringup)")
    ap.add_argument("--headless-smoke", type=float, default=0, help="record one episode with this many manipulation seconds, no UI")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = load_config(a.profile, mock=a.mock)
    if a.dataset:
        cfg.collector.dataset = a.dataset
    if a.headless_smoke:
        headless(cfg, a.headless_smoke); return
    from .ui import run_ui
    run_ui(cfg)


if __name__ == "__main__":
    main()
