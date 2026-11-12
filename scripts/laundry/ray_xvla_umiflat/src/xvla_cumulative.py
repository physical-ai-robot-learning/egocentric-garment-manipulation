#!/usr/bin/env python3
"""[2026-11-11 user] cumulative X-VLA UMI-flatten set for the daily continual schedule (ray_jobs/xvla_umiflat/src/continual.py).

Aggregates per-day X-VLA split roots (each made by flat_to_xvla_relcart20.py --stats-from <frozen stats root>) into ONE LeRobot
root, in the given order, and writes
  meta/episode_day.json  {episode_index: "YYYYMMDD"}  (continual.py: recent-day sampling)
  meta/stats.json        action / observation.state / aux.loss_mask stats copied from --stats-from (FROZEN normalisation: a
                         continued checkpoint must see the same mean / std every day; aggregate_datasets would recompute them)
  meta/cumulative.json   the parts, their days and episode counts
usage: python datasets/xvla_cumulative.py --out ROOT --stats-from FROZEN_ROOT --part ROOT:YYYYMMDD [--part ...]"""
from __future__ import annotations

import argparse
import json
import pathlib

FROZEN = ("action", "observation.state", "aux.loss_mask")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--stats-from", required=True)
    ap.add_argument("--part", action="append", required=True, help="ROOT:YYYYMMDD, in order")
    a = ap.parse_args()
    from lerobot.datasets.aggregate import aggregate_datasets

    tOut = pathlib.Path(a.out); assert not tOut.exists(), f"{tOut} exists"
    tParts = [(pathlib.Path(p.rsplit(":", 1)[0]), p.rsplit(":", 1)[1]) for p in a.part]
    tDay, tInfo, e0 = {}, [], 0
    for r, d in tParts:
        n = json.loads((r / "meta/info.json").read_text())["total_episodes"]
        assert (r / "videos").exists(), f"{r} has no videos"
        tDay.update({str(e0 + i): d for i in range(n)}); tInfo.append(dict(root=str(r), day=d, episodes=n, first_episode=e0)); e0 += n
    aggregate_datasets(repo_ids=[f"local/{r.name}" for r, _ in tParts], aggr_repo_id=f"local/{tOut.name}",
                       roots=[r for r, _ in tParts], aggr_root=tOut)
    tInfoOut = json.loads((tOut / "meta/info.json").read_text())
    assert tInfoOut["total_episodes"] == e0, f"aggregate has {tInfoOut['total_episodes']} episodes, parts {e0}"
    (tOut / "meta/episode_day.json").write_text(json.dumps(tDay))
    st = json.loads((tOut / "meta/stats.json").read_text())
    tFrozen = json.loads((pathlib.Path(a.stats_from) / "meta/stats.json").read_text())
    for k in FROZEN: st[k] = tFrozen[k]
    (tOut / "meta/stats.json").write_text(json.dumps(st))
    (tOut / "meta/cumulative.json").write_text(json.dumps(dict(stats_from=a.stats_from, parts=tInfo), indent=1))
    print(json.dumps(dict(out=str(tOut), episodes=e0, frames=tInfoOut["total_frames"],
                          days={d: sum(1 for v in tDay.values() if v == d) for d in sorted(set(tDay.values()))})), flush=True)


if __name__ == "__main__":
    main()
