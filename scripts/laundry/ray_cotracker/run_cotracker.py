"""CoTracker3 ONLINE (constant memory, forward only) on every pack dir -> tracks.npz {tracks (T,N,2) px, vis (T,N)} (Ray job entry point).

Standalone copy of laundry_tracking.cotrack.run_cotracker: the Ray job ships only this folder (--working-dir .) and runs it with the
pyproject env (torch, numpy, pillow), so it imports nothing from the repository.  Queries are at the seed frames (first tracked frame
of each arm); frames before a query frame get NaN for that point.  Dirs that already have tracks.npz are skipped.
    uv run python run_cotracker.py <pack dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HUB_REPO = "facebookresearch/co-tracker"
HUB_MODEL = "cotracker3_online"


def trackPack(pack: Path) -> None:
    import torch
    from PIL import Image
    tModel = torch.hub.load(HUB_REPO, HUB_MODEL).cuda().eval()
    tStep = tModel.step
    for tDir in sorted(pack.iterdir()):
        if not (tDir / "meta.json").exists() or (tDir / "tracks.npz").exists():
            continue
        tMeta = json.loads((tDir / "meta.json").read_text())
        tFrames = sorted((tDir / "frames").glob("*.jpg"))
        tVideo = torch.from_numpy(np.stack([np.asarray(Image.open(tF).convert("RGB")) for tF in tFrames])).permute(0, 3, 1,
                2)[None].float()
        tQueries = torch.tensor(tMeta["queries"])[None].float().cuda()
        tT = tVideo.shape[1]
        with torch.no_grad():
            tModel(video_chunk=tVideo[:, : tStep * 2].cuda(), is_first_step=True, queries=tQueries)
            for tI in range(0, tT - tStep, tStep):
                tTracks, tVis = tModel(video_chunk=tVideo[:, tI: tI + tStep * 2].cuda())
        tTracks, tVis = tTracks[0].cpu().numpy()[:tT], tVis[0].cpu().numpy()[:tT]
        np.savez(tDir / "tracks.npz", tracks=tTracks, vis=tVis)
        print(tDir.name, tTracks.shape, float(tVis.mean()), flush=True)


def main(argv: list[str] | None = None) -> int:
    tAp = argparse.ArgumentParser(description="CoTracker3 online on every demo dir of a pack -> tracks.npz")
    tAp.add_argument("pack")
    trackPack(Path(tAp.parse_args(argv).pack))
    return 0


if __name__ == "__main__":
    sys.exit(main())
