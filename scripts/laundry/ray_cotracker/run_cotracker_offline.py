"""11-04 option 2: CoTracker3 OFFLINE + backward_tracking (bidirectional) on every pack dir -> tracks.npz {tracks (T,N,2) px, vis (T,N)},
same contract as run_cotracker.py (online, forward only).  Backward tracking also fills frames BEFORE a query frame and re-links points
across short occlusions.  Standalone (Ray job folder, no repository import).
Environment: QCHUNK = queries per forward pass (default 256; memory grows with T x N, OOM at N=1340, T=441).
    uv run python run_cotracker_offline.py PACK_DIR [--force]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HUB_REPO = "facebookresearch/co-tracker"
HUB_MODEL = "cotracker3_offline"
QUERY_CHUNK_ENV = "QCHUNK"
DEFAULT_QUERY_CHUNK = "256"
BYTES_PER_GB = 2 ** 30


def trackPackOffline(pack: Path, force: bool) -> None:
    import torch
    from PIL import Image
    tModel = torch.hub.load(HUB_REPO, HUB_MODEL).cuda().eval()
    for tDir in sorted(pack.iterdir()):
        if not (tDir / "meta.json").exists() or ((tDir / "tracks.npz").exists() and not force):
            continue
        tT0 = time.time()
        tMeta = json.loads((tDir / "meta.json").read_text())
        tFrames = sorted((tDir / "frames").glob("*.jpg"))
        tVideo = torch.from_numpy(np.stack([np.asarray(Image.open(tF).convert("RGB")) for tF in tFrames])).permute(0, 3, 1, 2)[None].float().cuda()
        tQueries = torch.tensor(tMeta["queries"])[None].float().cuda()
        tChunk = int(os.environ.get(QUERY_CHUNK_ENV, DEFAULT_QUERY_CHUNK))
        tTrs, tVis = [], []
        for tI in range(0, tQueries.shape[1], tChunk):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                tTr, tVi = tModel(tVideo, queries=tQueries[:, tI:tI + tChunk], backward_tracking=True)
            tTrs.append(tTr[0].float().cpu())
            tVis.append(tVi[0].float().cpu())
            torch.cuda.empty_cache()
        tTracks, tVisAll = torch.cat(tTrs, 1).numpy(), torch.cat(tVis, 1).numpy()
        np.savez(tDir / "tracks.npz", tracks=tTracks, vis=tVisAll, mode="offline_bidir")
        print(tDir.name, tTracks.shape, round(float(tVisAll.mean()), 3),
              f"{time.time() - tT0:.0f}s peak {torch.cuda.max_memory_allocated() / BYTES_PER_GB:.1f}GB", flush=True)
        del tVideo
        torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> int:
    tAp = argparse.ArgumentParser(description="CoTracker3 offline (bidirectional) on every demo dir of a pack -> tracks.npz")
    tAp.add_argument("pack")
    tAp.add_argument("--force", action="store_true", help="re-track dirs that already have tracks.npz")
    tArgs = tAp.parse_args(argv)
    trackPackOffline(Path(tArgs.pack), tArgs.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
