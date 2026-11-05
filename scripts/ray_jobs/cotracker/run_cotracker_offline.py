"""11-04 option 2: CoTracker3 OFFLINE + backward_tracking (bidirectional) on every pack dir -> tracks.npz {tracks (T,N,2) px, vis (T,N)}, same contract
as run_cotracker.py (online, forward only).  Backward tracking also fills frames BEFORE a query frame and re-links points across short occlusions.
usage: run_cotracker_offline.py PACK_DIR [--force]"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
m = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").cuda().eval()
force = "--force" in sys.argv
for d in sorted(Path(sys.argv[1]).iterdir()):
    if not (d / "meta.json").exists() or ((d / "tracks.npz").exists() and not force): continue
    t0 = time.time(); meta = json.loads((d / "meta.json").read_text()); fr = sorted((d / "frames").glob("*.jpg"))
    v = torch.from_numpy(np.stack([np.asarray(Image.open(f).convert("RGB")) for f in fr])).permute(0, 3, 1, 2)[None].float().cuda()
    q = torch.tensor(meta["queries"])[None].float().cuda()
    QC = int(__import__("os").environ.get("QCHUNK", "256")); TR, VI = [], []          # query chunks: memory grows with T x N (OOM at N=1340, T=441)
    for i in range(0, q.shape[1], QC):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            tr, vis = m(v, queries=q[:, i:i + QC], backward_tracking=True)
        TR.append(tr[0].float().cpu()); VI.append(vis[0].float().cpu()); torch.cuda.empty_cache()
    tr, vis = torch.cat(TR, 1).numpy(), torch.cat(VI, 1).numpy()
    np.savez(d / "tracks.npz", tracks=tr, vis=vis, mode="offline_bidir")
    print(d.name, tr.shape, round(float(vis.mean()), 3), f"{time.time() - t0:.0f}s peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
    del v; torch.cuda.empty_cache()
