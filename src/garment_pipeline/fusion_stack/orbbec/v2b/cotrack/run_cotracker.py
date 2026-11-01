"""4090: CoTracker3 ONLINE (constant memory, forward only) on every pack dir -> tracks.npz {tracks (T,N,2) px, vis (T,N)}.
Queries are at the seed frames (first tracked frame of each arm); frames before a query frame get NaN for that point."""
import json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
m = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").cuda().eval(); S = m.step
for d in sorted(Path(sys.argv[1]).iterdir()):
    if not (d / "meta.json").exists() or (d / "tracks.npz").exists(): continue
    meta = json.loads((d / "meta.json").read_text()); fr = sorted((d / "frames").glob("*.jpg"))
    v = torch.from_numpy(np.stack([np.asarray(Image.open(f).convert("RGB")) for f in fr])).permute(0, 3, 1, 2)[None].float()
    q = torch.tensor(meta["queries"])[None].float().cuda(); T = v.shape[1]
    with torch.no_grad():
        m(video_chunk=v[:, : S * 2].cuda(), is_first_step=True, queries=q)
        for i in range(0, T - S, S):
            tr, vis = m(video_chunk=v[:, i: i + S * 2].cuda())
    tr, vis = tr[0].cpu().numpy()[:T], vis[0].cpu().numpy()[:T]
    np.savez(d / "tracks.npz", tracks=tr, vis=vis); print(d.name, tr.shape, float(vis.mean()), flush=True)
