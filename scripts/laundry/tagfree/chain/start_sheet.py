"""11-12: contact sheet of each demo's start frame (protocol '자유:' start) with the stored L spots -> did every demo start at the L?"""
import ast, json, sys
from pathlib import Path
import cv2, numpy as np
R = Path.home() / "ego_collector/datasets/m3_runs/_imu_consistency"; sys.path[:0] = [str(R)]
import camcal  # noqa
take = sys.argv[1]; tk = Path.home() / "ego_collector/datasets/tag_rig" / take
m = json.loads((tk / "meta.json").read_text()); I = m["intrinsics"]; t = np.load(tk / "t_ns.npy").astype(np.int64)
P = ast.literal_eval(m["protocol"]) if isinstance(m["protocol"], str) else m["protocol"]; t0 = int(m["t_start_monotonic_ns"])
T = camcal.T_C_B(take); LS = {"left": (0.289, 0.118, 0.015), "right": (0.300, -0.222, 0.021)}
tiles = []
for i, p in enumerate([p for p in P if p["text"].startswith("자유:") and p.get("ended_early")]):
    k = int(np.clip(np.searchsorted(t, t0 + int(p["t0"] * 1e9)), 0, len(t) - 1)); img = cv2.imread(str(tk / f"color_{k:05d}.jpg"))
    img = cv2.resize(img, (848, 480))
    for a, col in (("left", (255, 120, 0)), ("right", (0, 140, 255))):
        x = T[:3, :3] @ np.array(LS[a]) + T[:3, 3]; cv2.circle(img, (int(I["fx"] * x[0] / x[2] + I["cx"]), int(I["fy"] * x[1] / x[2] + I["cy"])), 14, col, 3)
    cv2.putText(img, f"d{i}", (10, 40), 0, 1.4, (0, 0, 255), 3); tiles.append(cv2.resize(img[180:480, 250:800], (440, 240)))
while len(tiles) % 5: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(sys.argv[2], np.vstack([np.hstack(tiles[r:r + 5]) for r in range(0, len(tiles), 5)])); print(sys.argv[2], len(tiles))
