"""Units shared by the fixed-camera pose stage (times are host monotonic int64 ns everywhere; new in the 11-04 refactor)."""
from __future__ import annotations

NS_PER_S = 1e9               # float on purpose: the stage divides / scales int64 ns by 1e9 (float64 results)
NS_PER_MS = 1e6
MM_PER_M = 1000.0
ORBBEC_FRAME_S = 0.067       # one Orbbec colour frame at 15 fps: added to run lengths measured first-to-last frame
