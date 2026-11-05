"""QA video of one unsplit demo: the three cameras over a timeline of the per-arm pos / rot / grip masks (dataset stage; was
render() of tag_rig_unsplit.py). Green = in the loss, gray = masked, dark gray = masked on a filled (unmeasured) pose."""
from __future__ import annotations

import cv2
import numpy as np

from laundry_shared.contracts.action_contract import ACTION_DT_S

from .export_common import frameFile

VIDEO_FPS = 20
TILE_WH = (320, 180)                 # each camera tile
PANEL_H = 150                        # mask timeline panel below the tiles
FRAME_WH = (3 * TILE_WH[0], TILE_WH[1] + PANEL_H)
BAR_X0, BAR_W = 50, 900              # timeline x range
BAR_Y0, BAR_DY, BAR_THICK = 30, 17, 10
CURSOR_Y = (15, 135)
TITLE_XY = (50, 145)
PANEL_BG = 25
COL_IN = (60, 200, 60)
COL_MASKED = (130, 130, 130)
COL_FILLED = (90, 90, 90)
COL_TEXT = (230, 230, 230)
COL_CURSOR = (255, 255, 255)
LABEL_X = 2
LABEL_BASELINE_PX = 4                # row label text baseline below its bar
LABEL_SCALE, TITLE_SCALE = 0.35, 0.4
ROWS = ((0, "pos"), (1, "rot"), (2, "grip"))


def renderMaskVideo(build, path, title) -> None:
    """write the QA video of a built demo to `path` (mp4v, 20 fps) (was render)"""
    tN = build["n"]
    tVideo = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, FRAME_WH)
    tBars = {tS: np.stack([build[f"pos_ok_{tS}"], build[f"rot_ok_{tS}"], build[f"jv_{tS}"]], 0) for tS in "LR"}
    for tI in range(tN):
        tTiles = [cv2.resize(cv2.imread(str(frameFile(build["cam_dir"][tC], tC, int(build[f"img_{tC}"][tI])))), TILE_WH)
                  for tC in ("global", "left_wrist", "right_wrist")]
        tPanel = np.full((PANEL_H, FRAME_WH[0], 3), PANEL_BG, np.uint8)
        tY0 = BAR_Y0
        for tS, tLabel in (("L", "LEFT"), ("R", "RIGHT")):
            for tR, tName in ROWS:
                tOk = tBars[tS][tR]
                tXs = (np.arange(tN) / max(tN - 1, 1) * BAR_W + BAR_X0).astype(int)
                for tJ in range(tN - 1):
                    tCol = COL_IN if tOk[tJ] else (COL_MASKED if build[f"meas_{tS}"][tJ] or tName != "pos" else COL_FILLED)
                    cv2.line(tPanel, (tXs[tJ], tY0), (tXs[tJ + 1], tY0), tCol, BAR_THICK)
                cv2.putText(tPanel, f"{tLabel} {tName}", (LABEL_X, tY0 + LABEL_BASELINE_PX), 0, LABEL_SCALE, COL_TEXT, 1)
                tY0 += BAR_DY
        tCx = int(tI / max(tN - 1, 1) * BAR_W + BAR_X0)
        cv2.line(tPanel, (tCx, CURSOR_Y[0]), (tCx, CURSOR_Y[1]), COL_CURSOR, 1)
        cv2.putText(tPanel, f"{title}  t={tI * ACTION_DT_S:5.1f}s  green = in loss, gray = masked (dark = filled pose)", TITLE_XY,
                    0, TITLE_SCALE,
                    COL_TEXT, 1)
        tVideo.write(np.vstack([np.hstack(tTiles), tPanel]))
    tVideo.release()
