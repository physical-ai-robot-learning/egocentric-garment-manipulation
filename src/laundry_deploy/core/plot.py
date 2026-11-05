"""Tiny cv2 line plot (no matplotlib in the robot Mac's .venv-mac): gripper traces of a run, domain-gap histograms."""

from __future__ import annotations

import cv2
import numpy as np

MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM = 50, 10, 24, 22  # px around the plot box
WHITE, BLACK, GREY = (255, 255, 255), (0, 0, 0), (150, 150, 150)
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
TICK_FONT_SCALE, LEGEND_FONT_SCALE, TITLE_FONT_SCALE = 0.4, 0.45, 0.5
LINE_WIDTH = 2
LEGEND_X_FROM_RIGHT = 190  # legend column starts this far left of the right edge
LEGEND_ROW_PX = 16
X_SPAN_EPS = 1e-9  # a single-x series still gets a non-zero x range


def cvPlot(series, yLim, title: str = "", size=(900, 260), hline=None, xLabel: str = "") -> np.ndarray:
    """series = [(x, y, BGR colour, label)] -> BGR image."""
    tW, tH = size
    tImg = np.full((tH, tW, 3), 255, np.uint8)
    tXs = np.concatenate([np.asarray(tX, float) for tX, *_ in series])
    tX0 = float(tXs.min())
    tX1 = float(max(tXs.max(), tXs.min() + X_SPAN_EPS))
    tY0, tY1 = yLim

    def _px(x, y):
        return (
            int(MARGIN_LEFT + (x - tX0) / (tX1 - tX0) * (tW - MARGIN_LEFT - MARGIN_RIGHT)),
            int(MARGIN_TOP + (tY1 - np.clip(y, tY0, tY1)) / (tY1 - tY0) * (tH - MARGIN_TOP - MARGIN_BOTTOM)),
        )

    cv2.rectangle(tImg, _px(tX0, tY1), _px(tX1, tY0), BLACK, 1)
    for tV in (tY0, (tY0 + tY1) / 2, tY1):
        cv2.putText(tImg, f"{tV:g}", (2, _px(tX0, tV)[1] + 4), FONT, TICK_FONT_SCALE, BLACK, 1)
    if hline is not None:
        cv2.line(tImg, _px(tX0, hline), _px(tX1, hline), GREY, 1)
    for tI, (tX, tY, tColour, tLabel) in enumerate(series):
        tPts = np.array([_px(tA, tB) for tA, tB in zip(tX, tY) if np.isfinite(tB)], np.int32)
        if len(tPts) > 1:
            cv2.polylines(tImg, [tPts], False, tColour, LINE_WIDTH)
        cv2.putText(
            tImg,
            tLabel,
            (tW - LEGEND_X_FROM_RIGHT, MARGIN_TOP + LEGEND_ROW_PX + LEGEND_ROW_PX * tI),
            FONT,
            LEGEND_FONT_SCALE,
            tColour,
            1,
        )
    cv2.putText(tImg, title, (MARGIN_LEFT, 16), FONT, TITLE_FONT_SCALE, BLACK, 1)
    cv2.putText(tImg, f"{tX0:.3g} .. {tX1:.3g} {xLabel}", (MARGIN_LEFT, tH - 6), FONT, TICK_FONT_SCALE, BLACK, 1)
    return tImg
