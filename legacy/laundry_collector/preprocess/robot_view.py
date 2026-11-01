"""Robot-wrist view of a bare-hand wrist video + the REAL robot gripper tips composited in (2026-10-08 user: only the gripper ends,
like the HandUMI / robot data).

  RobotC922View      fisheye wrist frame -> the MEASURED reBot right-wrist C922 (configs/calibration/robot_right_wrist_c922_v001.json,
                     640x480, R = I: the robot camera placed exactly at the wrist camera) -- the view of HRA right_wrist_c922.mp4.
  RobotTipsSprite    the real tips cut from a real robot wrist frame (robot_A_right_view_v001.jpg, jaws closed). The gripper is rigid
                     in the robot camera, so only the opening changes: each finger is shifted sideways by f * (w/2) / Z(row), Z
                     growing from the bottom edge to the tip row along the gripper axis (hand-eye TCP), so near rows move more.
  T_camera_tcp()     the robot TCP in this camera from the hand-eye calibration (X_tcp_cam, Park) -- used as the gripper pose for the
                     action labels, the same convention as the robot data.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

CAL = Path(__file__).resolve().parents[2] / "configs" / "calibration"
ROBOT_CAM = CAL / "robot_right_wrist_c922_v001.json"
ROBOT_REF = CAL / "robot_A_right_view_v001.jpg"


def robot_cam():
    r = json.loads(ROBOT_CAM.read_text())
    return np.asarray(r["K"], float), np.asarray(r["dist"], float), r


def T_camera_tcp(method: str = "park") -> np.ndarray:
    _, _, r = robot_cam()
    return np.linalg.inv(np.asarray(r["X_tcp_cam"][method], float))


class RobotC922View:
    def __init__(self, fisheye_intr: dict, src_size: tuple[int, int], size=(640, 480)):
        Kr, Dr, _ = robot_cam()
        K = np.asarray(fisheye_intr["K"], float).copy(); D = np.asarray(fisheye_intr["D"], float).reshape(4, 1)
        cw, ch = fisheye_intr.get("image_size") or src_size
        K[0] *= src_size[0] / cw; K[1] *= src_size[1] / ch
        w, h = size
        u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        n = cv2.undistortPoints(np.stack([u.ravel(), v.ravel()], 1).reshape(-1, 1, 2), Kr, Dr).reshape(-1, 2)
        pts = np.concatenate([n, np.ones((len(n), 1))], 1).reshape(-1, 1, 3)
        uv = cv2.fisheye.projectPoints(pts, np.zeros(3), np.zeros(3), K, D)[0].reshape(h, w, 2).astype(np.float32)
        self.mx, self.my, self.size = uv[..., 0], uv[..., 1], size

    def render(self, img):
        return cv2.remap(img, self.mx, self.my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class RobotTipsSprite:
    def __init__(self, ref: Path = ROBOT_REF, dark: int = 75, min_row: int = 290):
        img = cv2.imread(str(ref)); g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        m = ((g < dark) & (np.arange(g.shape[0])[:, None] >= min_row)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)); m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT] >= g.shape[0] - 2 and st[i, cv2.CC_STAT_AREA] > 2000]
        m = np.isin(lab, keep).astype(np.uint8)
        ys, xs = np.nonzero(m)
        self.tip_row = int(ys.min())
        # the gap between the closed fingers: the brightest column of the mask region near the tip
        # gap = the column inside the finger span (rows just below the tip) that the mask covers least / that is brightest
        r0, r1 = self.tip_row + 5, min(self.tip_row + 80, g.shape[0])
        cov = m[r0:r1].mean(0); lo, hi = np.flatnonzero(cov > 0.3)[[0, -1]]
        lo2, hi2 = int(lo + 0.2 * (hi - lo)), int(hi - 0.2 * (hi - lo))
        score = cov[lo2:hi2] - g[r0:r1, lo2:hi2].mean(0) / 255.0
        self.gap_col = int(lo2 + np.argmin(score))
        self.img, self.H, self.W = img, g.shape[0], g.shape[1]
        alpha = cv2.GaussianBlur(m.astype(np.float32), (5, 5), 0)
        cols = np.arange(self.W)[None, :]
        self.alpha = {"left": alpha * (cols < self.gap_col), "right": alpha * (cols >= self.gap_col)}
        Kr, _, _ = robot_cam(); self.f = float(Kr[0, 0]); self.cx, self.cy = float(Kr[0, 2]), float(Kr[1, 2])
        self.Z = self._row_depth()

    def _row_depth(self) -> np.ndarray:
        """Depth of the finger at each image row: the point on the gripper axis (TCP backwards along the approach) seen at that row."""
        T = T_camera_tcp(); p, Rm = T[:3, 3], T[:3, :3]
        a = Rm[:, int(np.argmax(np.abs(Rm[2, :])))]; a = a if a[2] > 0 else -a          # approach = TCP axis closest to the optical axis
        rows = np.arange(self.H, dtype=float); y = (rows - self.cy) / self.f
        # p - s a: (p_y - s a_y) = y (p_z - s a_z)  ->  s = (p_y - y p_z) / (a_y - y a_z)
        s = (p[1] - y * p[2]) / (a[1] - y * a[2] + 1e-12)
        Z = p[2] - s * a[2]
        return np.clip(Z, 0.03, p[2])

    def composite(self, frame: np.ndarray, jaw_width_m: float) -> tuple[np.ndarray, np.ndarray]:
        """frame: 640x480 robot-view image -> (image with the tips at this opening, alpha mask)."""
        out = frame.astype(np.float32); A = np.zeros(frame.shape[:2], np.float32)
        shift = self.f * (jaw_width_m / 2.0) / self.Z                                    # px per row
        xx, yy = np.meshgrid(np.arange(self.W, dtype=np.float32), np.arange(self.H, dtype=np.float32))
        for side, sgn in (("left", -1.0), ("right", 1.0)):
            mx = xx - sgn * shift[:, None].astype(np.float32)
            spr = cv2.remap(self.img, mx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            al = cv2.remap(self.alpha[side], mx, yy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[..., None]
            out = out * (1 - al) + spr.astype(np.float32) * al; A = np.maximum(A, al[..., 0])
        return out.clip(0, 255).astype(np.uint8), A
