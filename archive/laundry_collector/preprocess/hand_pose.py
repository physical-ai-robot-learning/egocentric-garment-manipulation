"""Hand-pose provider interface (spec §11). Collector v1 records raw video only; providers run offline on wrist frames.

Convention: 21 MediaPipe/MANO-order landmarks in the WRIST CAMERA frame (metres, OpenCV axes: x right, y down, z forward).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, PINKY_MCP = 0, 4, 5, 8, 9, 17
LANDMARKS = ["wrist", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip", "index_mcp", "index_pip", "index_dip", "index_tip",
             "middle_mcp", "middle_pip", "middle_dip", "middle_tip", "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
             "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip"]


@dataclass
class HandPose:
    keypoints_3d: np.ndarray                  # (21, 3) camera frame, metres; NaN where unknown
    confidence: np.ndarray                    # (21,) in [0, 1]
    handedness: str                           # "left" | "right" (the physical hand, not the image side)
    valid: bool = True
    keypoints_2d: np.ndarray | None = None    # (21, 2) pixels in the source image, if the provider has them
    extra: dict = field(default_factory=dict)

    @property
    def thumb_tip(self): return self.keypoints_3d[THUMB_TIP]

    @property
    def index_tip(self): return self.keypoints_3d[INDEX_TIP]

    def aperture_m(self) -> float:
        return float(np.linalg.norm(self.thumb_tip - self.index_tip))


class HandPoseProvider(ABC):
    """process(frame) -> HandPose | None for one wrist image. Implementations: WiLoR (offline, metric via MANO), MediaPipe
    (approximate, non-metric). Must not hold state across episodes; temporal smoothing lives in preprocess/smoothing.py."""

    name = "base"

    @abstractmethod
    def process(self, rgb: np.ndarray, *, intrinsics: dict, side: str) -> HandPose | None: ...


def aperture_normalized(d_m, pinch_min_m: float = 0.015, pinch_max_m: float = 0.100):
    """Spec §12: g = 0 closed .. 1 fully open, from the thumb-index distance."""
    return np.clip((np.asarray(d_m, float) - pinch_min_m) / (pinch_max_m - pinch_min_m), 0.0, 1.0)


class WilorProvider(HandPoseProvider):
    """WiLoR (wilor-mini: own YOLO hand detector + MANO) on the fisheye frame undistorted to a pinhole view; the pinhole focal
    is handed to WiLoR so its camera-frame keypoints are in that (= the wrist camera's) frame. 2026-10-08 bring-up: MediaPipe
    found no hand in the close dorsal wrist view; WiLoR found the left hand in ~half the frames, wrist 10-12 cm from the lens."""

    name = "wilor"

    def __init__(self, intrinsics: dict, image_size: tuple[int, int], balance: float = 0.3, hand_conf: float = 0.2):
        import cv2
        from ego_collector.hands3d.wilor import WilorTracker
        K, D = np.asarray(intrinsics["K"], float), np.asarray(intrinsics["D"], float).reshape(4, 1)
        w, h = image_size
        self.Kp = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=balance)
        self.Kp[0, 2], self.Kp[1, 2] = w / 2.0, h / 2.0           # WiLoR assumes the principal point at the image centre
        self.map = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), self.Kp, (w, h), cv2.CV_16SC2)
        self.t = WilorTracker(focal_px=float(self.Kp[0, 0]), image_size=(w, h), hand_conf=hand_conf)

    def process(self, bgr, *, intrinsics=None, side="left", timestamp_ms=None):
        import cv2
        und = cv2.remap(bgr, *self.map, interpolation=cv2.INTER_LINEAR)
        r = self.t(und)
        hf = r.get(side)
        if hf is None or not hf.visible:                      # WiLoR's own left/right label (not selfie-mirrored)
            others = [h for h in r.values() if h.visible]
            if not others:
                return None
            hf = others[0]
        conf = hf.confidence if np.isfinite(hf.confidence) else 1.0
        return HandPose(hf.kp3d, np.full(21, conf), side, True, keypoints_2d=hf.kp2d, extra=dict(reproj_px=np.nan, label_matched=r.get(side) is hf))


class MediaPipePnPProvider(HandPoseProvider):
    """MVP wrist-camera provider (2026-10-08 bring-up): fisheye frame -> pinhole undistortion (same camera frame, no rotation)
    -> MediaPipe Hands (2D landmarks + hand-centred metric 3D 'world' landmarks) -> solvePnP(world, 2D) -> 21 keypoints in the
    camera frame. Metric only as far as MediaPipe's average-hand scale is right (the hand size sets the depth). The hand
    nearest the camera (largest 2D extent) is taken as the wearer's hand; MediaPipe's left/right label is not used (it assumes
    a selfie-mirrored image)."""

    name = "mediapipe_pnp"

    def __init__(self, intrinsics: dict, image_size: tuple[int, int], balance: float = 0.3):
        import cv2
        from ego_collector.hands.mediapipe_tracker import HandLandmarker
        self.lm = HandLandmarker(num_hands=2, min_detection=0.4, min_presence=0.4, min_tracking=0.4)
        K, D = np.asarray(intrinsics["K"], float), np.asarray(intrinsics["D"], float).reshape(4, 1)
        w, h = image_size
        self.Kp = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=balance)
        self.map = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), self.Kp, (w, h), cv2.CV_16SC2)
        self.t_ms = 0

    def process(self, bgr, *, intrinsics=None, side="left", timestamp_ms=None):
        import cv2
        und = cv2.remap(bgr, *self.map, interpolation=cv2.INTER_LINEAR)
        self.t_ms = int(timestamp_ms) if timestamp_ms is not None else self.t_ms + 33
        dets = self.lm.detect_all(und, self.t_ms)
        if not dets:
            return None
        d = max(dets, key=lambda x: float(np.ptp(x.landmarks_2d, axis=0).prod()))
        ok, rv, tv = cv2.solvePnP(d.landmarks_3d.astype(np.float64), d.landmarks_2d.astype(np.float64), self.Kp, None, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None
        Rm = cv2.Rodrigues(rv)[0]
        kp = d.landmarks_3d @ Rm.T + tv.reshape(3)
        if (kp[:, 2] <= 0.01).any():
            return None
        proj, _ = cv2.projectPoints(d.landmarks_3d, rv, tv, self.Kp, None)
        reproj = float(np.linalg.norm(proj.reshape(-1, 2) - d.landmarks_2d, axis=1).mean())
        return HandPose(kp, np.full(21, d.label_confidence), side, True, keypoints_2d=d.landmarks_2d,
                        extra=dict(reproj_px=reproj, n_hands=len(dets)))
