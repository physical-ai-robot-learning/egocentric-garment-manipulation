"""Rig pose from ALL visible tag corners at once (2026-10-19, GPT review: "do not average the per-tag poses; solve one rig pose from every
corner with the known rigid tag-rig geometry"). Corners of tag i in the rig frame = T_rig_tag_i * corner_i; one PnP (SQPNP init, then
Levenberg-Marquardt refine) over 4*n points. Returns T_C_rig, reprojection RMSE (px), the visible tags and their pixel sides.
Single-tag fallback keeps the IPPE solution (the planar ambiguity is resolved by the multi-tag fit when >= 2 tags are seen)."""
from __future__ import annotations
import cv2
import numpy as np


# 10-23: 30 mm tags on the back of each UMI camera (black square = detector quad); every other rig tag uses the take-wide default size
TAG_SIZE_M = {220: 0.030, 221: 0.030}


def size_of(i: int, default: float) -> float: return TAG_SIZE_M.get(int(i), default)


# 10-23 (umi_tag_rig_v002): a LONE 30 mm camera-back tag has a planar IPPE flip (p90 70-130 mm TCP error). rig_pose() refuses a pose
# that would come ONLY from these tags unless allow_lone_guard=True; tag_rig_fuse.fuse_take resolves them with the IMU flip guard
# (single_tag_candidates + guard_pick). Frames with any old tag (200-202 / 210-212) use the normal multi-tag PnP unchanged.
GUARD_TAGS = frozenset(TAG_SIZE_M)
GUARD_ROT_DEG, GUARD_POS_M, GUARD_VMAX_MPS, GUARD_MAX_GAP_S = 10.0, 0.03, 1.0, 2.0


def only_guard_tags(tags) -> bool: return bool(tags) and all(int(i) in GUARD_TAGS for i in tags)


def single_tag_candidates(c, i: int, T_rig_tag: dict, K: np.ndarray, size: float, dist=None):
    """both IPPE solutions of ONE tag -> [(T_C_rig, reprojection error)]."""
    dist = np.zeros(5) if dist is None else dist
    _, rvs, tvs, err = cv2.solvePnPGeneric(corners_obj(size_of(i, size)), c.reshape(4, 2).astype(np.float64), K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    out = []
    for rv, tv, e in zip(rvs, tvs, np.ravel(err) if err is not None else [np.nan] * len(rvs)):
        T = np.eye(4); T[:3, :3] = cv2.Rodrigues(rv)[0]; T[:3, 3] = np.ravel(tv); out.append((T @ np.linalg.inv(T_rig_tag[int(i)]), float(e)))
    return out


def guard_pick(cands, R_pred: np.ndarray, T_prev: np.ndarray, tcp_in_rig, dt: float, rot_gate=GUARD_ROT_DEG, pos_gate=GUARD_POS_M, v_max=GUARD_VMAX_MPS):
    """IMU flip guard: candidate closest in rotation to the gyro-propagated previous pose R_pred; accept if angle <= rot_gate and the TCP
    jump vs the previous accepted pose <= pos_gate + v_max*dt.  -> (T or None, angle_deg, jump_m, reason)"""
    tcp = np.r_[np.asarray(tcp_in_rig, float), 1.0]
    ang = [float(np.degrees(np.arccos(np.clip((np.trace(R_pred.T @ T[:3, :3]) - 1) / 2, -1, 1)))) for T, _ in cands]
    j = int(np.argmin(ang)); T = cands[j][0]; jump = float(np.linalg.norm((T @ tcp - T_prev @ tcp)[:3]))
    if ang[j] > rot_gate: return None, ang[j], jump, "rot"
    if jump > pos_gate + v_max * dt: return None, ang[j], jump, "jump"
    return T, ang[j], jump, "ok"


def corners_obj(size: float) -> np.ndarray:
    s = size / 2; return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float64)


def rig_pose(cs, ids, T_rig_tag: dict, K: np.ndarray, size: float, min_side: float = 18.0, dist=None, allow_lone_guard: bool = False):
    dist = np.zeros(5) if dist is None else dist; P3, P2, used, sides = [], [], [], []
    for c, i in zip(cs if ids is not None else [], ids.ravel() if ids is not None else []):
        i = int(i)
        if i not in T_rig_tag: continue
        px = c.reshape(4, 2).astype(np.float64); side = float(np.mean(np.linalg.norm(px - np.roll(px, 1, 0), axis=1)))
        if side < min_side: continue
        T = T_rig_tag[i]; obj = corners_obj(size_of(i, size)); P3.append((T[:3, :3] @ obj.T).T + T[:3, 3]); P2.append(px); used.append(i); sides.append(side)
    if not used: return None
    if only_guard_tags(used) and not allow_lone_guard: return None      # lone 30 mm tag: flip-ambiguous -> caller must use the IMU guard
    P3, P2 = np.concatenate(P3), np.concatenate(P2)
    if len(used) == 1:
        ok, rv, tv = cv2.solvePnP(corners_obj(size_of(used[0], size)), P2, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok: return None
        Tc = np.eye(4); Tc[:3, :3] = cv2.Rodrigues(rv)[0]; Tc[:3, 3] = tv.ravel(); T = Tc @ np.linalg.inv(T_rig_tag[used[0]])
        rv, tv = cv2.Rodrigues(T[:3, :3])[0], T[:3, 3].reshape(3, 1)
    else:
        ok, rv, tv = cv2.solvePnP(P3, P2, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if not ok: return None
        rv, tv = cv2.solvePnPRefineLM(P3, P2, K, dist, rv, tv)
    rp = np.linalg.norm(cv2.projectPoints(P3, rv, tv, K, dist)[0].reshape(-1, 2) - P2, axis=1)
    T = np.eye(4); T[:3, :3] = cv2.Rodrigues(rv)[0]; T[:3, 3] = np.ravel(tv)
    return dict(T=T, rmse_px=float(np.sqrt(np.mean(rp ** 2))), tags=used, sides=sides)


class DepthStore:
    """10-20: depth of a tag_rig_record take in millimetres, depth.npz (short takes) or depth_%05d.png (robotcal / axes)."""
    def __init__(self, take):
        import json
        from pathlib import Path
        self.E = Path(take); meta = json.loads((self.E / "meta.json").read_text()); self.ds = float(meta.get("depth_scale") or 1.0)
        self.npz = np.load(self.E / "depth.npz")["depth"] if (self.E / "depth.npz").exists() else None

    def mm(self, k: int):
        if self.npz is not None: return self.npz[k].astype(np.float64) * self.ds
        d = cv2.imread(str(self.E / f"depth_{k:05d}.png"), cv2.IMREAD_UNCHANGED)
        return None if d is None else d.astype(np.float64) * self.ds


def quad_depth_m(dep_mm, quad, min_px: int = 30):
    """median depth (m) inside a tag quad (4,2), None if too few valid pixels"""
    if dep_mm is None: return None
    q = quad.reshape(4, 2).astype(np.float32); x0, y0 = np.maximum(np.floor(q.min(0)).astype(int), 0); x1, y1 = np.ceil(q.max(0)).astype(int)
    sub = dep_mm[y0:y1 + 1, x0:x1 + 1]; msk = np.zeros(sub.shape, np.uint8); cv2.fillConvexPoly(msk, (q - [x0, y0]).astype(np.int32), 1)
    z = sub[(msk > 0) & (sub > 100)]
    return float(np.median(z)) / 1000.0 if z.size >= min_px else None
