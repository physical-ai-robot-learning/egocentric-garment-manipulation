"""Robot geometry of the deploy: robotize_ik arms and FK, the capsule link-collision model, rotation helpers, the Orbbec
camera model (calibration files, projection) and a small cv2 line plot.
"""
from __future__ import annotations
import json
from types import SimpleNamespace

import cv2
import numpy as np

from egogarment import paths as egopaths
from garment_pipeline.eval_deploy.tagrig_core.policy import SIDES


ROBOT_CAM = egopaths.DATA_CALIB / "robot_cam_v001.json"


# 10-29: the Orbbec was raised on 10-26 -> the overlay extrinsic is v005 (fixed tags + table plane, 10-26 12:08; FK TCP lands on the
# real jaw tips, checked on the live frame), intrinsics still from v001 (v002+ store no intrinsics).  Display only (UI + run video).
ROBOT_CAM_EXT = egopaths.DATA_CALIB / "robot_cam_v005.json"


# ---------------------------------------------------------------- robot kinematics (reuse robotize_ik.Arm = umi_replay_bi settings)
ROBOTIZE_ARGS = dict(ik_step=7.5, ori_cost=0.1, table_z=0.003, tip_clear=0.005, tip_clear_left_extra=0.005, jaw_half=0.020, level_max=0.0,
                     roll_thr=115.0, roll_relax=1.5)


def make_arms(**over):
    """robotize_ik.Arm per side: Pink solver with the bridge joint limits (0.5 deg inside), joint/base/TCP overrides, orientation
    cost 0.1, 7.5 deg/step bound; corner_lift (finger-corner table floor); link-origin set for the 120 mm gap"""
    from garment_pipeline.labels_dataset.robotize_ik import Arm
    a = SimpleNamespace(**{**ROBOTIZE_ARGS, **over})
    return {s: Arm(s, a) for s in SIDES}, a


def fk(arms, side, q):
    return arms[side].pk.fk(np.asarray(q, np.float64))


# ---------------------------------------------------------------- capsule collision model (no link meshes in the repo)
# 10-23: the URDF references meshes_b601_gripper/link1-6.STL + gripper STLs, but only the three gripper STLs exist on this Mac
# (bh-egocentric/hra/hra_a100/urdf_gripper) and no code path loads them -> link-link collision uses CAPSULES between consecutive
# frame origins of the corrected pinocchio model (PinkIkSolver.full: base + joint offsets).  Radii are conservative half-widths
# of the B601 links (tape / photos, 10-23), fingers = segment finger-frame origin -> finger tip at the TCP plane.
CAPSULES = [("link1", "link2", 0.045), ("link2", "link3", 0.040), ("link3", "link4", 0.035), ("link4", "link5", 0.032),
            ("link5", "link6", 0.032), ("link6", "gbody", 0.032)]


# 10-23: "gbody" = midpoint of the two finger-joint frames (jaw base).  NOT gripper_link: with the tcp override (-0.019 m) the URDF
# gripper_link origin sits 19 mm BEYOND the real jaw tips, so a link6 -> gripper_link capsule dipped into the table on 48 % of the
# training grasp frames.  Table check = link capsules only; the fingers are covered by robotize_ik corner_lift (finger-corner floor).
FINGER_R = 0.012
TABLE_R_WRIST = 0.012


class Collision:
    def __init__(self, arms):
        import pinocchio as pin
        pk = arms["right"].pk; self.pin = pin; self.m = pk.full; self.d = self.m.createData(); self.q0 = pk.q_full_pin.copy()
        self.qidx = pk.qidx; self.joff = pk.joff
        self.fid = {f"{s}_{n}": self.m.getFrameId(f"{s}_{n}") for s in SIDES for n in
                    ("link1", "link2", "link3", "link4", "link5", "link6", "gripper_link", "gripper_left", "gripper_right", "tcp")}

    def capsules(self, q):
        """q {side: (6,) bridge rad} -> {side: [(a, b, r, name)]}"""
        pin = self.pin; qq = self.q0.copy()
        for s in SIDES: qq[self.qidx[s]] = np.asarray(q[s], np.float64) + self.joff[s]
        pin.framesForwardKinematics(self.m, self.d, qq); P = {k: self.d.oMf[i].translation.copy() for k, i in self.fid.items()}
        for s in SIDES: P[f"{s}_gbody"] = (P[f"{s}_gripper_left"] + P[f"{s}_gripper_right"]) / 2
        out = {}
        for s in SIDES:
            C = [(P[f"{s}_{a}"], P[f"{s}_{b}"], r, f"{a}-{b}") for a, b, r in CAPSULES]
            T = self.d.oMf[self.fid[f"{s}_tcp"]]; ax = T.rotation[:, 2]; tcp = T.translation
            for fn in ("gripper_left", "gripper_right"):
                pf = P[f"{s}_{fn}"]; lat = (pf - tcp) - ax * float(ax @ (pf - tcp))       # finger offset across the tool axis
                C.append((pf - 0.02 * ax, tcp + lat, FINGER_R, fn))
            out[s] = C
        return out

    def check(self, q, table_z=0.003, arm_clear=0.010, table_clear=0.0):
        """-> dict(min_arm_clear_m, pair, min_table_clear_m (non-finger capsules), finger_low_m)"""
        C = self.capsules(q); best = (np.inf, None)
        for a0, b0, r0, n0 in C["left"]:
            for a1, b1, r1, n1 in C["right"]:
                dd = seg_dist(a0, b0, a1, b1) - r0 - r1
                if dd < best[0]: best = (dd, f"L.{n0} / R.{n1}")
        # table: link capsules; the wrist capsule (link6-gbody) with TABLE_R_WRIST (its 32 mm arm-arm radius put 5.4 % of the training
        # grasp frames "under" the table -- the real jaw housing is narrower toward the fingers); fingers -> corner_lift limiter
        tab = min(min(a[2], b[2]) - (TABLE_R_WRIST if n == "link6-gbody" else r) - table_z for s in SIDES for a, b, r, n in C[s] if not n.startswith("gripper_"))
        fl = min(min(a[2], b[2]) - r - table_z for s in SIDES for a, b, r, n in C[s] if n.startswith("gripper_"))
        return dict(arm_clear_m=float(best[0]), pair=best[1], table_clear_m=float(tab), finger_low_m=float(fl),
                    ok_arm=bool(best[0] >= arm_clear), ok_table=bool(tab >= table_clear))


def seg_dist(p1, q1, p2, q2):
    """minimum distance between segments p1q1 and p2q2 (Ericson, Real-Time Collision Detection 5.1.9)"""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2; a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a <= 1e-12 and e <= 1e-12: return float(np.linalg.norm(r))
    if a <= 1e-12: s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e <= 1e-12: t, s = 0.0, np.clip(-c / a, 0, 1)
        else:
            b = d1 @ d2; den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0: t, s = 0.0, np.clip(-c / a, 0, 1)
            elif t > 1: t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def link_gap(arms, q):
    """umi_replay_bi / robotize_ik link-origin distance (m): min over link3-6, gripper_link, fingers, tcp of both arms"""
    return float(np.linalg.norm(arms["left"].links(q["left"])[:, None] - arms["right"].links(q["right"])[None], axis=2).min())


# ---------------------------------------------------------------- geometry helpers
def rot_interp(Ra, Rb, s):
    from scipy.spatial.transform import Rotation as Rot
    return (Rot.from_matrix(Ra) * Rot.from_rotvec(s * Rot.from_matrix(Ra.T @ Rb).as_rotvec())).as_matrix()


def rot_angle_deg(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def project(T_C_B, K, P_B):
    """base-frame points (N, 3) -> Orbbec pixels (N, 2) (848x480 intrinsics), points behind the camera -> nan"""
    P = (T_C_B[:3, :3] @ np.asarray(P_B).T).T + T_C_B[:3, 3]; z = P[:, 2:3]
    uv = np.c_[K["fx"] * P[:, 0] / P[:, 2] + K["cx"], K["fy"] * P[:, 1] / P[:, 2] + K["cy"]]
    uv[(z[:, 0] <= 0.05)] = np.nan; return uv


def robot_cam():
    j = json.loads(ROBOT_CAM.read_text()); return np.asarray(json.loads(ROBOT_CAM_EXT.read_text())["T_C_B"]), j["intrinsics"]


def cv_plot(series, ylim, title="", size=(900, 260), hline=None, xlabel=""):
    """tiny cv2 line plot (no matplotlib in .venv-mac): series = [(x, y, BGR colour, label)] -> BGR image"""
    W, H = size; img = np.full((H, W, 3), 255, np.uint8); l, r, t, b = 50, 10, 24, 22
    xs = np.concatenate([np.asarray(x, float) for x, *_ in series]); x0, x1 = float(xs.min()), float(max(xs.max(), xs.min() + 1e-9)); y0, y1 = ylim
    P = lambda x, y: (int(l + (x - x0) / (x1 - x0) * (W - l - r)), int(t + (y1 - np.clip(y, y0, y1)) / (y1 - y0) * (H - t - b)))
    cv2.rectangle(img, P(x0, y1), P(x1, y0), (0, 0, 0), 1)
    for v in (y0, (y0 + y1) / 2, y1): cv2.putText(img, f"{v:g}", (2, P(x0, v)[1] + 4), 0, 0.4, (0, 0, 0), 1)
    if hline is not None: cv2.line(img, P(x0, hline), P(x1, hline), (150, 150, 150), 1)
    for i, (x, y, c, lab) in enumerate(series):
        pts = np.array([P(a, b_) for a, b_ in zip(x, y) if np.isfinite(b_)], np.int32)
        if len(pts) > 1: cv2.polylines(img, [pts], False, c, 2)
        cv2.putText(img, lab, (W - 190, t + 16 + 16 * i), 0, 0.45, c, 1)
    cv2.putText(img, title, (l, 16), 0, 0.5, (0, 0, 0), 1); cv2.putText(img, f"{x0:.3g} .. {x1:.3g} {xlabel}", (l, H - 6), 0, 0.4, (0, 0, 0), 1)
    return img
