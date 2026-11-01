"""Virtual parallel-jaw gripper: mesh definition (gripper frame) + renderers (renderer spec §2, §8-§10, §12).

Mesh (gripper frame, metres; see gripper_frame.py for the axes):
  two jaw boxes centred at x = +/- jaw_width/2, from the TCP (y = +tip_overhang) back to y = -jaw_length, thin in x, deep in z
  a base / palm box behind the jaws spanning both, at y in [-jaw_length - base_depth, -jaw_length]
The TCP (frame origin) sits between the jaw TIPS, like the human pinch point. Swap PrimitiveGripperMesh for a CAD mesh with the
same frame later.

OpenCVRenderer: projects with the wrist camera model (Kannala-Brandt fisheye via cv2.fisheye, or pinhole + distortion) and fills
faces back-to-front with Lambert shading (painter's order, back faces culled). `render()` returns the image AND the gripper
mask + a per-pixel depth of the gripper, so a z-buffer test against hand / scene depth can be added without changing callers
(spec §12: MVP = gripper always in front).
"""
from __future__ import annotations

from dataclasses import dataclass

from pathlib import Path

import cv2
import numpy as np


@dataclass
class PrimitiveGripperMesh:
    jaw_length: float = 0.045
    jaw_thickness: float = 0.008     # along x
    jaw_depth: float = 0.018         # along z
    tip_overhang: float = 0.004      # jaws reach this far past the TCP (+y)
    base_depth: float = 0.020
    base_width_margin: float = 0.012

    @staticmethod
    def _box(lo, hi):
        x0, y0, z0 = lo; x1, y1, z1 = hi
        V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
        F = np.array([[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 4, 7, 3]])   # outward (right-hand rule)
        return V, F

    def parts(self, jaw_width: float) -> list[tuple[str, np.ndarray, np.ndarray]]:
        h, t, dz = jaw_width / 2, self.jaw_thickness, self.jaw_depth / 2
        y_tip, y_back = self.tip_overhang, -self.jaw_length
        out = [("jaw_neg", *self._box((-h - t, y_back, -dz), (-h, y_tip, dz))),
               ("jaw_pos", *self._box((h, y_back, -dz), (h + t, y_tip, dz)))]
        bw = max(h + t, 0.012) + self.base_width_margin
        out.append(("base", *self._box((-bw, y_back - self.base_depth, -dz * 1.3), (bw, y_back, dz * 1.3))))
        return out


@dataclass
class JawTipsMesh:
    """Only the two finger ENDS, as the robot / HandUMI wrist camera sees them (2026-10-08 user: "only the gripper tips").
    Each finger: a tapered black block from its tip (y = 0, at x = +/- jaw_width/2) back towards the camera (-y) for
    `length`, wider and taller at the back; a light pad strip on the inner face. Gripper frame as in gripper_frame.py."""
    cull = False                      # painter's order only (convex fingers, no winding assumptions)
    length: float = 0.075
    tip_width: float = 0.010          # along x at the tip
    back_width: float = 0.018
    tip_height: float = 0.016         # along z
    back_height: float = 0.030
    pad: float = 0.003

    def parts(self, jaw_width: float):
        out = []
        for sgn, name in ((-1, "finger_neg"), (1, "finger_pos")):
            h = jaw_width / 2
            def ring(y, w, hh):
                xi = sgn * h; xo = sgn * (h + w)
                return [(xi, y, -hh / 2), (xo, y, -hh / 2), (xo, y, hh / 2), (xi, y, hh / 2)]
            V = np.array(ring(0.0, self.tip_width, self.tip_height) + ring(-self.length, self.back_width, self.back_height), float)
            F = np.array([[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]])
            if sgn < 0:                                   # mirrored ring order -> flip winding to keep normals outward
                F = F[:, ::-1]
            out.append((name, V, F))
            # pad strip on the inner face
            xi = sgn * (h - 1e-4)
            P = np.array([(xi, 0.0, -self.tip_height * 0.35), (xi, 0.0, self.tip_height * 0.35),
                          (xi, -self.length * 0.8, self.back_height * 0.35), (xi, -self.length * 0.8, -self.back_height * 0.35)])
            out.append(("pad", P, np.array([[0, 1, 2, 3] if sgn > 0 else [3, 2, 1, 0]])))
        return out


PIPER_TIP_STL = Path.home() / "handumi-print" / "tips" / "stl" / "Agile-X-Piper-LEFT-Gripper.stl"


def _cluster_decimate(V, F, cell):
    """Vertex-clustering decimation (no extra dependency): snap vertices to a `cell` grid, merge, drop collapsed faces."""
    key = np.round(V / cell).astype(np.int64)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    Vn = np.zeros((cnt.size, 3)); np.add.at(Vn, inv, V); Vn /= cnt[:, None]
    Fn = inv[F]
    keep = (Fn[:, 0] != Fn[:, 1]) & (Fn[:, 1] != Fn[:, 2]) & (Fn[:, 0] != Fn[:, 2])
    Fn = np.unique(np.sort(Fn[keep], axis=1), axis=0, return_index=True)[1]
    return Vn, inv[F][keep][np.sort(Fn)]


class PiperTipsMesh:
    """The HandUMI fingertips actually mounted for the HandUMI takes: AgileX-Piper generic parallel-jaw tip
    (~/handumi-print/tips/stl, 71 x 26.5 x 56 mm; README: temporary, a reBot tip will replace it).
    STL frame (mm, after subtracting its bounds minimum): X = finger axis, mount face x = 0 -> tip x = 71; Y = thickness, the flat
    GRASP face (pad recess) at y = 0; Z = width (56 mm at the mount, narrower at the tip).
    Placed per finger: tip centre (x = 71, y = 0, z = width/2) at TCP +/- jaw/2 along the gripper x, finger axis along -y_g
    (back towards the camera), grasp face towards the other finger. Faces whose normal is the grasp direction are drawn as the
    grey pad, the rest black, like the printed tips."""

    cull = True

    def __init__(self, stl: Path = PIPER_TIP_STL, cell_mm: float = 1.5):
        import trimesh
        m = trimesh.load(str(stl))
        V = (m.vertices - m.bounds[0]); V, F = _cluster_decimate(V, np.asarray(m.faces), cell_mm)
        self.V, self.F = V / 1000.0, F
        n = np.cross(self.V[F[:, 1]] - self.V[F[:, 0]], self.V[F[:, 2]] - self.V[F[:, 0]])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        cen = self.V[F].mean(1)
        # orient faces outward (STL winding is not guaranteed): flip a face whose normal points into the solid's centroid side
        flip = np.einsum("ij,ij->i", n, cen - self.V.mean(0)) < 0
        self.F = np.where(flip[:, None], self.F[:, ::-1], self.F); n[flip] *= -1
        self.pad = n[:, 1] < -0.85                                  # grasp face (normal -Y)
        self.tip = np.array([self.V[:, 0].max(), 0.0, 0.5 * (self.V[:, 2].min() + self.V[:, 2].max())])

    def parts(self, jaw_width: float):
        out = []
        for sgn, name in ((1, "finger_pos"), (-1, "finger_neg")):
            # mesh X -> -y_g (mount behind the tip), mesh -Y (grasp normal) -> -sgn x_g (towards the other finger), Z -> z
            # columns = images of mesh X, Y, Z: X -> +y_g so (x - 71) <= 0 lands BEHIND the tip; Y -> sgn x_g; Z -> -/+ z_g (det +1)
            R = np.array([[0, sgn, 0], [1, 0, 0], [0, 0, -1.0 if sgn > 0 else 1.0]])
            assert np.isclose(np.linalg.det(R), 1.0)
            Vg = (self.V - self.tip) @ R.T + np.array([sgn * jaw_width / 2, 0, 0])
            out.append((name, Vg, self.F, np.where(self.pad[:, None], np.array([[150, 150, 155]]), np.array([[35, 35, 38]]))))
        return out


COLORS = {"jaw_neg": (60, 60, 230), "jaw_pos": (230, 120, 40), "base": (170, 170, 170),
          "finger_neg": (40, 40, 40), "finger_pos": (40, 40, 40), "pad": (175, 175, 180)}   # BGR: -x jaw red, +x jaw blue


def project(P_cam: np.ndarray, intr: dict) -> np.ndarray:
    """(N,3) camera-frame points -> (N,2) pixels with the camera's own model (fisheye KB or pinhole)."""
    K = np.asarray(intr["K"], float) if "K" in intr else np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], float)
    obj = P_cam.reshape(-1, 1, 3).astype(np.float64); rv = tv = np.zeros(3)
    if intr.get("model") == "kannala_brandt":
        px, _ = cv2.fisheye.projectPoints(obj, rv, tv, K, np.asarray(intr["D"], float).reshape(4, 1))
    else:
        d = intr.get("distortion")
        D = np.array([d.get(k, 0) for k in ("k1", "k2", "p1", "p2", "k3")], float) if isinstance(d, dict) else np.asarray(d or np.zeros(5), float)
        px, _ = cv2.projectPoints(obj, rv, tv, K, D)
    return px.reshape(-1, 2)


class VirtualGripperRenderer:
    """Interface: render(rgb, pose_cam_gripper, jaw_width, camera_intrinsics) -> (image, mask, depth)."""

    def render(self, rgb, pose_cam_gripper, jaw_width, camera_intrinsics):
        raise NotImplementedError


class OpenCVRenderer(VirtualGripperRenderer):
    def __init__(self, mesh: PrimitiveGripperMesh | None = None, light_dir=(0.3, -0.5, -1.0), alpha: float = 1.0):
        self.mesh = mesh or PrimitiveGripperMesh()
        self.light = np.asarray(light_dir, float) / np.linalg.norm(light_dir)
        self.alpha = alpha

    def render(self, rgb, pose_cam_gripper, jaw_width, camera_intrinsics):
        img = rgb.copy(); H, W = img.shape[:2]
        mask = np.zeros((H, W), np.uint8); depth = np.full((H, W), np.inf, np.float32)
        T = np.asarray(pose_cam_gripper, float)
        polys = []
        for part in self.mesh.parts(float(jaw_width)):
            name, V, F = part[:3]; cols = part[3] if len(part) > 3 else None
            Vc = V @ T[:3, :3].T + T[:3, 3]
            if (Vc[:, 2] <= 0.005).any():
                continue                                          # behind / at the camera: skip the part
            uv = project(Vc, camera_intrinsics)
            P = Vc[F]; nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
            ctr = P.mean(1); facing = np.einsum("ij,ij->i", nrm, ctr) < 0
            if getattr(self.mesh, "cull", True):
                sel = np.flatnonzero(facing | (np.array([name == "pad"] * len(F))))
            else:
                sel = np.arange(len(F)); nrm = np.where(facing[:, None], nrm, -nrm)
            shade = 0.35 + 0.65 * np.clip(-(nrm @ self.light), 0, None)
            base = np.asarray(cols, float) if cols is not None else np.tile(COLORS[name], (len(F), 1))
            for f in sel:
                polys.append((float(ctr[f, 2]), uv[F[f]], tuple(int(c) for c in base[f] * shade[f])))
        overlay = img.copy()
        for z, pts, col in sorted(polys, key=lambda t: -t[0]):   # far -> near (painter's order)
            p = np.round(pts).astype(np.int32)
            cv2.fillConvexPoly(overlay, p, col, cv2.LINE_AA); cv2.fillConvexPoly(mask, p, 255)
        depth = None                                               # z-buffer output: TODO (spec section 12), MVP = always in front
        img = cv2.addWeighted(overlay, self.alpha, img, 1 - self.alpha, 0) if self.alpha < 1 else overlay
        return img, mask, depth


def draw_debug(img, intr, kp=None, pose=None, axis_len=0.03):
    """Debug overlay (spec §10): skeleton, thumb (red) / index (blue) tips, thumb-index line, TCP (green), gripper axes x/y/z = R/G/B."""
    from .hand_pose import INDEX_TIP, THUMB_TIP
    bones = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14),
             (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]
    if kp is not None and np.isfinite(kp).all() and (kp[:, 2] > 0.005).all():
        uv = project(kp, intr).astype(int)
        for a, b in bones:
            cv2.line(img, tuple(uv[a]), tuple(uv[b]), (220, 220, 220), 1, cv2.LINE_AA)
        cv2.line(img, tuple(uv[THUMB_TIP]), tuple(uv[INDEX_TIP]), (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(img, tuple(uv[THUMB_TIP]), 5, (0, 0, 255), -1); cv2.circle(img, tuple(uv[INDEX_TIP]), 5, (255, 0, 0), -1)
    if pose is not None:
        T = np.asarray(pose); o = T[:3, 3]
        pts = np.array([o] + [o + T[:3, i] * axis_len for i in range(3)])
        if (pts[:, 2] > 0.005).all():
            uv = project(pts, intr).astype(int)
            for i, col in enumerate(((0, 0, 255), (0, 200, 0), (255, 0, 0))):
                cv2.arrowedLine(img, tuple(uv[0]), tuple(uv[i + 1]), col, 2, cv2.LINE_AA, tipLength=0.25)
            cv2.circle(img, tuple(uv[0]), 5, (0, 255, 0), -1)
    return img
