"""[2026-10-08] Minimal stand-in for UMI's umi/common/pose_util.py (the upstream copy in ~/holobrain-mac-model/umi_pkg was removed).
Same conventions: pose = xyz + rotvec; rot6d = the first two ROWS of R; rot6d -> R by Gram-Schmidt on those rows."""
import numpy as np
from scipy.spatial.transform import Rotation


def pose_to_mat(pose):
    pose = np.asarray(pose, np.float64)
    out = np.zeros(pose.shape[:-1] + (4, 4)); out[..., :3, :3] = Rotation.from_rotvec(pose[..., 3:].reshape(-1, 3)).as_matrix().reshape(pose.shape[:-1] + (3, 3))
    out[..., :3, 3] = pose[..., :3]; out[..., 3, 3] = 1
    return out


def mat_to_rot6d(mat):
    mat = np.asarray(mat)
    return mat[..., :2, :].copy().reshape(mat.shape[:-2] + (6,))


def rot6d_to_mat(d6):
    d6 = np.asarray(d6, np.float64)
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    return np.stack((b1, b2, np.cross(b1, b2)), axis=-2)


def mat_to_pose10d(mat):
    mat = np.asarray(mat)
    return np.concatenate([mat[..., :3, 3], mat_to_rot6d(mat[..., :3, :3])], axis=-1)


def pose10d_to_mat(d10):
    d10 = np.asarray(d10, np.float64)
    out = np.zeros(d10.shape[:-1] + (4, 4)); out[..., :3, :3] = rot6d_to_mat(d10[..., 3:9]); out[..., :3, 3] = d10[..., :3]; out[..., 3, 3] = 1
    return out
