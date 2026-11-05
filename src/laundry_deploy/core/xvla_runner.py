"""[10-28] Exp 3 X-VLA (REL32-bin, GOLD-281 domain 21) through xvla_server.py (separate process, :8775), same interface as
core.policy_runner.PolicyRunner (deploy stage).

Frames: the deploy (= label A) uses eef_kin axes; the X-VLA set uses dataset-TCP axes T_x = T_A @ Ry(-90) (scratchpad
exp3/gold_export.py), so state rel_x = Ry^T rel_A Ry and chunk rel_A = Ry rel_x Ry^T (Ry = Ry(-90)). state20 = RELCART20 task
anchor = proprio[0:20] (episode-anchor poses + grippers) in those axes; chunk out (32, 20) contract rows with grippers = P(open).
Workspace = the TCP range of the GOLD train root (Exp 1 DiT checkpoint stats, same data).
"""

from __future__ import annotations
import io

import numpy as np

from laundry_common import paths as egopaths
from laundry_shared.contracts.action_contract import T_to_pose9, inv_T, make_T, pose9_to_T

from .chunk import retime
from .contract import ACTION_DIM, CAMS, CHUNK_ROWS, GRIP_INDEX, POSE_SLICE, SIDES
from .errors import CheckpointError

XVLA_URL = "http://127.0.0.1:8775"
XVLA_KIND = "xvla"
XVLA_DIM = 32  # X-VLA action / chunk row width: [L xyz rot6d g | R xyz rot6d g | 0 ...]
XVLA_LEFT = slice(0, 9)
XVLA_RIGHT = slice(10, 19)
XVLA_GRIP = {"left": 9, "right": 19}
LOAD_TIMEOUT_S = 300  # first load of the 0.9B model on MPS
PREDICT_TIMEOUT_S = 30
WORKSPACE_MARGIN_M = 0.05
GSM_TAG = "-GSM"  # 10-29 *-GSM: trained with the jaw state = 0
NORMALIZER_GLOB = "policy_preprocessor_step_*_normalizer_processor.safetensors"


class XvlaRunner:
    RY = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    WS_STATS = (
        egopaths.CKPT_ROOT / "ckpts/runpod/pod1/DITFT-SFPRE300K-FLAT300-A-GOLD281-BS16-300K/checkpoints/300000/pretrained_model"
    )
    ROW_DT_S = 1.0 / 15.0  # X-VLA REL32 rows are 15 Hz; the deploy loop executes rows every ROW_DT_S of core.contract (20 Hz)
    mKind = XVLA_KIND
    mNObs = 1
    mGripFakeOpen = False  # 10-29 diagnostic (UI): feed the jaw state as OPEN (the k1 jaw copies the input state)
    mTimeScale: float | None = 1.0  # 10-29: set from the UI speed (runSession) -- slow motion ALONG the predicted path

    def __init__(self, ckpt, url: str = XVLA_URL) -> None:
        import requests

        self.mSession = requests.Session()
        self.mUrl = url
        self.mCkpt = str(ckpt)
        tReply = self.mSession.post(url + "/load", json={"ckpt": str(ckpt)}, timeout=LOAD_TIMEOUT_S).json()
        if not tReply.get("ok"):
            raise CheckpointError(f"X-VLA 서버 체크포인트 로드 실패: {tReply}")
        self.mInfo = tReply["info"]
        from safetensors.numpy import load_file

        self.mStats = load_file(str(next(self.WS_STATS.glob(NORMALIZER_GLOB))))
        self.mRh = make_T(self.RY, np.zeros(3))

    def _toX(self, pose9):
        return T_to_pose9(inv_T(self.mRh) @ pose9_to_T(np.asarray(pose9, np.float64)) @ self.mRh)

    def _fromX(self, pose9):
        return T_to_pose9(self.mRh @ pose9_to_T(np.asarray(pose9, np.float64)) @ inv_T(self.mRh))

    def item(self, imgsRgb: dict, state30, prev=None) -> dict:
        tS = np.asarray(state30, np.float64)
        if GSM_TAG in str(getattr(self, "mCkpt", "")):
            tGl, tGr = 0.0, 0.0
        elif self.mGripFakeOpen:
            tGl, tGr = 1.0, 1.0
        else:
            tGl, tGr = tS[GRIP_INDEX["left"]], tS[GRIP_INDEX["right"]]
        tState20 = np.r_[self._toX(tS[POSE_SLICE["left"]]), self._toX(tS[POSE_SLICE["right"]]), tGl, tGr].astype(np.float32)
        return dict(imgs={tCam: np.ascontiguousarray(imgsRgb[tCam]) for tCam in CAMS}, state20=tState20)

    def predict(self, item: dict, seed: int = 0, rtc: dict | None = None):
        """rtc: dict(Y=(32, 20) previous plan rows in the deploy's relative contract (eef_kin axes, jaws = P(open)), W=(32,)) ->
        sent in the X-VLA dataset-TCP axes; the server blends the flow's clean-chunk prediction (xvla_server.installRtc)."""
        tExtra = {}
        if rtc is not None:
            tY = np.zeros((CHUNK_ROWS, XVLA_DIM))
            tYd = np.asarray(rtc["Y"], np.float64)
            tY[:, XVLA_LEFT] = np.stack([self._toX(tRow) for tRow in tYd[:, POSE_SLICE["left"]]])
            tY[:, XVLA_RIGHT] = np.stack([self._toX(tRow) for tRow in tYd[:, POSE_SLICE["right"]]])
            tY[:, XVLA_GRIP["left"]], tY[:, XVLA_GRIP["right"]] = tYd[:, GRIP_INDEX["left"]], tYd[:, GRIP_INDEX["right"]]
            tExtra = dict(rtc_Y=tY.astype(np.float32), rtc_W=np.asarray(rtc["W"], np.float32))
        tBuf = io.BytesIO()
        np.savez(tBuf, state20=item["state20"], seed=np.int64(seed), **item["imgs"], **tExtra)
        tReply = self.mSession.post(self.mUrl + "/predict", data=tBuf.getvalue(), timeout=PREDICT_TIMEOUT_S)
        tReply.raise_for_status()
        tA = np.load(io.BytesIO(tReply.content))["chunk"]  # (32, 32) [L xyz rot6d g | R ... | 0]
        tChunk = np.zeros((CHUNK_ROWS, ACTION_DIM))
        tChunk[:, POSE_SLICE["left"]] = self._fromX(tA[:, XVLA_LEFT])
        tChunk[:, POSE_SLICE["right"]] = self._fromX(tA[:, XVLA_RIGHT])
        tChunk[:, GRIP_INDEX["left"]], tChunk[:, GRIP_INDEX["right"]] = tA[:, XVLA_GRIP["left"]], tA[:, XVLA_GRIP["right"]]
        if self.mTimeScale is not None:  # None = raw 15 Hz rows (deploy look-ahead runs them at their own tick)
            tChunk = self._retime(tChunk)
        return tChunk.copy(), tChunk

    def _retime(self, chunk):
        """[10-29] 15 Hz rows k/15 -> deploy rows at k * ROW_DT_S * mTimeScale.  Before this the 15 Hz rows were played at 20 Hz (1.33x
        too fast) and the UI speed only shrank each target toward the current pose (jittery creep)."""
        return retime(chunk, self.ROW_DT_S, self.mTimeScale)

    def workspace(self, margin: float = WORKSPACE_MARGIN_M) -> dict:
        tLo, tHi = self.mStats["pose_abs.min"], self.mStats["pose_abs.max"]
        return {tSide: (tLo[POSE_SLICE[tSide]][:3] - margin, tHi[POSE_SLICE[tSide]][:3] + margin) for tSide in SIDES}
