"""[2026-10-28] X-VLA (Exp 3 REL32-bin, GOLD-281 domain 21) inference server for the tag-rig deploy UI on the Mac (MPS).

Separate process because X-VLA needs pod2's lerobot fork + transformers 4.57.6 (the DiT deploy venv has lerobot 3c15fd8 /
tf 5.5.4).
Inference = the training contract and the offline evaluator (scratchpad xvla_eval/xvla_gold_eval.py): rel_relonly_bce.install()
(action dims 20:32 zeroed in/out, action grippers 9/19 zeroed at the transformer input), domain = the checkpoint's processor
fallback, task "laundry folding", images = the deploy's 384x216 frames letterboxed into 320x240 exactly like scratchpad
exp3/gold_export.py, sigmoid on 9/19 (P(open)). The caller (core XvlaRunner) converts state / chunk between the deploy's eef_kin
axes and the dataset-TCP axes (Ry(-90)). Deploy stage: policy server.
  POST /load     {"ckpt": <step dir or pretrained_model>}            -> {"ok", "info"}
  POST /predict  npz {global, left_wrist, right_wrist: uint8 HxWx3 RGB, state20: float32 (20,), seed: int}
                 -> npz {chunk: (32, 32)}
                 optional rtc_Y (32, 32) post-processed rows (grippers = P(open)) + rtc_W (32,): Real-Time Chunking (installRtc)
  GET  /status
run: PYTHONPATH=<repo>/src:~/xvla_deploy/pkgs:~/xvla_deploy/lerobot-seeed/src:~/xvla_deploy/umi_bridge/umi76 \\
         ~/xvla-mac/bin/python \\
         -m laundry_deploy.xvla_server [--port 8775]
"""

from __future__ import annotations
import argparse
import io
import json
import pathlib
import sys
import threading
import time

import cv2
import numpy as np
import torch

TASK = "laundry folding"
GRIP = (9, 19)  # gripper columns of the 32-wide X-VLA action
CAMS = ("global", "left_wrist", "right_wrist")
RENAME = {
    "observation.images.global": "observation.images.image",
    "observation.images.left_wrist": "observation.images.image2",
    "observation.images.right_wrist": "observation.images.image3",
}
TRAIN_HW = (240, 320)  # training image size (gold_export.letterbox)
INNER_WH = (320, 180)  # 16:9 frame inside the letterbox
INNER_ROWS = (30, 210)  # rows of the letterbox the frame occupies
ACTION_DIM = 32
ACTIVE_ACTION_DIM = 20  # dims 20:32 are zeroed (rel_relonly_bce)
STD_FLOOR = 1e-8
RTC_P_CLIP = (0.02, 0.98)  # P(open) of the previous plan clipped before the logit
DEFAULT_PORT = 8775
MS_PER_S = 1e3


def letterbox(img: np.ndarray) -> np.ndarray:
    """= gold_export.letterbox: 384x216 (16:9) -> 320x180 INTER_AREA into rows 30..210 of a black 320x240."""
    if img.shape[:2] == TRAIN_HW:  # already the training 320x240 (parity tests)
        return img
    tSmall = cv2.resize(img, INNER_WH, interpolation=cv2.INTER_AREA)
    tOut = np.zeros((TRAIN_HW[0], TRAIN_HW[1], 3), np.uint8)
    tOut[INNER_ROWS[0] : INNER_ROWS[1]] = tSmall
    return tOut


def installRtc(modelCls) -> None:
    """[10-29] Real-Time Chunking (Black, Galliker, Levine 2025) for X-VLA, gradient-free form: X-VLA's transformer predicts the
    CLEAN chunk at every flow step, so the RTC soft mask is applied to that prediction -- action <- action + W * (Y - action) --
    before the next x_t = x1 t + action (1 - t). Rows the robot executes while this chunk is computed (W = 1) continue the
    previous plan exactly, the overlap fades out exponentially (RTC weights, deploy side), later rows are free. The full RTC
    guidance (a vector-Jacobian product through the 0.9B transformer every step) would take the ~950 ms MPS inference to
    ~2.5-3 s. Off unless the model instance's mRtc is set to (Y, W)."""
    if getattr(modelCls, "mRtcPatched", False):
        return

    def generate_actions(self, input_ids, image_input, image_mask, domain_id, proprio, steps, x1=None):
        self.eval()
        tDtype = self._get_target_dtype()
        image_input = image_input.to(dtype=tDtype)
        proprio = proprio.to(dtype=tDtype)
        tEnc = self.forward_vlm(input_ids, image_input, image_mask)
        tB = input_ids.shape[0]
        if x1 is None:
            x1 = torch.randn(tB, self.chunk_size, self.dim_action, device=proprio.device, dtype=tDtype)
        else:
            x1 = x1.to(device=proprio.device, dtype=tDtype)
        tAction = torch.zeros_like(x1)
        steps = max(1, int(steps))
        tRtc = getattr(self, "mRtc", None)
        for tI in range(steps, 0, -1):
            tT = torch.full((tB,), tI / steps, device=proprio.device, dtype=tDtype)
            tXt = x1 * tT.view(-1, 1, 1) + tAction * (1 - tT).view(-1, 1, 1)
            tProprioM, tXtM = self.action_space.preprocess(proprio, tXt)
            tAction = self.transformer(domain_id=domain_id, action_with_noise=tXtM, proprio=tProprioM, t=tT, **tEnc)
            if tRtc is not None:
                tAction = tAction + tRtc[1] * (tRtc[0] - tAction)
        return self.action_space.postprocess(tAction)

    modelCls.generate_actions = generate_actions
    modelCls.mRtcPatched = True


class XvlaModel:
    """One loaded X-VLA checkpoint with its pre / post processors; predict() is serialised by mLock."""

    def __init__(self, device: str) -> None:
        self.mDevice = device
        self.mLock = threading.Lock()
        self.mCkpt = None
        self.mPolicy = None
        self.mInfo = None

    def load(self, ckpt) -> dict:
        import rel_relonly_bce

        rel_relonly_bce.install()
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        from safetensors.numpy import load_file

        tPm = pathlib.Path(ckpt)
        tPm = tPm / "pretrained_model" if (tPm / "pretrained_model").is_dir() else tPm
        with self.mLock:
            tT0 = time.time()
            self.mPolicy = XVLAPolicy.from_pretrained(str(tPm)).to(self.mDevice).eval()
            self.mPre, self.mPost = make_pre_post_processors(
                self.mPolicy.config,
                pretrained_path=str(tPm),
                preprocessor_overrides={
                    "device_processor": {"device": self.mDevice},
                    "rename_observations_processor": {"rename_map": RENAME},
                },
                postprocessor_overrides={"device_processor": {"device": self.mDevice}},
            )
            tPp = json.loads((tPm / "policy_preprocessor.json").read_text())
            tDomain = [tStep["config"].get("domain_id") for tStep in tPp["steps"] if tStep.get("registry_name") == "xvla_add_domain_id"]
            installRtc(type(self.mPolicy.model))
            tUn = load_file(str(next(tPm.glob("policy_postprocessor_step_*_unnormalizer_processor.safetensors"))))
            self.mActionMean = tUn["action.mean"].astype(np.float64)
            self.mActionStd = np.maximum(tUn["action.std"].astype(np.float64), STD_FLOOR)
            self.mCkpt = str(ckpt)
            self.mInfo = dict(
                ckpt=str(ckpt),
                domain=tDomain[0] if tDomain else None,
                chunk=self.mPolicy.config.chunk_size,
                denoise=self.mPolicy.config.num_denoising_steps,
                load_s=round(time.time() - tT0, 1),
            )
        return self.mInfo

    @torch.no_grad()
    def predict(self, imgs: dict, state20, seed: int, rtcY=None, rtcW=None) -> np.ndarray:
        """-> (32, 32) post-processed chunk with grippers 9 / 19 as P(open)."""
        tGrip = list(GRIP)
        with self.mLock:
            if rtcY is not None:
                # post-processed rows -> the transformer's output space (MEAN_STD normalised, grippers = logits)
                tY = (np.asarray(rtcY, np.float64) - self.mActionMean) / self.mActionStd
                tP = np.clip(np.asarray(rtcY, np.float64)[:, tGrip], *RTC_P_CLIP)
                tY[:, tGrip] = np.log(tP / (1 - tP))
                tY[:, ACTIVE_ACTION_DIM:] = 0.0
                self.mPolicy.model.mRtc = (
                    torch.from_numpy(tY).float()[None].to(self.mDevice),
                    torch.from_numpy(np.asarray(rtcW, np.float32)).reshape(1, -1, 1).to(self.mDevice),
                )
            else:
                self.mPolicy.model.mRtc = None
            tObs = {}
            for tCam in CAMS:
                tChw = torch.from_numpy(np.ascontiguousarray(letterbox(imgs[tCam]))).permute(2, 0, 1)
                tObs[f"observation.images.{tCam}"] = tChw.float()[None] / 255.0
            tObs["observation.state"] = torch.from_numpy(np.asarray(state20, np.float32))[None]
            tObs["task"] = [TASK]
            tBatch = self.mPre(tObs)
            tGenerator = torch.Generator().manual_seed(int(seed))
            tNoise = torch.randn(1, self.mPolicy.config.chunk_size, ACTION_DIM, generator=tGenerator).to(self.mDevice)
            tChunk = self.mPost(self.mPolicy.predict_action_chunk(tBatch, noise=tNoise))
            tA = tChunk[0].detach().float().cpu().numpy().astype(np.float64)
        tA[:, tGrip] = 1.0 / (1.0 + np.exp(-tA[:, tGrip]))
        return tA


def createApp(model: XvlaModel, device: str):
    from flask import Flask, Response, jsonify, request

    tApp = Flask("xvla_server")

    @tApp.post("/load")
    def load():
        tCkpt = (request.json or {}).get("ckpt")
        if model.mCkpt == tCkpt:
            if model.mInfo is None:
                # kept as before: a /load without "ckpt" before any load answered 500 (the info attribute did not exist yet)
                raise AttributeError("'Model' object has no attribute 'info'")
            return jsonify(ok=True, info=model.mInfo)
        try:
            return jsonify(ok=True, info=model.load(tCkpt))
        except Exception as tErr:
            return jsonify(ok=False, error=f"{type(tErr).__name__}: {tErr}"), 500

    @tApp.post("/predict")
    def predict():
        if model.mPolicy is None:
            return Response("no checkpoint loaded", status=409)
        tZ = np.load(io.BytesIO(request.get_data()))
        tT0 = time.time()
        tRtcY = tZ["rtc_Y"] if "rtc_Y" in tZ.files else None
        tRtcW = tZ["rtc_W"] if "rtc_W" in tZ.files else None
        tChunk = model.predict({tCam: tZ[tCam] for tCam in CAMS}, tZ["state20"], int(tZ["seed"]), tRtcY, tRtcW)
        tBuf = io.BytesIO()
        np.savez(tBuf, chunk=tChunk, ms=np.float64((time.time() - tT0) * MS_PER_S))
        return Response(tBuf.getvalue(), mimetype="application/octet-stream")

    @tApp.get("/status")
    def status():
        return jsonify(ok=True, ckpt=model.mCkpt, info=model.mInfo, device=device)

    return tApp


def main(argv: list[str] | None = None) -> int:
    tAp = argparse.ArgumentParser()
    tAp.add_argument("--port", type=int, default=DEFAULT_PORT)
    tAp.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    tAp.add_argument("--ckpt", default="")
    tArgs = tAp.parse_args(argv)
    tModel = XvlaModel(tArgs.device)
    tApp = createApp(tModel, tArgs.device)
    import logging

    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    if tArgs.ckpt:
        print("[xvla] loaded", tModel.load(tArgs.ckpt), flush=True)
    print(f"[xvla] http://127.0.0.1:{tArgs.port} device {tArgs.device}", flush=True)
    tApp.run(host="127.0.0.1", port=tArgs.port, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
