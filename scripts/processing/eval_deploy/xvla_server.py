#!/usr/bin/env python3
"""[2026-10-28] X-VLA (Exp 3 REL32-bin, GOLD-281 domain 21) inference server for the tag-rig deploy UI on the Mac (MPS).
Separate process because X-VLA needs pod2's lerobot fork + transformers 4.57.6 (the DiT deploy venv has lerobot 3c15fd8 / tf 5.5.4).
Inference = the training contract and the offline evaluator (scratchpad xvla_eval/xvla_gold_eval.py): rel_relonly_bce.install()
(action dims 20:32 zeroed in/out, action grippers 9/19 zeroed at the transformer input), domain = the checkpoint's processor fallback,
task "laundry folding", images = the deploy's 384x216 frames letterboxed into 320x240 exactly like scratchpad exp3/gold_export.py,
sigmoid on 9/19 (P(open)).  The caller converts state/chunk between the deploy's eef_kin axes and the dataset-TCP axes (Ry(-90)).
  POST /load     {"ckpt": <step dir or pretrained_model>}            -> {"ok", "info"}
  POST /predict  npz {global, left_wrist, right_wrist: uint8 HxWx3 RGB, state20: float32 (20,), seed: int}  -> npz {chunk: (32, 32)}
                 optional rtc_Y (32, 32) post-processed rows (grippers = P(open)) + rtc_W (32,): Real-Time Chunking (see install_rtc)
run: PYTHONPATH=~/xvla_deploy/pkgs:~/xvla_deploy/lerobot-seeed/src:~/xvla_deploy/umi_bridge/umi76 ~/xvla-mac/bin/python deploy/xvla_server.py [--port 8775]"""
import argparse, io, json, pathlib, threading, time

import cv2
import numpy as np
import torch

TASK = "laundry folding"; GRIP = (9, 19); CAMS = ("global", "left_wrist", "right_wrist")
RENAME = {"observation.images.global": "observation.images.image", "observation.images.left_wrist": "observation.images.image2",
          "observation.images.right_wrist": "observation.images.image3"}


def letterbox(img):
    """= gold_export.letterbox: 384x216 (16:9) -> 320x180 INTER_AREA into rows 30..210 of a black 320x240"""
    if img.shape[:2] == (240, 320): return img                       # already the training 320x240 (parity tests)
    s = cv2.resize(img, (320, 180), interpolation=cv2.INTER_AREA); out = np.zeros((240, 320, 3), np.uint8); out[30:210] = s; return out


def install_rtc(model_cls):
    """[10-29] Real-Time Chunking (Black, Galliker, Levine 2025) for X-VLA, gradient-free form: X-VLA's transformer predicts the CLEAN chunk
    at every flow step, so the RTC soft mask is applied to that prediction -- action <- action + W * (Y - action) -- before the next
    x_t = x1 t + action (1 - t).  Rows the robot executes while this chunk is computed (W = 1) continue the previous plan exactly, the
    overlap fades out exponentially (RTC weights, deploy side), later rows are free.  The full RTC guidance (vector-Jacobian product
    through the 0.9B transformer every step) would take the ~950 ms MPS inference to ~2.5-3 s.  Off unless self._rtc is set."""
    if getattr(model_cls, "_rtc_patched", False): return
    def generate_actions(self, input_ids, image_input, image_mask, domain_id, proprio, steps, x1=None):
        self.eval(); dt_ = self._get_target_dtype(); image_input = image_input.to(dtype=dt_); proprio = proprio.to(dtype=dt_)
        enc = self.forward_vlm(input_ids, image_input, image_mask); B = input_ids.shape[0]
        x1 = torch.randn(B, self.chunk_size, self.dim_action, device=proprio.device, dtype=dt_) if x1 is None else x1.to(device=proprio.device, dtype=dt_)
        action = torch.zeros_like(x1); steps = max(1, int(steps)); rtc = getattr(self, "_rtc", None)
        for i in range(steps, 0, -1):
            t = torch.full((B,), i / steps, device=proprio.device, dtype=dt_)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)
            action = self.transformer(domain_id=domain_id, action_with_noise=x_t_m, proprio=proprio_m, t=t, **enc)
            if rtc is not None: action = action + rtc[1] * (rtc[0] - action)
        return self.action_space.postprocess(action)
    model_cls.generate_actions = generate_actions; model_cls._rtc_patched = True


class Model:
    def __init__(self, device):
        self.device = device; self.lock = threading.Lock(); self.ckpt = None; self.pol = None

    def load(self, ckpt):
        import rel_relonly_bce
        rel_relonly_bce.install()
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors
        pm = pathlib.Path(ckpt); pm = pm / "pretrained_model" if (pm / "pretrained_model").is_dir() else pm
        with self.lock:
            t0 = time.time(); self.pol = XVLAPolicy.from_pretrained(str(pm)).to(self.device).eval()
            self.pre, self.post = make_pre_post_processors(self.pol.config, pretrained_path=str(pm),
                                                           preprocessor_overrides={"device_processor": {"device": self.device}, "rename_observations_processor": {"rename_map": RENAME}},
                                                           postprocessor_overrides={"device_processor": {"device": self.device}})
            pp = json.loads((pm / "policy_preprocessor.json").read_text())
            dom = [s["config"].get("domain_id") for s in pp["steps"] if s.get("registry_name") == "xvla_add_domain_id"]
            install_rtc(type(self.pol.model))
            from safetensors.numpy import load_file
            un = load_file(str(next(pm.glob("policy_postprocessor_step_*_unnormalizer_processor.safetensors"))))
            self.a_mean, self.a_std = un["action.mean"].astype(np.float64), np.maximum(un["action.std"].astype(np.float64), 1e-8)
            self.ckpt = str(ckpt); self.info = dict(ckpt=str(ckpt), domain=dom[0] if dom else None, chunk=self.pol.config.chunk_size,
                                                     denoise=self.pol.config.num_denoising_steps, load_s=round(time.time() - t0, 1))
        return self.info

    @torch.no_grad()
    def predict(self, imgs, state20, seed, rtc_Y=None, rtc_W=None):
        with self.lock:
            if rtc_Y is not None:          # post-processed rows -> the transformer's output space (MEAN_STD normalised, grippers = logits)
                Y = (np.asarray(rtc_Y, np.float64) - self.a_mean) / self.a_std; p = np.clip(np.asarray(rtc_Y, np.float64)[:, list(GRIP)], 0.02, 0.98)
                Y[:, list(GRIP)] = np.log(p / (1 - p)); Y[:, 20:] = 0.0
                self.pol.model._rtc = (torch.from_numpy(Y).float()[None].to(self.device), torch.from_numpy(np.asarray(rtc_W, np.float32)).reshape(1, -1, 1).to(self.device))
            else: self.pol.model._rtc = None
            obs = {f"observation.images.{c}": torch.from_numpy(np.ascontiguousarray(letterbox(imgs[c]))).permute(2, 0, 1).float()[None] / 255.0 for c in CAMS}
            obs["observation.state"] = torch.from_numpy(np.asarray(state20, np.float32))[None]; obs["task"] = [TASK]
            b = self.pre(obs)
            noise = torch.randn(1, self.pol.config.chunk_size, 32, generator=torch.Generator().manual_seed(int(seed))).to(self.device)
            a = self.post(self.pol.predict_action_chunk(b, noise=noise))[0].detach().float().cpu().numpy().astype(np.float64)
        a[:, list(GRIP)] = 1.0 / (1.0 + np.exp(-a[:, list(GRIP)])); return a


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8775); ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--ckpt", default=""); a = ap.parse_args()
    from flask import Flask, Response, jsonify, request
    M = Model(a.device); app = Flask("xvla_server")
    import logging; logging.getLogger("werkzeug").setLevel(logging.ERROR)
    if a.ckpt: print("[xvla] loaded", M.load(a.ckpt), flush=True)

    @app.post("/load")
    def load():
        ck = (request.json or {}).get("ckpt")
        if M.ckpt == ck: return jsonify(ok=True, info=M.info)
        try: return jsonify(ok=True, info=M.load(ck))
        except Exception as e: return jsonify(ok=False, error=f"{type(e).__name__}: {e}"), 500

    @app.post("/predict")
    def predict():
        if M.pol is None: return Response("no checkpoint loaded", status=409)
        z = np.load(io.BytesIO(request.get_data()))
        t0 = time.time(); ch = M.predict({c: z[c] for c in CAMS}, z["state20"], int(z["seed"]), z["rtc_Y"] if "rtc_Y" in z.files else None, z["rtc_W"] if "rtc_W" in z.files else None)
        buf = io.BytesIO(); np.savez(buf, chunk=ch, ms=np.float64((time.time() - t0) * 1e3)); return Response(buf.getvalue(), mimetype="application/octet-stream")

    @app.get("/status")
    def status(): return jsonify(ok=True, ckpt=M.ckpt, info=getattr(M, "info", None), device=a.device)

    print(f"[xvla] http://127.0.0.1:{a.port} device {a.device}", flush=True)
    app.run(host="127.0.0.1", port=a.port, threaded=True)


if __name__ == "__main__":
    main()
