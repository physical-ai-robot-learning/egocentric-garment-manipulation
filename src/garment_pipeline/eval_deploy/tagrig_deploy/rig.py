"""The persistent hardware / kinematics side shared by every run: wrist cameras, live Orbbec, robot poller, arms + collision
model, and the policy cache.
"""
from __future__ import annotations
import json
import os
import pathlib
import time

import numpy as np

from garment_pipeline.eval_deploy import act_tagrig_core as core
from garment_pipeline.eval_deploy.tagrig_deploy.bridge import RobotPoller
from garment_pipeline.eval_deploy.tagrig_deploy.cameras import auto_remap_cams, cam_key, FakeWristCam, GlobalCam, recorder_running, umi_location_warning, uvc_list, WristCam
from garment_pipeline.eval_deploy.tagrig_deploy.paths import CAMS_CFG, say, START_FILE


# ---------------------------------------------------------------- run
class Rig:
    """persistent hardware/kinematics side shared by every run (CLI: one run; UI: many): wrist cameras, live Orbbec, 25 Hz robot
    poller (read-only HttpBridge), pinocchio arms + capsule collision, and a cache of loaded policies"""
    def __init__(self, a):
        from visual_teleop import config as vcfg
        from visual_teleop.robot.bridge import HttpBridge
        from visual_teleop.tools.robot_moves import READY_Q_DEG
        rec = recorder_running()
        if rec: raise SystemExit(f"녹화/카메라 프로세스가 실행 중 -- 끄고 다시: {rec}")
        self.c = vcfg.load(); self.url = getattr(a, "bridge_url", None) or self.c.robot["base_url"]
        self.qr = {s: np.radians(READY_Q_DEG[s]) for s in core.SIDES}
        cfg = json.loads(CAMS_CFG.read_text()); lst = uvc_list()
        if not [x for x in os.environ.get("DEPLOY_FAKE_WRIST", "").split(",") if x in core.SIDES]: cfg = auto_remap_cams(cfg, lst)
        self.cfg = cfg
        fake = [x for x in os.environ.get("DEPLOY_FAKE_WRIST", "").split(",") if x in core.SIDES]
        if fake: say(f"주의: {fake} 손목 카메라 = 검은 화면 (DEPLOY_FAKE_WRIST, 학습 분포 밖 -- 테스트 전용)")
        for s in core.SIDES:
            if s in fake: continue
            if not cfg.get(s, {}).get("location") or cfg[s]["location"].lower() not in lst.lower(): raise SystemExit(f"{s} 손목 카메라 USB 위치 {cfg.get(s, {}).get('location')} 없음 -- 연결/매핑 확인 (wave 테스트)")
        if cfg.get("cam_key") != cam_key() and not fake: raise SystemExit("USB 카메라 구성이 바뀜 -- 먼저: act_tagrig_deploy.py cams (index 다시 찾기)")
        self.wave_ok = bool(cfg.get("wave_test", {}).get("confirmed"))
        if not self.wave_ok: say("주의: 손목 카메라 좌우가 wave 테스트로 아직 확인되지 않음 (act_tagrig_cams.json wave_test)")
        umi_location_warning(cfg)
        self.arms, self.ra = core.make_arms(); self.col = core.Collision(self.arms)
        self.fake_q = None
        if getattr(a, "assume_start", False):
            if not START_FILE.exists(): raise SystemExit("start_pose.json 없음 -- 먼저 `start`")
            self.fake_q = {s: np.asarray(json.loads(START_FILE.read_text())["q"][s]) for s in core.SIDES}
        elif getattr(a, "assume_ready", False): self.fake_q = self.qr
        self.br = None if self.fake_q is not None else HttpBridge(self.url, live=False, timeout_s=max(self.c.robot["request_timeout_s"], 0.3))
        self.rob = RobotPoller(self.br, self.fake_q if self.fake_q is not None else self.qr, self.fake_q is not None)
        align = cfg.get("wrist_align") if getattr(a, "wrist_align", False) else None
        self.replay = bool(getattr(a, "replay_global", ""))
        self.gl = GlobalCam(("dir:" + a.replay_global) if self.replay else a.global_url)
        self.wr = {s: (FakeWristCam(s) if s in fake else WristCam(s, cfg[s]["location"], cfg[s]["index"], a.exposure, (align or {}).get(s))) for s in core.SIDES}
        self.runners = {}

    def start(self):
        for x in (*self.wr.values(), self.gl, self.rob): x.start()
        t_w = time.monotonic()
        while any(x.ring.latest()[1] is None for x in (*self.wr.values(), self.gl)):
            if time.monotonic() - t_w > 8: raise SystemExit(f"카메라 프레임 없음: { {s: w.n for s, w in self.wr.items()} } global {self.gl.n}")
            time.sleep(0.05)
        return self

    def runner(self, ckpt, device, n_samples):
        key = (str(ckpt), n_samples)
        if key not in self.runners:
            self.runners = {}                       # keep ONE model in memory (DiT is 1.1 GB)
            is_xvla = '"xvla"' in (pathlib.Path(ckpt) / "pretrained_model" / "config.json").read_text() if (pathlib.Path(ckpt) / "pretrained_model" / "config.json").exists() else False
            self.runners[key] = core.XvlaRunner(ckpt) if is_xvla else core.PolicyRunner(ckpt, device, n_samples=n_samples)   # 10-28 X-VLA = server
        return self.runners[key]

    def close(self):
        for x in (*self.wr.values(), self.gl, self.rob): x.stop = True
        if self.br is not None: self.br.close()
