"""Persisted UI settings (act_tagrig_ui_settings.json in the eval_deploy folder -- or in $TAGRIG_STATE_DIR, egogarment.paths --
same file, keys and defaults as before the split), the checkpoint folders per policy and the heartbeat timeout.
"""
from __future__ import annotations
import json
import pathlib

from egogarment import paths as egopaths
from garment_pipeline.eval_deploy import act_tagrig_core as core


HERE = pathlib.Path(__file__).resolve().parent.parent     # eval_deploy/ (the folder of the old single file)
SETTINGS = egopaths.TAGRIG_STATE_DIR / "act_tagrig_ui_settings.json"   # default: HERE
XVLA_FT = (egopaths.CKPT_ROOT / "ckpts/runpod/pod2/XVLA-FT-GOLD281-D21-FROM-MIX3-100K-RELBIN32-B8-300K-FP32/checkpoints")
DEFAULTS = dict(policy="act", ckpt_act=str(core.CKPT), ckpt_dit=str(core.DIT_CKPT), ckpt_xvla=str(XVLA_FT / "100000"), commit_act=1, commit_dit=4, commit_xvla=8,
                speed=0.5, slow_start_speed=0.25, active_arms="both", grip_fake_open=False, rtc="blend", z_offset_mm=0.0, height_calib=False, tens=False, grip_cont=True, grip_mode="linear", grip_gain=10.0,
                secs=60.0, dit_steps=10, show_umi_ref=False)
CK_DIRS = {"act": [core.LDT / "checkpoints_pod/act_tagrig_v1_1m/checkpoints"],
           "dit": [core.LDT / "checkpoints_pod/dit_tagrig_v1/checkpoints",
                   # 10-28: Exp 1 = DiT Soft-Fold pretrain 300k -> GOLD-281 300k (best offline val/test at 300k), SSD archive
                   (egopaths.CKPT_ROOT / "ckpts/runpod/pod1/DITFT-SFPRE300K-FLAT300-A-GOLD281-BS16-300K/checkpoints"),
                   # 10-30: UMI-style S03 (0.3x) GOLD-281 all-281 B64 BCE GSM, H100 (stopped at epoch 26) + UMI50 0.5x 150 epochs, 4090
                   (egopaths.CKPT_ROOT / "ckpts/runpod/h100/DITFT-MIX3PT100K-S03ALL281-B64-BCE-GSM-E150-NOCLIP/checkpoints"),
                   (egopaths.CKPT_ROOT / "ckpts/4090/DITFT-S03PT50K-UMI50-S05-B64ACC4-BCE-GSM-E150-NOCLIP/checkpoints"),
                   # 10-30: today's pinching (A' 53, 0.5x) 1000-epoch run from E150, every 25 epochs; P10 test: 50 ep 18.5 mm vs E150 28.5
                   (egopaths.CKPT_ROOT / "ckpts/4090/DITFT-UMI50E150-U1030AP-S05-B64ACC4-BCE-GSM-E1000-NOCLIP/checkpoints"),
                   # 11-01: pinching 197 demos (all Cg, 0.5x, AUG) 500-epoch fine-tune from E1000 227795; P20: 25 ep 17.29, 50 ep 18.89
                   (egopaths.CKPT_ROOT / "ckpts/4090/DITFT-E1000F227795-U200CG-S05-B64ACC4-BCE-GSM-AUG-E500-NOCLIP/checkpoints")],
           "xvla": [XVLA_FT]}         # 10-28: Exp 3 X-VLA through deploy/xvla_server.py (:8775, separate venv)


# 10-30: 1.0 s tripped two GO runs (heartbeat 1.04 s old) -- the DiT forward in this process holds the GIL while /api/hb waits, and a
# background browser tab throttles setInterval to 1 s.  Space / [정지] still stop at once.
HB_TIMEOUT_S = 3.0


def load_settings():
    s = dict(DEFAULTS)
    try: s.update(json.loads(SETTINGS.read_text()))
    except Exception: pass
    return s


def save_settings(s): SETTINGS.write_text(json.dumps(s, indent=1, ensure_ascii=False))


def ckpts(kind):
    return [str(p) for d in CK_DIRS[kind] if d.exists() for p in sorted(d.iterdir()) if (p / "pretrained_model").is_dir()]
