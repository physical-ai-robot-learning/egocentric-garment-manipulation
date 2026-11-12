"""Persisted UI settings (act_tagrig_ui_settings.json in $TAGRIG_STATE_DIR, laundry_common.paths -- same file, keys and defaults
as always), the checkpoint folders listed per policy, and the GO heartbeat timeout.
"""

from __future__ import annotations
import json

from laundry_common import paths as egopaths

from ..core.contract import ACT_CKPT, DIT_CKPT, TRAIN_ROOT

SETTINGS_FILE = egopaths.TAGRIG_STATE_DIR / "act_tagrig_ui_settings.json"  # was SETTINGS
XVLA_FT_CKPTS = (
    egopaths.CKPT_ROOT / "ckpts/runpod/pod2/XVLA-FT-GOLD281-D21-FROM-MIX3-100K-RELBIN32-B8-300K-FP32/checkpoints"
)  # was XVLA_FT
DEFAULTS = dict(
    policy="act",
    ckpt_act=str(ACT_CKPT),
    ckpt_dit=str(DIT_CKPT),
    ckpt_xvla=str(XVLA_FT_CKPTS / "100000"),
    commit_act=1,
    commit_dit=4,
    commit_xvla=8,
    speed=0.5,
    slow_start_speed=0.25,
    active_arms="both",
    grip_fake_open=False,
    rtc="blend",
    z_offset_mm=0.0,
    height_calib=False,
    tens=False,
    grip_cont=True,
    grip_mode="linear",
    grip_gain=10.0,
    secs=60.0,
    dit_steps=10,
    show_umi_ref=False,
)
CKPT_DIRS = {
    "act": [TRAIN_ROOT / "checkpoints_pod/act_tagrig_v1_1m/checkpoints"],  # was CK_DIRS
    "dit": [
        TRAIN_ROOT / "checkpoints_pod/dit_tagrig_v1/checkpoints",
        # 10-28: Exp 1 = DiT Soft-Fold pretrain 300k -> GOLD-281 300k (best offline val/test at 300k), SSD archive
        (egopaths.CKPT_ROOT / "ckpts/runpod/pod1/DITFT-SFPRE300K-FLAT300-A-GOLD281-BS16-300K/checkpoints"),
        # 10-30: UMI-style S03 (0.3x) GOLD-281 all-281 B64 BCE GSM, H100 (stopped at epoch 26) + UMI50 0.5x 150 epochs, 4090
        (egopaths.CKPT_ROOT / "ckpts/runpod/h100/DITFT-MIX3PT100K-S03ALL281-B64-BCE-GSM-E150-NOCLIP/checkpoints"),
        (egopaths.CKPT_ROOT / "ckpts/4090/DITFT-S03PT50K-UMI50-S05-B64ACC4-BCE-GSM-E150-NOCLIP/checkpoints"),
        # 10-30: today's pinching (A' 53, 0.5x) 1000-epoch run from E150, every 25 epochs; P10 test: 50 ep 18.5 mm vs E150 28.5
        (egopaths.CKPT_ROOT / "ckpts/4090/DITFT-UMI50E150-U1030AP-S05-B64ACC4-BCE-GSM-E1000-NOCLIP/checkpoints"),
        # 11-01: pinching 197 demos (all Cg, 0.5x, AUG) 500-epoch fine-tune from E1000 227795; P20: 25 ep 17.29, 50 ep 18.89
        (egopaths.CKPT_ROOT / "ckpts/4090/DITFT-E1000F227795-U200CG-S05-B64ACC4-BCE-GSM-AUG-E500-NOCLIP/checkpoints"),
        # 11-03: flattening 149 demos (10-30 48 + 11-02 101, Cg, 0.5x), UMI recipe 100 epochs from H100 663012, 4xH100;
        # CONTINUOUS jaw (no BCE head) -> deploy with grip_mode "binary"; val best ep80 012160 (60.89), test best ep90 013680
        # (64.36)
        (TRAIN_ROOT / "checkpoints/FLAT149_h100"),
        # 11-05: flattening 250 demos (11-03 + 11-04, Cg, 0.3x, gated masks), UMI recipe 100 epochs from H100 663012, 2x4090 B1024;
        # CONTINUOUS jaw -> grip_mode "binary"; early rollout candidates ep08 (val 60.74) + ep10-12
        (TRAIN_ROOT / "checkpoints/FLAT250_rollout"),
        # 11-06: FLAT-PGRASP best / last of S03 (0.3x, RunPod Ada) and S10 (1.0x, cluster); state 18, 19 = robot-side p_grasp
        # (core.pgrasp, needs $TAGRIG_STATE_DIR/pgrasp_roi.json); S10 moves at human speed -> start with a low UI speed
        (TRAIN_ROOT / "checkpoints/FLATPG_rollout"),
    ],
    "xvla": [XVLA_FT_CKPTS],
}  # 10-28: Exp 3 X-VLA through xvla_server.py (:8775, separate venv)
# 10-30: 1.0 s tripped two GO runs (heartbeat 1.04 s old) -- the DiT forward in this process holds the GIL while /api/hb waits,
# and a
# background browser tab throttles setInterval to 1 s.  Space / [정지] still stop at once.
HB_TIMEOUT_S = 3.0
SETTINGS_INDENT = 1


def loadSettings() -> dict:
    tSettings = dict(DEFAULTS)
    try:
        tSettings.update(json.loads(SETTINGS_FILE.read_text()))
    except Exception:
        pass
    return tSettings


def saveSettings(settings: dict) -> None:
    SETTINGS_FILE.write_text(json.dumps(settings, indent=SETTINGS_INDENT, ensure_ascii=False))


def listCkpts(kind: str) -> list:
    """Checkpoint step folders (with a pretrained_model/) of one policy kind, folder by folder, sorted. Was ckpts()."""
    return [
        str(tP)
        for tDir in CKPT_DIRS[kind]
        if tDir.exists()
        for tP in sorted(tDir.iterdir())
        if (tP / "pretrained_model").is_dir()
    ]
