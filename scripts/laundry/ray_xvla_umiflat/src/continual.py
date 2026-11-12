"""[2026-11-11 user] daily continual fine-tuning ("매일 50-100개 쌓으면서 업그레이드"): WSD learning rate + recent-day sampling.

Both are OFF unless their env vars are set, so the 11-10 cosine run (train_umiflat.py without these vars) is unchanged.

WSD (XVLA_SCHED=wsd) -- warmup-stable-decay on the OPTIMIZER-step counter (global, continues across days via --resume):
  lr(step) = peak * min(1, (step + 1) / XVLA_WSD_WARMUP)                     stable branch (no decay ever)
  decay branch (XVLA_WSD_DECAY_FROM = s0, XVLA_WSD_DECAY_STEPS = D):
  lr(step) = peak * [1 - (1 - XVLA_WSD_FLOOR) * clip((step - s0) / D, 0, 1)]  linear to peak * FLOOR
  The stable checkpoint of day d is the start of day d+1 (new data, same optimizer state); the decay branch is only for
  evaluation / deployment and is never continued.  peak = the optimizer's base lr (--policy.optimizer_lr; the VLM group keeps its x0.1).
  On --resume lerobot loads the scheduler state (last_epoch, base_lrs) into this LambdaLR: the lambda is rebuilt from env.

Recent-day sampling (XVLA_RECENT_DAYS = n, XVLA_RECENT_FRAC = f): frames of episodes whose day (meta/episode_day.json
  {episode_index: "YYYYMMDD"}, written by datasets/xvla_cumulative.py) is among the n latest days get total probability f, all other
  frames 1 - f (uniform inside each group; with replacement).  Fewer than 2 groups -> plain shuffle.
"""
from __future__ import annotations

import json
import os
import pathlib

import torch


def _wsd_lambda():
    tWarm = int(os.environ.get("XVLA_WSD_WARMUP", "1000"))
    tFrom = os.environ.get("XVLA_WSD_DECAY_FROM")
    tD = int(os.environ.get("XVLA_WSD_DECAY_STEPS", "0"))
    tFloor = float(os.environ.get("XVLA_WSD_FLOOR", "0.025"))

    def lam(step: int) -> float:
        w = min(1.0, (step + 1) / max(1, tWarm))
        if tFrom is None or tD <= 0:
            return w
        x = min(1.0, max(0.0, (step - int(tFrom)) / tD))
        return w * (1.0 - (1.0 - tFloor) * x)

    return lam, f"WSD warmup {tWarm}, " + (f"decay {tFrom} -> {int(tFrom) + tD} to x{tFloor}" if tFrom and tD else "stable (no decay)")


def _episode_weights(dataset):
    tRoot = pathlib.Path(dataset.root)
    tDay = json.loads((tRoot / "meta/episode_day.json").read_text())
    tN = int(os.environ["XVLA_RECENT_DAYS"]); tF = float(os.environ.get("XVLA_RECENT_FRAC", "0.5"))
    tDays = sorted(set(tDay.values())); tRecent = set(tDays[-tN:])
    ep = dataset.meta.episodes
    tFrom, tTo = list(ep["dataset_from_index"]), list(ep["dataset_to_index"])
    tIdx = list(ep["episode_index"])
    w = torch.zeros(len(dataset), dtype=torch.double)
    tCnt = {True: 0, False: 0}
    for e, a, b in zip(tIdx, tFrom, tTo):
        tCnt[tDay[str(int(e))] in tRecent] += b - a
    if not tCnt[True] or not tCnt[False]:
        return None, f"recent sampling off (one group only: recent {tCnt[True]} / old {tCnt[False]} frames)"
    for e, a, b in zip(tIdx, tFrom, tTo):
        r = tDay[str(int(e))] in tRecent
        w[a:b] = (tF if r else 1.0 - tF) / tCnt[r]
    return w, (f"recent days {sorted(tRecent)}: {tCnt[True]} frames get {tF:.2f}, older {tDays[:-tN] or '-'}: "
               f"{tCnt[False]} frames get {1 - tF:.2f}")


def install(LT):
    """patch lerobot.scripts.lerobot_train (module object LT) in place"""
    if os.environ.get("XVLA_SCHED") == "wsd":
        _make = LT.make_optimizer_and_scheduler

        def make_optimizer_and_scheduler(cfg, policy):
            tOpt, _ = _make(cfg, policy)
            lam, tMsg = _wsd_lambda()
            print(f"[continual] {tMsg}; base lrs {[g['lr'] for g in tOpt.param_groups]}", flush=True)
            return tOpt, torch.optim.lr_scheduler.LambdaLR(tOpt, lam)

        LT.make_optimizer_and_scheduler = make_optimizer_and_scheduler

    if os.environ.get("XVLA_RECENT_DAYS"):
        _DL = torch.utils.data.DataLoader

        class DataLoader(_DL):
            def __init__(self, dataset, *a, **k):
                if hasattr(dataset, "meta") and hasattr(dataset, "root"):
                    w, tMsg = _episode_weights(dataset)
                    print(f"[continual] {tMsg}", flush=True)
                    if w is not None:
                        k["sampler"] = torch.utils.data.WeightedRandomSampler(w, num_samples=len(dataset), replacement=True)
                        k["shuffle"] = False
                super().__init__(dataset, *a, **k)

        LT.torch.utils.data.DataLoader = DataLoader
