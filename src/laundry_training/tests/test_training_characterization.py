"""Characterization of laundry_training on synthetic data: relative chunks, stats, loss masks, augmentation, the rel datasets,
the BCE jaw head, the metrics hook and the two launchers. Recorded against the pre-refactor modules (tagrig_common, rel_chunk,
train_dit_tagrig, train_dit_mix); the expected values live in golden_training.npz."""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as Rot

from laundry_shared.contracts.action_contract import T_to_pose9, make_T

from .conftest import REGISTRY, registerDataset

N_EP = (40, 50)
INDEX_OFFSET = 5


@pytest.fixture(scope="module")
def mods(stubbed):
    tImp = lambda name: stubbed.imp(f"laundry_training.{name}")
    return SimpleNamespace(rc=tImp("rel_chunk"), st=tImp("stats"), lm=tImp("loss_masks"), aug=tImp("augment"),
                           opt=tImp("env_options"),
                           lay=tImp("layout"), ds=tImp("datasets"), met=tImp("metrics"), pkg=stubbed.imp("laundry_training"),
                           stubbed=stubbed)


# ---------------------------------------------------------------- synthetic data
def _poses(rng, n):
    tR = Rot.from_rotvec(np.cumsum(rng.normal(0, 0.05, (n, 3)), 0)).as_matrix()
    tP = np.cumsum(rng.normal(0, 0.01, (n, 3)), 0) + [0.3, 0.1, 0.2]
    return make_T(tR, tP)


def _absRows(rng, n):
    tL, tR = _poses(rng, n), _poses(rng, n)
    tG = (rng.random((n, 2)) > 0.3).astype(np.float64)
    return np.concatenate([T_to_pose9(tL), T_to_pose9(tR), tG], -1).astype(np.float32)


def _columns(seed=0, withMask=True):
    tRng = np.random.default_rng(seed)
    tA, tP, tS, tE = [], [], [], []
    for tE_i, tN in enumerate(N_EP):
        tAbs = _absRows(tRng, tN + 1)
        tP.append(tAbs[:-1]); tA.append(tAbs[1:])
        tS.append(tRng.normal(0, 1, (tN, 30)).astype(np.float32)); tE.append(np.full(tN, tE_i))
    tN = sum(N_EP)
    tCols = {"action": np.concatenate(tA), "pose_abs": np.concatenate(tP), "observation.state": np.concatenate(tS),
             "episode_index": np.concatenate(tE), "index": np.arange(tN) + INDEX_OFFSET,
             "weight": np.where(np.arange(tN) < N_EP[0], 1.0, 0.3).astype(np.float32)[:, None]}
    tCols["observation.state"][:, 29] = 1.0
    tCols["observation.state"][:, 5] = 0.5                     # constant dim -> std / range floors
    if withMask:
        tCols["action_mask"] = (tRng.random((tN, 20)) > 0.15).astype(np.float32)
        tCols["anchor_ok"] = (tRng.random((tN, 4)) > 0.2).astype(np.float32)
    return tCols


def _register(tmp_path, name, withMask=True):
    tRoot = tmp_path / name
    (tRoot / "meta").mkdir(parents=True)
    tCols = _columns(withMask=withMask)
    registerDataset(tRoot, tCols, stats={"action": {"old": np.zeros(1)}, "observation.state": {}})
    return tRoot, tCols


# ---------------------------------------------------------------- rel_chunk
def test_rel_chunk_contract33_and_stats(mods, golden):
    tRng = np.random.default_rng(1)
    tChunk = _absRows(tRng, 33)
    golden.check("rc/to_contract33", mods.rc.toContract33(tChunk))
    tCols = _columns()
    for tStride in (1, 2):
        golden.check(f"rc/stats_s{tStride}", mods.rc.relativeActionStats(tCols["action"], tCols["episode_index"], stride=tStride))
    assert mods.rc.HORIZON == 33


def test_rel_chunk_patch_dataset(mods, golden, tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    tRoot, tCols = _register(tmp_path, "rc_patch")
    tRng = np.random.default_rng(2)
    REGISTRY[str(tRoot)]["items"] = [{"action": torch.from_numpy(_absRows(tRng, 33))}, {"action": torch.zeros(5, 20)}]
    tDs = LeRobotDataset("local/x", root=tRoot)
    mods.rc.patchDataset(tDs, stride=2)
    golden.check("rc/patch_stats", tDs.meta.stats["action"])
    assert type(tDs).__name__ == "LeRobotDatasetRelChunk"
    golden.check("rc/patch_item0", tDs[0]["action"])
    golden.check("rc/patch_item1", tDs[1]["action"])
    tCache = tRoot / "meta" / "rel_action_stats_h33.json"
    assert tCache.exists()
    tDs2 = LeRobotDataset("local/x", root=tRoot)
    mods.rc.patchDataset(tDs2)
    for tKey, tVal in tDs.meta.stats["action"].items():
        assert np.array_equal(tVal, tDs2.meta.stats["action"][tKey])


# ---------------------------------------------------------------- relative windows + stats
def test_act_rel_and_windows(mods, golden):
    tRng = np.random.default_rng(3)
    tAbs = _absRows(tRng, 33)
    golden.check("tc/act_rel", mods.rc.actRelative(tAbs[0], tAbs[1:]))
    tCols = _columns()
    for tKind in ("dit", "act"):
        for tStride in (1, 3):
            golden.check(f"tc/windows_{tKind}_{tStride}", mods.rc.relativeWindows(tCols["action"], tCols["pose_abs"],
                                                                                  tCols["episode_index"], tKind, tStride))
    tConsts = (mods.rc.H_ACT, mods.st.STD_FLOOR, mods.st.RANGE_FLOOR, mods.lay.GRIP_A, mods.lay.STATE_UNSCALED)
    assert tConsts == (32, 1e-2, 2e-2, [18, 19], [18, 19, 29])
    assert mods.lay.GROUPS == {"pos": [0, 1, 2, 9, 10, 11], "rot": list(range(3, 9)) + list(range(12, 18)), "grip": [18, 19]}


def test_stats_and_fixed_stats(mods, golden):
    tTc = mods.st
    tRng = np.random.default_rng(4)
    tX = tRng.normal(0, 1, (200, 6))
    tX[:, 2] = 0.3
    tX[:, 3] = tRng.normal(0, 1e-3, 200)
    tM = (tRng.random((200, 6)) > 0.3).astype(np.float32)
    tM[:, 4] = 0
    golden.check("tc/_stats", tTc._dimStats(tX))
    golden.check("tc/_stats_masked", tTc._dimStats(tX, tM))
    tSt, tAudit = tTc.fixedStats(tX, [5])
    golden.check("tc/fixed", tSt)
    golden.check("tc/fixed_audit", tAudit)
    tSt, tAudit = tTc.fixedStats(tX, [0], tM)
    golden.check("tc/fixed_masked", tSt)
    golden.check("tc/fixed_masked_audit", tAudit)


@pytest.mark.parametrize("kind", ["dit", "act"])
@pytest.mark.parametrize("withMask", [True, False])
def test_compute_all_and_install_stats(mods, golden, tmp_path, kind, withMask):
    tTc = mods.st
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    tRoot, _ = _register(tmp_path, f"ca_{kind}_{withMask}", withMask)
    tR = tTc.computeAll(tRoot, kind, stride=2)
    golden.check(f"tc/compute_all_{kind}_{withMask}", {tK: tR[tK] for tK in ("W", "MW", "action_stats", "state_stats",
                                                                             "action_audit", "state_audit")})
    tDs = LeRobotDataset("local/x", root=tRoot)
    tTc.installStats(tDs, kind, stride=2)
    tCache = tRoot / "meta" / f"tagrig_fixed_stats_{kind}_v4.json"
    assert tCache.exists()
    assert json.loads(tCache.read_text())["note"] == "train root only; floors + unscaled grippers, see training/tagrig_common.py"
    golden.check(f"tc/install_stats_{kind}_{withMask}", tDs.meta.stats)
    tDs2 = LeRobotDataset("local/x", root=tRoot)
    tTc.installStats(tDs2, kind)                             # from the cache
    golden.check(f"tc/install_stats_{kind}_{withMask}", tDs2.meta.stats)


# ---------------------------------------------------------------- loss masks + weights
def test_masked_elementwise(mods, golden):
    tTc = mods.lm
    tG = torch.Generator().manual_seed(5)
    tOut = torch.rand(3, 4, 20, generator=tG)
    tM = (torch.rand(3, 4, 20, generator=tG) > 0.3).float()
    tM[..., 18:20] = 0
    tPad = torch.zeros(3, 4, dtype=torch.bool)
    tPad[1, 2:] = True
    golden.check("tc/me_plain", tTc.maskedElementwise(tOut, tM))
    golden.check("tc/me_pad", tTc.maskedElementwise(tOut, tM, tPad))
    golden.check("tc/me_pad_nomask", tTc.maskedElementwise(tOut, tM, tPad, padMasking=False))
    golden.check("tc/me_groups", tTc.maskedElementwise(tOut, tM, None, True, {"a": list(range(10)), "b": list(range(10, 20))}))


@pytest.mark.parametrize("kind", ["act", "dit"])
@pytest.mark.parametrize("cross", ["0", "1"])
def test_install_masks_and_chunk_mask(mods, golden, tmp_path, monkeypatch, capsys, kind, cross):
    tTc = mods.lm
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    tRoot, tCols = _register(tmp_path, f"m_{kind}_{cross}")
    tDs = LeRobotDataset("local/x", root=tRoot)
    assert tTc.installMasks(tDs, kind) is tDs
    assert "[tagrig] action_mask:" in capsys.readouterr().out
    monkeypatch.setenv("MASK_CROSS_GAP", cross)
    tIdx = torch.tensor([INDEX_OFFSET, INDEX_OFFSET + 3, INDEX_OFFSET + 38, INDEX_OFFSET + 40, INDEX_OFFSET + 89])
    golden.check(f"tc/chunk_mask_{kind}_{cross}", tTc.chunkMask(tIdx))
    golden.check(f"tc/mask_tables_{kind}", {tK: tV for tK, tV in tTc.MASK_TABLES.items() if tK != "kind"})


def test_install_masks_without_mask_and_weights(mods, golden, tmp_path, capsys):
    tTc = mods.lm
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    tRoot, _ = _register(tmp_path, "nomask", withMask=False)
    tDs = LeRobotDataset("local/x", root=tRoot)
    assert tTc.installMasks(tDs, "dit") is tDs
    assert capsys.readouterr().out == "[tagrig] dataset has no action_mask -> unmasked\n"
    assert tTc.installWeights(tDs) is tDs
    assert "[tagrig] episode weights:" in capsys.readouterr().out
    golden.check("tc/weight_table", tTc.EPISODE_WEIGHTS["byIndex"])


# ---------------------------------------------------------------- augmentation
def test_hue_rotate(mods, golden):
    tG = torch.Generator().manual_seed(6)
    tY = torch.rand(2, 3, 5, 7, generator=tG)
    for tF in (0.0, 0.08, -0.3):
        golden.check(f"tc/hue_{tF}", mods.aug.hueRotate(tY, tF))


@pytest.mark.parametrize("sat_hue", [(0.0, 0.0), (0.5, 0.08)])
def test_aug_image(mods, golden, monkeypatch, sat_hue):
    tTc = mods.aug
    monkeypatch.setitem(mods.opt.AUG_P, "AUG_SAT", sat_hue[0])
    monkeypatch.setitem(mods.opt.AUG_P, "AUG_HUE", sat_hue[1])
    tG = torch.Generator().manual_seed(7)
    tX = torch.rand(2, 3, 12, 16, generator=tG)
    torch.manual_seed(11)
    golden.check(f"tc/aug_float_{sat_hue}", tTc.augImage(tX, 0.08, 5.0, 0.05))
    tU = (torch.rand(3, 12, 16, generator=tG) * 255).to(torch.uint8)
    torch.manual_seed(12)
    tOut = tTc.augImage(tU, 0.045, 2.0, 0.02)
    assert tOut.dtype == torch.uint8
    golden.check(f"tc/aug_uint8_{sat_hue}", tOut)


def test_env_defaults(mods):
    tTc = mods.opt
    assert tTc.AUG_P == dict(AUG_G_SHIFT=0.045, AUG_G_ROT=2.0, AUG_G_ZOOM=0.02, AUG_W_SHIFT=0.08, AUG_W_ROT=5.0, AUG_W_ZOOM=0.05,
                             AUG_BC=0.15, AUG_CH=0.05, AUG_SAT=0.0, AUG_HUE=0.0, AUG_B=0.15, AUG_C=0.15)
    assert (tTc.AUG, tTc.GRIP_STATE_MASK, tTc.GRIP_BCE, tTc.GRIP_BCE_W, tTc.WEIGHT_GRIP_ONLY, tTc.GRIP_HEAD_ONLY,
            tTc.GRIP_HEAD_RESET,
            tTc.GRAD_ACCUM) == (False, False, False, 1.0, False, False, False, 1)
    assert mods.lay.GRIP_STATE_DIMS == (18, 19)


# ---------------------------------------------------------------- rel datasets
def _items(rng):
    tImg = torch.rand(2, 3, 8, 10)
    return [{"action": torch.from_numpy(_absRows(rng, 33)), "pose_abs": torch.from_numpy(_absRows(rng, 1)[0]),
             "observation.state": torch.rand(2, 30), "observation.images.global": tImg.clone(),
             "observation.images.left_wrist": tImg.clone(), "observation.images.global_is_pad": torch.zeros(2, dtype=torch.bool)},
            {"action": torch.from_numpy(_absRows(rng, 32)), "pose_abs": torch.from_numpy(_absRows(rng, 1)[0]),
             "observation.state": torch.rand(30)}]


@pytest.mark.parametrize("aug,gsm", [(False, False), (True, True)])
def test_rel_datasets(mods, golden, tmp_path, monkeypatch, aug, gsm):
    tTc = mods.pkg
    torch.manual_seed(13)
    tRng = np.random.default_rng(13)
    tRoot = tmp_path / "items"
    registerDataset(tRoot, {"episode_index": np.zeros(2)}, items=_items(tRng))
    monkeypatch.setattr(mods.opt, "AUG", aug)
    monkeypatch.setattr(mods.opt, "GRIP_STATE_MASK", gsm)
    tDit = tTc.DitRelDataset("local/x", root=tRoot)
    tAct = tTc.ActRelDataset("local/x", root=tRoot)
    torch.manual_seed(14)
    golden.check(f"tc/dit_items_{aug}", [tDit[0], tDit[1]])
    golden.check(f"tc/act_items_{aug}", [tAct[0], tAct[1]])


def test_patch_datasets(mods, tmp_path):
    tTc = mods.ds
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    tRoot = tmp_path / "p"
    registerDataset(tRoot, {"episode_index": np.zeros(1)}, items=[])
    tDs = LeRobotDataset("local/x", root=tRoot)
    assert type(tTc.patchActDataset(tDs)) is tTc.ActRelDataset
    with pytest.raises(AssertionError):
        tTc.patchDitDataset(tDs)
    tDs2 = LeRobotDataset("local/x", root=tRoot)
    assert type(tTc.patchDitDataset(tDs2)) is tTc.DitRelDataset


# ---------------------------------------------------------------- BCE jaw head
HORIZON = 4
COND_DIM = 6


def _fakePolicyClass():
    class Encoder(torch.nn.Module):
        conditioning_dim = COND_DIM

        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(5, COND_DIM)

        def encode(self, batch):
            return self.lin(batch["obs"])

    class Policy(torch.nn.Module):
        def __init__(self, config, **kw):
            super().__init__()
            self.config = config
            self.observation_encoder = Encoder()
            self.noise_predictor = torch.nn.Linear(COND_DIM, 3)
            self.objective = SimpleNamespace(compute_loss=lambda net, batch, cond: (net(cond) ** 2).mean())

        def _prepare_batch(self, batch):
            return dict(batch)

    return Policy


def _gripBatch(g):
    tAct = torch.rand(3, HORIZON, 20, generator=g) * 2 - 1
    tPad = torch.zeros(3, HORIZON, dtype=torch.bool)
    tPad[2, -1] = True
    return {"obs": torch.rand(3, 5, generator=g), "action": tAct, "action_is_pad": tPad}


@pytest.mark.parametrize("variant", ["plain", "mask_weight", "head_only"])
def test_install_grip_bce(mods, golden, monkeypatch, capsys, variant):
    tTc = mods.opt
    tP = _fakePolicyClass()
    tMd = SimpleNamespace(MultiTaskDiTPolicy=tP)
    monkeypatch.setattr(tTc, "GRIP_HEAD_ONLY", variant == "head_only")
    monkeypatch.setattr(tTc, "GRIP_HEAD_RESET", variant == "head_only")
    monkeypatch.setattr(tTc, "WEIGHT_GRIP_ONLY", variant == "mask_weight")
    monkeypatch.setattr(tTc, "GRIP_BCE_W", 0.7)
    mods.pkg.installGripBce(tMd)
    assert "[tagrig] GRIP_BCE: grip_head" in capsys.readouterr().out
    mods.pkg.installGripBce(tMd)                                  # second install is a no-op
    assert capsys.readouterr().out == ""
    torch.manual_seed(21)
    tPolicy = tP(SimpleNamespace(horizon=HORIZON))
    tG = torch.Generator().manual_seed(22)
    tBatch = _gripBatch(tG)
    mods.lm.BATCH.clear()
    if variant == "mask_weight":
        mods.lm.BATCH["Mg"] = (torch.rand(3, HORIZON, 2, generator=tG) > 0.3).float()
        mods.lm.BATCH["w"] = torch.tensor([1.0, 0.3, 2.0])
    torch.manual_seed(23)
    tLoss, tOut = tPolicy.forward(tBatch)
    assert tOut is None
    tLoss.backward()
    golden.check(f"tc/grip_{variant}", {"loss": tLoss.detach(), "groups": mods.lm.BATCH["groups"],
                                        "grads": [tParam.grad if tParam.grad is not None else torch.zeros(0)
                                                  for tParam in tPolicy.parameters()],
                                        "trainable": [tParam.requires_grad for tParam in tPolicy.parameters()]})
    mods.lm.BATCH.clear()


# ---------------------------------------------------------------- metrics hook
class _Acc:
    def __init__(self):
        from contextlib import nullcontext
        self.autocast = nullcontext

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, params, maxNorm):
        import accelerate
        return accelerate.Accelerator.clip_grad_norm_(self, params, maxNorm)

    def unwrap_model(self, policy, keep_fp32_wrapper=True):
        return policy


class _LinPolicy(torch.nn.Module):
    def __init__(self, F):
        super().__init__()
        self.lin = torch.nn.Linear(5, 20)
        self.config = SimpleNamespace(do_mask_loss_for_padding=True)
        self.mF = F
        self.mUpdates = 0

    def forward(self, batch):
        tPred = self.lin(batch["obs"])[:, None].expand(-1, batch["action"].shape[1], -1)
        tErr = self.mF.mse_loss(tPred, batch["action"], reduction="none")
        return tErr.mean(), {"extra": 1.5, "skip": "x"}

    def update(self):
        self.mUpdates += 1


class _Tracker:
    """lerobot MetricsTracker: assigning a number updates the meter whose `.val` is read back"""

    def __setattr__(self, key, value):
        object.__setattr__(self, key, SimpleNamespace(val=float(value)) if isinstance(value, (int, float)) else value)


def _metricsBatch(g, start):
    return {"obs": torch.rand(4, 5, generator=g), "action": torch.rand(4, 33, 20, generator=g),
            "action_is_pad": torch.zeros(4, 33, dtype=torch.bool), "index": torch.arange(start, start + 4)}


@pytest.mark.parametrize("accum", [1, 2])
def test_install_metrics_update(mods, golden, tmp_path, monkeypatch, accum):
    tTc = mods.lm
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    tRoot, _ = _register(tmp_path, f"met{accum}")
    tDs = LeRobotDataset("local/x", root=tRoot)
    tTc.installMasks(tDs, "dit")
    tTc.installWeights(tDs)
    monkeypatch.setattr(mods.opt, "GRAD_ACCUM", accum)
    monkeypatch.setenv("GRAD_SPLIT_EVERY", "2")
    monkeypatch.setenv("METRICS_EVERY", "1")
    tPolicyModule = SimpleNamespace(F=torch.nn.functional)
    tLt = SimpleNamespace(_TAGRIG_OUT=str(tmp_path / f"out{accum}"))
    torch.manual_seed(31)

    def origUpdate(train_metrics, policy, batch, optimizer, grad_clip_norm, *a, **kw):
        policy.train()
        tLoss, tOd = policy.forward(batch)
        tLoss.backward()
        import accelerate
        tGn = accelerate.Accelerator().clip_grad_norm_(policy.parameters(), grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad()
        train_metrics.loss = float(tLoss)
        train_metrics.grad_norm = float(tGn)
        return train_metrics, tOd

    tLt.update_policy = origUpdate
    import accelerate
    tOrigClip = accelerate.Accelerator.clip_grad_norm_
    try:
        mods.met.installMetrics(tLt, tPolicyModule, "mse_loss")
        tPolicy = _LinPolicy(tPolicyModule.F)
        tOpt = torch.optim.SGD(tPolicy.parameters(), lr=0.1)
        tG = torch.Generator().manual_seed(32)
        tLosses = []
        for tStep in range(3):
            tTm = _Tracker()
            tB = _metricsBatch(tG, INDEX_OFFSET + 10 * tStep)
            tTm, tOd = tLt.update_policy(tTm, tPolicy, tB, tOpt, 0.5 if tStep < 2 else 0.0, accelerator=_Acc())
            tLosses.append(tTm.loss.val)
        tRecs = [json.loads(tX) for tX in (tmp_path / f"out{accum}" / "metrics.jsonl").read_text().splitlines()]
        for tRec in tRecs:
            tRec.pop("t")
        golden.check(f"tc/metrics_{accum}", {"recs": tRecs, "losses": tLosses, "params": list(tPolicy.parameters()),
                                             "updates": tPolicy.mUpdates})
    finally:
        accelerate.Accelerator.clip_grad_norm_ = tOrigClip
        mods.lm.BATCH.clear()


def test_metrics_nan_stops(mods, tmp_path, monkeypatch):
    tTc = mods.lm
    monkeypatch.setattr(mods.opt, "GRAD_ACCUM", 1)
    monkeypatch.setenv("GRAD_SPLIT_EVERY", "0")
    monkeypatch.setitem(tTc.MASK_TABLES, "mask", None)
    monkeypatch.setitem(tTc.EPISODE_WEIGHTS, "byIndex", None)
    tLt = SimpleNamespace(_TAGRIG_OUT=str(tmp_path / "nan"))
    tLt.update_policy = lambda tm, *a, **kw: (SimpleNamespace(loss=SimpleNamespace(val=float("nan")), grad_norm=1.0), None)
    import accelerate
    tOrigClip = accelerate.Accelerator.clip_grad_norm_
    try:
        mods.met.installMetrics(tLt, SimpleNamespace(F=torch.nn.functional), "mse_loss")
        tOpt = SimpleNamespace(param_groups=[{"lr": 0.1}])
        with pytest.raises(SystemExit, match="NaN/Inf at step 1"):
            tLt.update_policy(SimpleNamespace(), torch.nn.Linear(1, 1), {"index": torch.arange(2)}, tOpt, 1.0)
        assert (tmp_path / "nan" / "metrics.jsonl").exists()
    finally:
        accelerate.Accelerator.clip_grad_norm_ = tOrigClip
        mods.lm.BATCH.clear()


# ---------------------------------------------------------------- launchers
def test_train_dit_tagrig_patches(stubbed, mods, tmp_path, monkeypatch):
    monkeypatch.setenv("GRAD_CLIP", "0")
    monkeypatch.setenv("SCHED_HORIZON", "1000000")
    tLt = stubbed.lt
    tSeen = {}
    tLt.make_dataset = lambda cfg: tSeen.setdefault("ds", cfg)
    tLt.update_policy = lambda *a, **kw: None
    import accelerate
    tOrigClip = accelerate.Accelerator.clip_grad_norm_
    from lerobot.optim.schedulers import DiffuserSchedulerConfig
    import lerobot.policies.multi_task_dit.configuration_multi_task_dit as MC
    tOrigBuild, tOrigPreset = DiffuserSchedulerConfig.build, MC.MultiTaskDiTConfig.get_optimizer_preset
    try:
        tMod = stubbed.imp("laundry_training.train_dit_tagrig")
        assert tMod.installPatches() is tLt
        assert DiffuserSchedulerConfig().build(None, 500) == ("built", 1000000)
        assert MC.MultiTaskDiTConfig().get_optimizer_preset().grad_clip_norm == 0.0
        assert tLt.make_dataset.__name__ == "make_dataset_contract"
        with pytest.raises(SystemExit, match="contract chunk needs policy.horizon=33 and policy.n_obs_steps=2"):
            tLt.make_dataset(SimpleNamespace(policy=SimpleNamespace(horizon=32, n_obs_steps=2)))
    finally:
        accelerate.Accelerator.clip_grad_norm_ = tOrigClip
        DiffuserSchedulerConfig.build, MC.MultiTaskDiTConfig.get_optimizer_preset = tOrigBuild, tOrigPreset


def test_train_dit_mix(stubbed, golden, tmp_path, monkeypatch):
    tMix = stubbed.imp("laundry_training.train_dit_mix")
    assert tMix.installPatches() is stubbed.lt
    tRng = np.random.default_rng(41)
    golden.check("mix/state30", tMix.state30(tRng.normal(0, 1, (3, 20))))
    golden.check("mix/chunk33", tMix.chunk33(tRng.normal(0, 1, (32, 32)), [0.2, 0.9]))
    golden.check("mix/consts", {"REORDER": tMix.REORDER, "TLR_PAD": tMix.TLR_PAD, "ID9": tMix.ID9})
    tSpec = []
    for tI, tN in enumerate((30, 50)):
        tRoot = tmp_path / f"mix{tI}"
        tState = tRng.normal(0, 1, (tN, 20)).astype(np.float32)
        tAct = tRng.normal(0, 1, (tN, 32, 32)).astype(np.float32)
        tItems = [{"observation.state": torch.from_numpy(np.stack([tState[max(tRow - 1, 0)], tState[tRow]])),
                   "observation.state_is_pad": torch.tensor([tRow == 0, False]), "action": torch.from_numpy(tAct[tRow]),
                   "observation.images.global": torch.full((2, 3, 4, 4), float(tRow)),
                   "observation.images.global_is_pad": torch.zeros(2, dtype=torch.bool),
                   "task": "fold", "index": torch.tensor(tRow), "episode_index": torch.tensor(0),
                   "frame_index": torch.tensor(tRow), "timestamp": torch.tensor(tRow / 15), "task_index": torch.tensor(0),
                   "aux.x": torch.zeros(1)} for tRow in range(tN)]
        registerDataset(tRoot, {"observation.state": tState, "action": tAct, "episode_index": np.zeros(tN)}, items=tItems,
                        stats={"observation.images.global": {"mean": torch.zeros(3)}, "action": {"x": 1}},
                        info={"features": {"aux.x": {}, "state_prev_valid": {}, "observation.state": {}, "action": {},
                                           "keep": {}}},
                        cameras=["observation.images.global"], fps=15)
        tSpec.append({"name": f"m{tI}", "root": str(tRoot), "share": 1 + tI})
    monkeypatch.setenv("MIX_SPEC", json.dumps(tSpec))
    monkeypatch.setenv("MIX_STATS", str(tmp_path / "mix_stats.json"))
    tCfg = SimpleNamespace(policy=SimpleNamespace(horizon=33, n_obs_steps=2),
                           dataset=SimpleNamespace(image_transforms=SimpleNamespace(enable=False), video_backend="pyav"),
                           tolerance_s=1e-4)
    tDs = tMix.makeDatasetMix(tCfg)
    assert stubbed.lt.make_dataset is tMix.makeDatasetMix
    golden.check("mix/stats", tDs.meta.stats)
    assert sorted(tDs.meta.info["features"]) == ["action", "keep", "observation.state"]
    assert (len(tDs), tDs.num_frames, tDs.num_episodes) == (80, 80, 2)
    golden.check("mix/items", [tDs[0], tDs[29], tDs[30], tDs[79]])
    golden.check("mix/weights", tDs.sampleWeights())
    tSampler = tMix.weightedSampler()
    golden.check("mix/sampler", list(iter(tSampler))[:20])
    tDs2 = tMix.makeDatasetMix(tCfg)                    # stats from the cache
    golden.check("mix/stats", tDs2.meta.stats)
    with pytest.raises(SystemExit, match="contract chunk needs"):
        tMix.makeDatasetMix(SimpleNamespace(policy=SimpleNamespace(horizon=33, n_obs_steps=1)))
