"""[2026-10-22] Offline evaluation of an ACT or Multi-Task DiT checkpoint on the tag-rig val/test roots (no robot; model selection).
  deviation   predicted contract chunk (denormalised by the checkpoint's own post-processor) vs the robotized-trajectory chunk,
              translation mm / rotation deg at k = 1, 8, 16, 32 per arm (= base-frame deviation of T(t) @ dT from the robotized
              target, the anchor is shared); baselines: zero motion and the train mean chunk
  L/R swap    cosine of the predicted k=32 base-frame displacement of each arm with the GT of the SAME arm vs the OTHER arm
  gripper     per episode, the k=1 gripper prediction at every --stride frame, thresholded 0.5 (deploy rule) vs the binary target;
              open/close events matched within +-1 s -> timing error ms, missed / spurious events
Input: --ckpt step folder, --data LeRobot root. Output: <out>/offline_eval.json (also printed).
usage: python -m laundry_deploy.tools.offline_eval --ckpt checkpoints/RUN/checkpoints/last --data data/lerobot/tagrig_laundry_v1_val [--stride 2] --out reports/RUN_val
Was laundry_deploy/tagrig_offline_eval.py.
"""

from __future__ import annotations
import argparse
import csv
import json
import pathlib
import sys

import numpy as np

from laundry_shared.contracts.action_contract import pose9_to_T

from ..core.checkpoint import loadPolicy

KS = (1, 8, 16, 32)  # chunk rows reported
GH_ROWS = (8, 16, 32)  # 10-29 jaw at chunk rows k: (pred open, gt open, gt open at k1) per anchor and arm
BREAKDOWN_ROWS = (16, 32)
PER_EPISODE_ROWS = (8, 16, 32)
FPS = 20
CHUNK = 32
ACTION_DIM = 20
GRIP_OPEN = 0.5
LEFT_SLICE, RIGHT_SLICE = slice(0, 9), slice(9, 18)
ARMS = (("L", LEFT_SLICE), ("R", RIGHT_SLICE))
GRIP_COLS = slice(18, 20)
GRIP_COL0 = 18
MIN_GT_DISP_M = 0.02  # L/R swap check only on anchors where both arms really move
COS_EPS = 1e-9
EVENT_TOL_S = 1.0
VAL_BATCH = 16
MM_PER_M = 1e3
MS_PER_S = 1e3
ZERO_POSE9 = np.r_[0, 0, 0, 1, 0, 0, 0, 1, 0]
CONFIDENT_RULE = (
    "anchor t and target t+32 both tag-backed or fgo gap <= 0.5 s (per arm); grasp = anchor within +-0.5 s of a jaw transition; "
    "cut-offs: < 60 mm meaningful, 40-50 promising, 30-40 rollout-worthy"
)
VAL_SCORE_DEF = (
    "mean (not median: val is ~40 % near-static windows) L/R translation deviation (mm) at k=16 and k=32; early-stop metric"
)
BREAKDOWN_DEF = (
    "mean over every evaluated anchor and arm (no confidence filter); grasp = anchor within +-0.5 s of a jaw transition (per arm); "
    "phase = thirds of the episode; zero_mm = stay-still baseline"
)
GRIPPER_RULE = "pred k=1 gripper > 0.5 = open; events matched within +-1 s"
GRIPPER_HORIZON_DEF = (
    "10-29: k1 agreement is reachable by COPYING the input jaw state (Exp 1/2 did: 98 % here, never opened on the robot). "
    "acc_k = jaw at chunk row k vs target; switch_k = only anchors whose target jaw at k differs from row 1 "
    "(a copy-the-state policy scores ~0 there); copy_baseline_* = predicting the row-1 target for every row"
)


def predict(cfg, policy, pre, post, item: dict, seed: int) -> np.ndarray:
    """(32, 20) denormalised contract rows of one item (ACT chunk / DiT rows 1..32 of the 33, BCE jaw head when present)."""
    import torch

    with torch.no_grad():
        tBatch = {
            tKey: (tValue.unsqueeze(0) if torch.is_tensor(tValue) else [tValue])
            for tKey, tValue in item.items()
            if tKey.startswith("observation") or tKey == "task"
        }
        tBatch = pre(tBatch)
        torch.manual_seed(seed)
        if cfg.type == "act":
            tActs = policy.predict_action_chunk(tBatch)  # (1, 32, 20) normalised
            return post(tActs)[0].float().cpu().numpy()
        tPrepared = policy._prepare_batch(dict(tBatch))
        tCond = policy.observation_encoder.encode(tPrepared)
        tActs = policy.objective.conditional_sample(policy.noise_predictor, 1, tCond)  # (1, 33, 20)
        if hasattr(policy, "grip_head"):  # BCE jaw head
            tActs[..., GRIP_COLS] = 2.0 * torch.sigmoid(policy.grip_logits(tCond)).to(tActs.dtype) - 1.0
        return post(tActs)[0].float().cpu().numpy()[1:]


def poseError(pred: np.ndarray, gt: np.ndarray, sl: slice):
    """Per row: translation error mm, rotation error deg of one arm's pose9 columns. Was err()."""
    tTp, tTg = pose9_to_T(pred[:, sl].astype(np.float64)), pose9_to_T(gt[:, sl].astype(np.float64))
    tCos = (np.trace(np.swapaxes(tTp[:, :3, :3], 1, 2) @ tTg[:, :3, :3], axis1=1, axis2=2) - 1) / 2
    return np.linalg.norm(tTp[:, :3, 3] - tTg[:, :3, 3], axis=1) * MM_PER_M, np.degrees(np.arccos(np.clip(tCos, -1, 1)))


def valLoss(cfg, policy, pre, dataset, batches: int, batchSize: int = VAL_BATCH):
    """Mean training loss of the policy on `batches` fixed val batches (same targets / normalisation as training; DiT noise fixed by
    the seed)."""
    import torch
    from torch.utils.data import DataLoader

    with torch.no_grad():
        tGen = torch.Generator().manual_seed(0)
        tLoader = DataLoader(dataset, batch_size=batchSize, shuffle=True, generator=tGen, num_workers=0, drop_last=True)
        torch.manual_seed(0)
        tOut = []
        for tI, tB in enumerate(tLoader):
            if tI >= batches:
                break
            tB = pre(tB)
            if (
                cfg.type == "act"
            ):  # eval-mode ACT has no VAE posterior (forward() would fail on the KL): L1 of the prior-latent chunk
                tMask = (~tB["action_is_pad"]).unsqueeze(-1).float()
                tOut.append(
                    float(((policy.predict_action_chunk(tB) - tB["action"]).abs() * tMask).sum() / (tMask.sum() * ACTION_DIM))
                )
            else:
                tLoss, _ = policy.forward(tB)
                tOut.append(float(tLoss))
    return round(float(np.mean(tOut)), 5) if tOut else None


def gripEvents(binary: np.ndarray) -> list:
    """[(frame, "open" | "close")] of a binary jaw series. Was events()."""
    tD = np.diff(binary.astype(int))
    return [(tI + 1, "open" if tD[tI] > 0 else "close") for tI in np.flatnonzero(tD)]


def matchEvents(predEvents: list, gtEvents: list, fps: float, tolS: float = EVENT_TOL_S):
    """-> (timing errors ms of matched events, missed GT events, spurious predicted events)."""
    tUsed, tErrs, tMissed = set(), [], 0
    for tGi, tGk in gtEvents:
        tCands = [
            (abs(tPi - tGi), tJ)
            for tJ, (tPi, tPk) in enumerate(predEvents)
            if tPk == tGk and tJ not in tUsed and abs(tPi - tGi) <= tolS * fps
        ]
        if tCands:
            _, tJ = min(tCands)
            tUsed.add(tJ)
            tErrs.append((predEvents[tJ][0] - tGi) / fps * MS_PER_S)
        else:
            tMissed += 1
    return tErrs, tMissed, len(predEvents) - len(tUsed)


def gripHorizonReport(gh: dict) -> dict:
    """Was gh_report()."""
    tOut = {}
    for tK, tV in gh.items():
        if not tV:
            continue
        tA = np.array(tV, bool)
        tPred, tGt, tGt1 = tA[:, 0], tA[:, 1], tA[:, 2]
        tSwitch = tGt != tGt1
        tOut[f"k{tK}"] = dict(
            acc=round(float((tPred == tGt).mean()), 3),
            copy_baseline_acc=round(float((tGt1 == tGt).mean()), 3),
            switch_n=int(tSwitch.sum()),
            switch_acc=round(float((tPred[tSwitch] == tGt[tSwitch]).mean()), 3) if tSwitch.any() else None,
            pred_open_frac=round(float(tPred.mean()), 3),
            gt_open_frac=round(float(tGt.mean()), 3),
        )
    return tOut


def _cosine(u, v) -> float:
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + COS_EPS))


def _summary(x):
    return (
        dict(mean=round(float(np.mean(x)), 1), median=round(float(np.median(x)), 1), p90=round(float(np.percentile(x, 90)), 1))
        if len(x)
        else None
    )


def _meanOrNone(x):
    return round(float(np.mean(x)), 1) if len(x) else None


def _bdAdd(breakdown: dict, key: str, mm, deg, zeroMm) -> None:
    tD = breakdown.setdefault(key, {f"{tM}_k{tK}": [] for tM in ("mm", "deg", "zero_mm") for tK in BREAKDOWN_ROWS})
    for tK in BREAKDOWN_ROWS:
        tD[f"mm_k{tK}"].append(mm[tK - 1])
        tD[f"deg_k{tK}"].append(deg[tK - 1])
        tD[f"zero_mm_k{tK}"].append(zeroMm[tK - 1])


def _defaultDevice() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser()
    tParser.add_argument("--ckpt", required=True)
    tParser.add_argument("--data", required=True)
    tParser.add_argument("--stride", type=int, default=2)
    tParser.add_argument("--out", required=True)
    tParser.add_argument("--device", default=_defaultDevice())
    tParser.add_argument(
        "--val-loss-batches", type=int, default=20, help="policy training loss on N fixed val batches (eval mode, seed 0)"
    )
    tParser.add_argument("--max-frames", type=int, default=0)
    tParser.add_argument("--video-backend", default=None)
    return tParser


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    from laundry_training import ActRelDataset, DitRelDataset

    tOutDir = pathlib.Path(tArgs.out)
    tOutDir.mkdir(parents=True, exist_ok=True)
    tCfg, tPolicy, tPre, tPost = loadPolicy(tArgs.ckpt, tArgs.device)
    tRoot = pathlib.Path(tArgs.data)
    tKw = dict(video_backend=tArgs.video_backend) if tArgs.video_backend else {}
    if tCfg.type == "act":
        tDts = {"action": [tI / FPS for tI in range(CHUNK)]}
        tDs = ActRelDataset("local/" + tRoot.name, root=tRoot, delta_timestamps=tDts, **tKw)
    else:
        tDts = {
            tKey: [tI / FPS for tI in range(1 - tCfg.n_obs_steps, 1)]
            for tKey in list(tCfg.image_features) + ["observation.state"]
        }
        tDts["action"] = [tI / FPS for tI in range(1 - tCfg.n_obs_steps, 1 - tCfg.n_obs_steps + tCfg.horizon)]
        tDs = DitRelDataset("local/" + tRoot.name, root=tRoot, delta_timestamps=tDts, **tKw)
    tValLoss = valLoss(tCfg, tPolicy, tPre, tDs, tArgs.val_loss_batches) if tArgs.val_loss_batches else None
    tHf = tDs.hf_dataset.with_format("numpy")
    tEpi = np.asarray(tHf["episode_index"])
    tPoses = np.asarray(tHf["pose_abs"])
    tFeats = set(tDs.features)
    tPosConf = np.asarray(tHf["pos_conf"]) if "pos_conf" in tFeats else None  # 10-23: confident = tag or fgo gap <= 0.5 s
    tGraspWin = np.asarray(tHf["grasp_win"]) if "grasp_win" in tFeats else None  # +-0.5 s around jaw transitions
    tConf = {f"{tS}_{tSub}_{tM}": [] for tS in "LR" for tSub in ("all", "grasp", "free") for tM in ("mm", "zero")}
    tCkptDir = pathlib.Path(tArgs.ckpt) / "pretrained_model"
    _ = json.loads((tCkptDir / "train_config.json").read_text()) if tCkptDir.is_dir() else {}
    tE = {f"{tS}_{tM}_k{tK}": [] for tS in "LR" for tM in ("mm", "deg", "zero_mm") for tK in KS}
    tCosSame, tCosOther = [], []
    tEvErr, tEvMiss, tEvSpur, tEvN = [], 0, 0, 0
    tGAgree = []
    tN = 0
    tGh = {tK: [] for tK in GH_ROWS}
    # 10-26: unconditional breakdown (every evaluated anchor): grasp (anchor within +-0.5 s of a jaw transition, per arm) vs free,
    # and
    # episode phase (anchor position in the episode, thirds) -> k16 / k32 translation mm, rotation deg, zero-motion mm
    tBreakdown = {}
    tPerEpisode = []  # 10-30: per-episode means (small aux sets: a mean over 9 episodes hides one bad demo)
    try:
        tKeys = [tRow["episode_key"] for tRow in csv.DictReader(open(tRoot / "episode_table.csv"))]
    except Exception:
        tKeys = None
    for tEp in np.unique(tEpi):
        tN0 = {tK: len(tV) for tK, tV in tE.items()}
        tC0 = {tK: len(tV) for tK, tV in tConf.items()}
        tLast = int(np.flatnonzero(tEpi == tEp).max())
        tFirst = int(np.flatnonzero(tEpi == tEp).min())
        tSel = np.flatnonzero(tEpi == tEp)[:: tArgs.stride]
        if tArgs.max_frames:
            tSel = tSel[: tArgs.max_frames]
        tGp, tGg = [], []
        for tJ, tI in enumerate(tSel):
            tItem = tDs[int(tI)]
            tGt = tItem["action"].numpy()
            tGt = tGt[1:] if tGt.shape[0] == CHUNK + 1 else tGt
            tPr = predict(tCfg, tPolicy, tPre, tPost, tItem, tJ)
            tN += 1
            tZero = np.tile(ZERO_POSE9, 2)
            tZ = np.tile(np.r_[tZero, tGt[0, GRIP_COLS]], (CHUNK, 1))
            tRt = {tS: pose9_to_T(tPoses[int(tI), tSl].astype(np.float64))[:3, :3] for tS, tSl in ARMS}
            tDisp = {}
            for tS, tSl in ARMS:
                tMm, tDg = poseError(tPr, tGt, tSl)
                tZm, _ = poseError(tZ, tGt, tSl)
                for tK in KS:
                    tE[f"{tS}_mm_k{tK}"].append(tMm[tK - 1])
                    tE[f"{tS}_deg_k{tK}"].append(tDg[tK - 1])
                    tE[f"{tS}_zero_mm_k{tK}"].append(tZm[tK - 1])
                tArmIdx = 0 if tS == "L" else 1
                tGsub = "grasp" if (tGraspWin is not None and tGraspWin[int(tI), tArmIdx] > 0) else "free"
                tPhase = ("early", "mid", "late")[min(2, int(3 * (int(tI) - tFirst) / max(1, tLast - tFirst + 1)))]
                for tKey in ("all", tGsub, f"phase_{tPhase}", f"{tGsub}_{tS}"):
                    _bdAdd(tBreakdown, tKey, tMm, tDg, tZm)
                tDisp[tS] = (tRt[tS] @ tPr[CHUNK - 1, tSl][:3], tRt[tS] @ tGt[CHUNK - 1, tSl][:3])
                if tPosConf is not None and int(tI) + CHUNK <= tLast:  # confident anchor AND confident 32-step target only
                    if tPosConf[int(tI), tArmIdx] > 0 and tPosConf[int(tI) + CHUNK, tArmIdx] > 0:
                        tSub = "grasp" if (tGraspWin is not None and tGraspWin[int(tI), tArmIdx] > 0) else "free"
                        for tSb in ("all", tSub):
                            tConf[f"{tS}_{tSb}_mm"].append(tMm[CHUNK - 1])
                            tConf[f"{tS}_{tSb}_zero"].append(tZm[CHUNK - 1])
            for tS, tO in (("L", "R"), ("R", "L")):
                if np.linalg.norm(tDisp[tS][1]) > MIN_GT_DISP_M and np.linalg.norm(tDisp[tO][1]) > MIN_GT_DISP_M:
                    tCosSame.append(_cosine(tDisp[tS][0], tDisp[tS][1]))
                    tCosOther.append(_cosine(tDisp[tS][0], tDisp[tO][1]))
            tGp.append(tPr[0, GRIP_COLS] > GRIP_OPEN)
            tGg.append(tGt[0, GRIP_COLS] > GRIP_OPEN)
            for tK in tGh:
                tGh[tK] += [
                    (
                        tPr[tK - 1, GRIP_COL0 + tC] > GRIP_OPEN,
                        tGt[tK - 1, GRIP_COL0 + tC] > GRIP_OPEN,
                        tGt[0, GRIP_COL0 + tC] > GRIP_OPEN,
                    )
                    for tC in (0, 1)
                ]
        tPe = dict(episode_index=int(tEp), key=tKeys[int(tEp)] if tKeys and int(tEp) < len(tKeys) else None, n=len(tSel))
        for tK, tV in tE.items():
            if any(f"_k{tX}" == tK[-len(f"_k{tX}") :] for tX in PER_EPISODE_ROWS):
                tPe[tK] = round(float(np.mean(tV[tN0[tK] :])), 1) if len(tV) > tN0[tK] else None
        tCm = tConf["L_all_mm"][tC0["L_all_mm"] :] + tConf["R_all_mm"][tC0["R_all_mm"] :]
        tCz = tConf["L_all_zero"][tC0["L_all_zero"] :] + tConf["R_all_zero"][tC0["R_all_zero"] :]
        tPe["conf_k32_mm"] = round(float(np.mean(tCm)), 1) if tCm else None
        tPe["conf_k32_zero_mm"] = round(float(np.mean(tCz)), 1) if tCz else None
        tPe["conf_n"] = len(tCm)
        tPerEpisode.append(tPe)
        tGp, tGg = np.array(tGp), np.array(tGg)
        tGAgree.append((tGp == tGg).mean(0))
        for tC in (0, 1):
            tErrs, tMiss, tSpur = matchEvents(gripEvents(tGp[:, tC]), gripEvents(tGg[:, tC]), FPS / tArgs.stride)
            tEvErr += tErrs
            tEvMiss += tMiss
            tEvSpur += tSpur
            tEvN += len(gripEvents(tGg[:, tC]))
    tDev = [np.mean(tE[f"{tS}_mm_k{tK}"]) for tS in "LR" for tK in BREAKDOWN_ROWS]
    tConfReport = None
    if tPosConf is not None:

        def _meanMedianN(x):
            return dict(mean=round(float(np.mean(x)), 1), median=round(float(np.median(x)), 1), n=len(x)) if len(x) else None

        tConfReport = {tK: _meanMedianN(tV) for tK, tV in tConf.items()}
        tMmAll = tConf["L_all_mm"] + tConf["R_all_mm"]
        tZAll = tConf["L_all_zero"] + tConf["R_all_zero"]
        tConfReport["k32_trans_mm_conf"] = round(float(np.mean(tMmAll)), 1) if tMmAll else None
        tConfReport["stay_still_mm_conf"] = round(float(np.mean(tZAll)), 1) if tZAll else None
        for tSub in ("grasp", "free"):
            tV = tConf[f"L_{tSub}_mm"] + tConf[f"R_{tSub}_mm"]
            tZz = tConf[f"L_{tSub}_zero"] + tConf[f"R_{tSub}_zero"]
            tConfReport[f"{tSub}_k32_mm"] = round(float(np.mean(tV)), 1) if tV else None
            tConfReport[f"{tSub}_stay_still_mm"] = round(float(np.mean(tZz)), 1) if tZz else None
    tBreakdownReport = {
        tKey: {**{tM: _meanOrNone(tV) for tM, tV in tD.items()}, "n": len(tD["mm_k32"])}
        for tKey, tD in sorted(tBreakdown.items())
    }
    tReport = dict(
        ckpt=tArgs.ckpt,
        policy=tCfg.type,
        data=str(tRoot),
        samples=tN,
        stride=tArgs.stride,
        val_loss=tValLoss,
        confident_frames_metrics=tConfReport,
        confident_rule=CONFIDENT_RULE,
        val_score_mm=round(float(np.mean(tDev)), 2),
        val_score_def=VAL_SCORE_DEF,
        deviation={tK: _summary(tV) for tK, tV in tE.items()},
        per_episode=tPerEpisode,
        breakdown=tBreakdownReport,
        breakdown_def=BREAKDOWN_DEF,
        lr_swap_check=dict(
            cos_pred_vs_same_arm_gt=_summary(tCosSame), cos_pred_vs_other_arm_gt=_summary(tCosOther), n=len(tCosSame)
        ),
        gripper=dict(
            state_agreement_LR=np.mean(tGAgree, 0).round(3).tolist() if tGAgree else None,
            gt_events=tEvN,
            matched=len(tEvErr),
            timing_err_ms=dict(
                median_abs=round(float(np.median(np.abs(tEvErr))), 0), mean_signed=round(float(np.mean(tEvErr)), 0)
            )
            if tEvErr
            else None,
            missed=tEvMiss,
            spurious=tEvSpur,
            rule=GRIPPER_RULE,
        ),
        gripper_horizon=gripHorizonReport(tGh),
        gripper_horizon_def=GRIPPER_HORIZON_DEF,
    )
    (tOutDir / "offline_eval.json").write_text(json.dumps(tReport, indent=1))
    print(json.dumps(tReport, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
