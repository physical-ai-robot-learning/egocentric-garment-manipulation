"""Teleop log -> UMI action contract (2026-10-13). data_source = VISUAL_TELEOP.

Contract (ego_cart20, IMPORTED via ego_teleop/ego_cart20_link.py, never copied):
    action[t]  per horizon step k = 1..H:  [pose9(T_L(t)^-1 T_L(t+k)) | pose9(T_R(t)^-1 T_R(t+k)) | g_L(t+k) | g_R(t+k)]  (H, 20)
               = relpose.temporal_action_chunk + relpose.pose9 (xyz + rot6d = first two ROWS of R)  -- UMI-relative, NOT incremental
    obs[t]     pose9(relpose.inter_gripper_relative_pose(T_L, T_R)(t))  (9)   + g_L, g_R
    g          0 = closed, 1 = open (rebot_client.g_from_raw / gripper_cmd_from_g, mapping A)
Poses are the EXECUTED trajectory (label), converted FK -> reBot dataset-TCP axes with transforms/umi_robot_frame.py, so they
match UMI human streams (X_TOOL convention). Stage 0: FK of the accepted q; live (P1): FK of the OBSERVED joints. Human
targets are never a label.

NEVER exported as an action: the coordinator's per-tick incremental `arm_action6` ([dxyz, rotvec] of inv(T_prev) T_cmd,
rebot_command columns arm_dx..arm_drz) -- a different (incremental, rotvec) contract. Any attempt raises."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from ..ego_cart20_link import relpose
from ..transforms.umi_robot_frame import fk_to_dataset_tcp, TOOL_FRAME_NAME

DATA_SOURCES = ("UMI", "VISUAL_TELEOP", "ROBOT_AUTONOMOUS")
FORBIDDEN_ACTION_FIELDS = ("arm_action6", "arm_dx", "arm_dy", "arm_dz", "arm_drx", "arm_dry", "arm_drz", "action6")
ACTION_LAYOUT = "[pose9(dT_L k) 0:9 | pose9(dT_R k) 9:18 | g_L(t+k) 18 | g_R(t+k) 19], k = 1..H; pose9 = xyz + rot6d(rows)"


def _refuse_incremental(names) -> None:
    bad = [n for n in names if any(n == f or n.endswith("__" + f) for f in FORBIDDEN_ACTION_FIELDS)]
    if bad: raise ValueError(f"incremental arm_action6 fields {bad} must never be exported as a training action")


def export_umi_contract(t: dict, T_fk: dict, valid: dict, g: dict, *, horizon: int = 16, dt_s: float = 1 / 30,
                        max_gap_s: float = 0.1, data_source: str = "VISUAL_TELEOP") -> dict:
    """t/T_fk/valid/g: per side ('left', 'right') executed FK-frame TCP poses on that arm's own timeline.
    -> dict(t, action (n, H, 20), action_index (n,) row of t each chunk starts at, obs_T_LR9 (m, 9), g (m, 2), valid (m,), meta)."""
    if data_source not in DATA_SOURCES: raise ValueError(f"data_source must be one of {DATA_SOURCES}")
    rp = relpose(); H = int(horizon)
    t0 = max(float(np.asarray(t[s])[0]) for s in ("left", "right")); t1 = min(float(np.asarray(t[s])[-1]) for s in ("left", "right"))
    tc = np.arange(t0, t1 + 1e-9, dt_s)
    T, V, G = {}, {}, {}
    for s in ("left", "right"):
        ts = np.asarray(t[s], np.float64); Tds = fk_to_dataset_tcp(np.asarray(T_fk[s], np.float64))
        T[s], V[s] = rp.resample_poses(ts, Tds, np.asarray(valid[s], bool), tc, max_gap_s=max_gap_s)
        gs = np.asarray(g[s], np.float64); j = np.clip(np.searchsorted(ts, tc, side="right") - 1, 0, len(ts) - 1)   # zero-order hold
        G[s] = np.clip(gs[j], 0.0, 1.0)
    v = V["left"] & V["right"] & np.isfinite(G["left"]) & np.isfinite(G["right"])
    starts = [i for i in range(len(tc) - H) if v[i:i + H + 1].all()]
    act = np.zeros((len(starts), H, 20))
    for n, i in enumerate(starts):
        act[n, :, 0:9] = rp.pose9(rp.temporal_action_chunk(T["left"], i, H))
        act[n, :, 9:18] = rp.pose9(rp.temporal_action_chunk(T["right"], i, H))
        act[n, :, 18] = G["left"][i + 1:i + 1 + H]; act[n, :, 19] = G["right"][i + 1:i + 1 + H]
    obs = rp.pose9(rp.inter_gripper_relative_pose(T["left"], T["right"]))
    meta = dict(data_source=data_source, action_contract="ego_cart20.umi_fusion.relpose.temporal_action_chunk + pose9", action_layout=ACTION_LAYOUT,
                horizon=H, dt_s=dt_s, pose_frame=TOOL_FRAME_NAME, label="executed trajectory (FK of executed/observed q)",
                gripper="0 = closed, 1 = open", observation="pose9(inter_gripper_relative_pose(T_L, T_R)) + g_L, g_R",
                excluded="arm_action6 (incremental rotvec) is never exported")
    return dict(t=tc, action=act, action_index=np.asarray(starts, int), obs_T_LR9=obs, g=np.stack([G["left"], G["right"]], 1), valid=v,
                T_left_ds=T["left"], T_right_ds=T["right"], meta=meta)


def export_stage0_log(npz_in: str | Path, npz_out: str | Path | None = None, *, horizon: int = 16, dt_s: float = 1 / 30) -> dict:
    """tools/v0_replay.py stage0_log.npz -> UMI contract npz. Uses {side}__T_exec_fk (FK of executed q), {side}__accepted-
    derived validity and {side}__g ONLY."""
    d = np.load(npz_in, allow_pickle=False); _refuse_incremental(d.files)
    t, T, V, G = {}, {}, {}, {}
    for s in ("left", "right"):
        t[s] = d[f"{s}__t"]; T[s] = d[f"{s}__T_exec_fk"]
        V[s] = d[f"{s}__reason"] != "not_anchored"           # held frames are real (the robot holds still); before anchor is not
        G[s] = d[f"{s}__g"]
        if not np.isfinite(G[s]).any(): G[s] = np.full(len(t[s]), 1.0)          # no gripper source: open, flagged in meta
    no_grip = [s for s in ("left", "right") if not np.isfinite(d[f"{s}__g"]).any()]
    out = export_umi_contract(t, T, V, G, horizon=horizon, dt_s=dt_s)
    if no_grip: out["meta"]["gripper_missing_filled_open"] = no_grip
    if npz_out is not None:
        import json
        np.savez_compressed(npz_out, **{k: v for k, v in out.items() if k != "meta"}, meta_json=json.dumps(out["meta"]))
    return dict(n_chunks=int(len(out["action_index"])), timeline_frames=int(len(out["t"])), valid_frac=round(float(out["valid"].mean()), 4),
                horizon=horizon, out=str(npz_out) if npz_out else None, **{k: out["meta"][k] for k in ("data_source", "action_layout")},
                **({"gripper_missing_filled_open": no_grip} if no_grip else {}))


def executed_poses_from_command_rows(df):
    """rebot_command parquet rows -> executed FK-frame poses from x..qw ONLY (P1 live logs). Refuses to fall back to arm_*.
    NOTE: the live label must come from FK of OBSERVED joints (rebot_state), this is the commanded pose."""
    need = ("x", "y", "z", "qx", "qy", "qz", "qw")
    if not all(c in df.columns for c in need): raise ValueError(f"executed pose columns {need} missing; arm_action6 is not a substitute")
    from ..transforms.se3 import poses7_to_T
    return poses7_to_T(df[list(need)].to_numpy(np.float64))
