"""Run outputs: log.npz + summary.json, the gripper plot and the Desktop video (Orbbec projection + top-down + wrists).
"""
from __future__ import annotations
import json
import subprocess

import cv2
import numpy as np

from garment_pipeline.eval_deploy import act_tagrig_core as core
from garment_pipeline.eval_deploy.tagrig_deploy.paths import DESKTOP


# ---------------------------------------------------------------- visualisation
COL = {"left": (255, 140, 0), "right": (0, 140, 255)}      # BGR: left = blue, right = orange (tagrig_qa convention)


def topdown(Tm, Tc, WS, train_pts, size=480):
    """base frame from above: x forward = up, +y (robot left) = left"""
    img = np.full((size, size, 3), 255, np.uint8); x0, x1, y0, y1 = -0.1, 0.8, -0.65, 0.65
    px = lambda p: (int((y1 - p[1]) / (y1 - y0) * size), int((x1 - p[0]) / (x1 - x0) * size))
    for p in train_pts: cv2.circle(img, px(p), 1, (215, 215, 215), -1)
    for s in core.SIDES:
        a_, b_ = WS[s]; cv2.rectangle(img, px((b_[0], b_[1])), px((a_[0], a_[1])), COL[s], 1)
        pts = [px(T[:3, 3]) for T in Tc[s]]
        for i in range(len(pts) - 1): cv2.line(img, pts[i], pts[i + 1], COL[s], 2)
        cv2.circle(img, px(Tm[s][:3, 3]), 7, (0, 0, 0), 2); cv2.circle(img, pts[0], 4, COL[s], -1); cv2.circle(img, pts[-1], 6, COL[s], 2)
    cv2.circle(img, px((0, 0.28)), 5, (0, 0, 0), -1); cv2.circle(img, px((0, -0.28)), 5, (0, 0, 0), -1)
    cv2.putText(img, "top view: up = +x fwd, left = +y (robot L)", (8, 18), 0, 0.45, (0, 0, 0), 1); cv2.putText(img, "L", px((0.02, 0.33)), 0, 0.6, COL["left"], 2); cv2.putText(img, "R", px((0.02, -0.25)), 0, 0.6, COL["right"], 2)
    return img


def summarize(L, a, stop_reason, n_viol, run_dir, WS, off, cfg, prev, ckpt, runner):
    n = len(L["t"])
    if n == 0: print("기록 없음"); return {}
    big = ("flags", "gimg_jpg", "wl", "wr")
    arr = {k: np.asarray(v) for k, v in L.items() if k not in big}
    np.savez_compressed(run_dir / "log.npz", **arr, flags=np.array([";".join(f) for f in L["flags"]]), wl=np.stack(L["wl"]), wr=np.stack(L["wr"]),
                        gimg_jpg=np.array(L["gimg_jpg"], dtype=object), preview=np.array([json.dumps(r) for r in prev.log]))
    Tm, Tc = arr["T_meas"], arr["T_chunk"]
    d1 = np.linalg.norm(Tc[:, :, 0, :3, 3] - Tm[:, :, :3, 3], axis=2) * 1e3; d32 = (Tc[:, :, 31, :3, 3] - Tm[:, :, :3, 3]) * 1e3
    cnt = {}
    for fs in L["flags"]:
        for f_ in fs:
            if ":" not in f_: cnt[f_] = cnt.get(f_, 0) + 1; continue      # 10-30: a colon-less flag crashed the summary
            h, b = f_.split(":", 1); k = h + ":" + "_".join(b.split("_")[:2]); cnt[k] = cnt.get(k, 0) + 1
    q = lambda x: dict(p50=round(float(np.nanmedian(x)), 1), p90=round(float(np.nanpercentile(x, 90)), 1), max=round(float(np.nanmax(x)), 1)) if np.isfinite(np.asarray(x, float)).any() else None
    G = arr["grip_raw"]                       # (n, 32, 2) continuous gripper outputs before the threshold
    thirds = np.array_split(np.arange(n), 3)
    grip = {s: dict(all_steps=dict(zip(("p01", "p10", "p50", "p90", "p99"), np.round(np.percentile(G[:, :, j], [1, 10, 50, 90, 99]), 3).tolist())),
                    k1_by_third=[round(float(np.mean(G[ix, 0, j])), 3) for ix in thirds], k32_by_third=[round(float(np.mean(G[ix, 31, j])), 3) for ix in thirds],
                    open_frac_k1=round(float(np.mean(G[:, 0, j] > 0.5)), 3), open_frac_any_step=round(float(np.mean((G[:, :, j] > 0.5).any(1))), 3),
                    out_of_0_1=round(float(np.mean((G[:, :, j] < -0.05) | (G[:, :, j] > 1.05))), 3)) for j, s in enumerate(core.SIDES)}
    pv = prev.log
    S = dict(run=str(run_dir), policy=a.policy, ckpt=ckpt, dit_sampler=("DDIM 10 x%d" % a.dit_samples if a.policy == "dit" else None), go=a.go, commit=a.commit,
             cycles=n, secs=round(float(L["t"][-1]), 1), plan_hz=round(n / max(L["t"][-1], 1e-6), 1), stop_reason=stop_reason, violation_cycles=n_viol,
             assume_ready=a.assume_ready, global_src=("REPLAY " + a.replay_global) if a.replay_global else a.global_url, ready_offset_deg=off,
             cams={s: cfg[s] for s in core.SIDES}, wave_test=cfg.get("wave_test"),
             latency_ms=dict(loop=q(arr["lat_total_ms"]), obs=q(arr["lat_obs_ms"]), policy=q(arr["lat_pol_ms"]), ctrl_ik_safety=q(arr["lat_ctrl_ms"]),
                             obs_to_commit=q(arr["obs_to_commit_ms"]), k0_steps_skipped=q(arr["k0"]), preview=q([r["ms"] for r in pv]) if pv else None),
             dit_history_used_frac=round(float(np.mean(arr["hist_used"])), 3) if a.policy == "dit" else None,
             k1_disp_mm=dict(left=q(d1[:, 0]), right=q(d1[:, 1])),
             k32_disp_mm_median=dict(left=np.round(np.median(d32[:, 0], 0)).tolist(), right=np.round(np.median(d32[:, 1], 0)).tolist()),
             k32_dz_mm=dict(left=q(d32[:, 0, 2]), right=q(d32[:, 1, 2])), k32_norm_mm=dict(left=q(np.linalg.norm(d32[:, 0], axis=1)), right=q(np.linalg.norm(d32[:, 1], axis=1))),
             chunk_y_median_m=dict(left=round(float(np.median(Tc[:, 0, :, 1, 3])), 3), right=round(float(np.median(Tc[:, 1, :, 1, 3])), 3)),
             left_on_plus_y=bool(np.median(Tc[:, 0, :, 1, 3]) > 0 and np.median(Tc[:, 1, :, 1, 3]) < 0),
             grip_raw=grip, chunk_smoothness_mm={k: q(arr[f"dchunk_{k}_mm"]) for k in ("k1", "k8", "k16", "k32")},
             preview=dict(n=len(pv), ik_ok=np.round(np.nanmean([r["ik_ok"] for r in pv], 0), 3).tolist() if pv else None,
                          ik_max_mm=q([max(r["ik_max_mm"]) for r in pv]) if pv else None, min_clear_mm=q([r["min_clear"] * 1e3 for r in pv]) if pv else None,
                          min_table_mm=q([r["min_table"] * 1e3 for r in pv]) if pv else None),
             ik_commit_mm=q(arr["ik_mm"].ravel()), ws_chunk_out_frac=np.round(arr["ws_chunk_out"].mean(0), 3).tolist(),
             arm_clear_mm=q(arr["arm_clear"] * 1e3), link_gap_mm=q(arr["link_gap"] * 1e3), flag_counts=cnt, img_age_ms=q(arr["img_age_ms"].ravel()),
             workspace={s: [np.round(WS[s][0], 3).tolist(), np.round(WS[s][1], 3).tolist()] for s in core.SIDES})
    (run_dir / "summary.json").write_text(json.dumps(S, indent=1, ensure_ascii=False)); print(json.dumps(S, indent=1, ensure_ascii=False))
    if not a.no_video:
        S["video"] = str(render_video(L, arr, run_dir, WS)); S["grip_plot"] = str(grip_plot(arr, run_dir))
        (run_dir / "summary.json").write_text(json.dumps(S, indent=1, ensure_ascii=False))
    return S


def grip_plot(arr, run_dir):
    """raw gripper outputs over time (pre-threshold, 1 = open): k = 1 / 8 / 32 per arm -> Desktop png"""
    t = arr["t"]; G = arr["grip_raw"]; C = {1: (0, 0, 0), 8: (200, 120, 0), 32: (0, 120, 230)}
    im = [core.cv_plot([(t, G[:, k - 1, j], C[k], f"k={k}") for k in (1, 8, 32)], (-0.2, 1.2), f"{run_dir.name} {s} gripper raw (threshold 0.5)", hline=0.5, xlabel="s")
          for j, s in enumerate(core.SIDES)]
    p = DESKTOP / f"act_{run_dir.name}_grip.png"; cv2.imwrite(str(p), np.vstack(im)); return p


def render_video(L, arr, run_dir, WS):
    T_C_B, K = core.robot_cam()
    tp = core.LDT / "deploy/train_tcp_sample.npy"; train = np.load(tp) if tp.exists() else np.zeros((0, 3))
    path = DESKTOP / f"act_{run_dir.name}.mp4"; tmp = run_dir / "video_raw.mp4"
    W, H = 848 + 480, 480 + 216; vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 10.0 if run_dir.name.startswith("dit") else 20.0, (W, H))
    for i in range(len(L["t"])):
        g = cv2.resize(cv2.imdecode(np.frombuffer(L["gimg_jpg"][i], np.uint8), cv2.IMREAD_COLOR), (848, 480))
        Tm = {s: arr["T_meas"][i][j] for j, s in enumerate(core.SIDES)}; Tc = {s: arr["T_chunk"][i][j] for j, s in enumerate(core.SIDES)}
        for s in core.SIDES:
            uv = core.project(T_C_B, K, np.array([T[:3, 3] for T in Tc[s]]))
            for k in range(31):
                if np.isfinite(uv[k]).all() and np.isfinite(uv[k + 1]).all(): cv2.line(g, tuple(uv[k].astype(int)), tuple(uv[k + 1].astype(int)), COL[s], 2)
            u0 = core.project(T_C_B, K, Tm[s][None, :3, 3])[0]
            if np.isfinite(u0).all(): cv2.circle(g, tuple(u0.astype(int)), 8, (0, 0, 0), 2)
            if np.isfinite(uv[-1]).all(): cv2.circle(g, tuple(uv[-1].astype(int)), 6, COL[s], -1)
        txt = f"t {L['t'][i]:5.1f}s loop {L['lat_total_ms'][i]:4.0f}ms pol {L['lat_pol_ms'][i]:4.0f}ms  gL {arr['g_pred'][i][0]:.2f} gR {arr['g_pred'][i][1]:.2f}  spd {L['speed'][i]:.2f}"
        cv2.rectangle(g, (0, 0), (848, 30), (0, 0, 0), -1); cv2.putText(g, txt, (8, 21), 0, 0.55, (0, 255, 0), 1)
        st = [x for x in L["flags"][i] if not x.startswith("LIM")]
        if st: cv2.putText(g, " ".join(st)[:90], (8, 470), 0, 0.5, (0, 0, 255), 2)
        top = np.hstack([g, topdown(Tm, Tc, WS, train)])
        wl = cv2.cvtColor(L["wl"][i], cv2.COLOR_RGB2BGR); wr_ = cv2.cvtColor(L["wr"][i], cv2.COLOR_RGB2BGR)
        cv2.putText(wl, "L wrist (policy input)", (5, 15), 0, 0.45, (0, 0, 255), 1); cv2.putText(wr_, "R wrist (policy input)", (5, 15), 0, 0.45, (0, 0, 255), 1)
        bot = np.hstack([wl, wr_, np.full((216, W - 768, 3), 40, np.uint8)])
        k1 = [np.linalg.norm(Tc[s][0][:3, 3] - Tm[s][:3, 3]) * 1e3 for s in core.SIDES]; k32 = [(Tc[s][31][:3, 3] - Tm[s][:3, 3]) * 1e3 for s in core.SIDES]
        for r_, t_ in enumerate([f"k1 step  L {k1[0]:.1f} mm  R {k1[1]:.1f} mm", f"k32 L dxyz {np.round(k32[0]).astype(int)} mm", f"k32 R dxyz {np.round(k32[1]).astype(int)} mm",
                                 f"preview IK ok L/R {arr['pv_ik_ok'][i][0]:.2f}/{arr['pv_ik_ok'][i][1]:.2f}", f"arm clear {arr['arm_clear'][i] * 1e3:.0f} mm  gap {arr['link_gap'][i] * 1e3:.0f} mm"]):
            cv2.putText(bot, t_, (778, 30 + 36 * r_), 0, 0.6, (255, 255, 255), 1)
        vw.write(np.vstack([top, bot]))
    vw.release()
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(path)], capture_output=True)
    if r.returncode: import shutil; shutil.copy(tmp, path)
    print("영상", path, flush=True); return path
