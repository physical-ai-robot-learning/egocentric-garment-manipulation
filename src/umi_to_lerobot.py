#!/usr/bin/env python3
"""[2026-10-14 Agent B] HandUMI UMIalign episodes -> LeRobotDataset in the Agent A contract (same layout as the X-VLA converter).

Per episode inputs (produced by the UMI pose pipeline, not by this script):
  derived/umi_fusion/smoothed_{left,right}.npz   batch smoother: t (s from t0), T_W_tcp (dataset-TCP axes = contract TCP axes,
                                                 table world W shared by both arms), full_6dof (valid mask)
  sensors.mcap frame_meta                        capture_ns of head_depth / left_wrist / right_wrist video frames
  gripper                                        servo samples when present (hfold_export.gripper_binary), else the visual jaw
                                                 track (jaw_visual.visual_gripper); both binary, 1 = open, 0 = closed
t0 is recomputed exactly as run_smoother.main: min(first left_wrist, first right_wrist, first canonical t_ns).
Grid 20 Hz (contract ACTION_DT_S) inside the common valid span; pose = linear / SLERP between full-6DoF samples (gap <= 150 ms),
gripper = nearest sample, images = nearest frame (<= 60 ms).  Both arms live in W, so T_LR = inv(T_L) @ T_R is VALID here
(unlike X-VLA).  Actions are stored as the absolute next-frame target; training/rel_chunk.py builds inv(T(t)) @ T(t+k).
usage: umi_to_lerobot.py --out ROOT --repo-id NAME --episodes EP_DIR ... [--split train|val|all|both]
"""
import argparse, csv, hashlib, json, pathlib, shutil, sys

import cv2, numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(pathlib.Path.home() / "ego_cart20"))   # cube-project package for the ego-data branches only (lazy imports below); see docs/provenance.md
from xvla_core import abs_action_rows, nearest_frame, resample_pose, valid_segments  # noqa: E402
from contracts.action_contract import ACTION_DT_S, T_to_pose9, inv_T  # noqa: E402
from contracts.observation_contract import (PROPRIO_DIM, PROPRIO_NAMES, P_GRIP_L, P_GRIP_R, P_L_POSE, P_R_POSE,  # noqa: E402
                                            P_T_LR, P_T_LR_VALID)
from contracts.skill_contract import SKILL_INSTRUCTION, Skill  # noqa: E402

CAMS = {"global": ("head_depth", (320, 240)), "left_wrist": ("left_wrist", (384, 216)), "right_wrist": ("right_wrist", (384, 216))}
MAX_SRC_GAP_S, MAX_IMG_RESIDUAL_S, MIN_SECONDS = 0.15, 0.06, 4.0
FUSION = "umi_fusion"
ROBOTIZED_TCP_MAPPED = False      # [2026-10-16] True when --fusion-dir is a robotize output whose meta says tcp_map_applied
ABS_NAMES = ([f"abs_left_{n}" for n in ("x", "y", "z", *[f"r{i}" for i in range(6)])] + [f"abs_right_{n}" for n in ("x", "y", "z", *[f"r{i}" for i in range(6)])]
             + ["abs_gripper_left", "abs_gripper_right"])
META = dict(data_source="UMI", source_robot="HANDUMI_X2", skill="UNFOLD_ALIGN", instruction=SKILL_INSTRUCTION[Skill.UNFOLD_ALIGN],
            pose_source="umi_fusion batch smoother (Orbbec anchor + IMU + MASt3R)", pose_frame="table world W (both arms)",
            tcp_axes="dataset-TCP (contract)", tcp_point="HandUMI jaw TCP (handumi_camera_tcp_v2: t_camera_tcp=[0,0.0553,0.1225] m); NOT the Piper flange used by the X-VLA pretrain (flange + 0.136 m) -> map before combining", T_LR_valid=True, gripper="binary 1=open (servo or visual jaw)",
            rate_hz=1.0 / ACTION_DT_S)


def split_of(key):
    return "val" if int(hashlib.sha1(key.encode()).hexdigest(), 16) % 10 == 0 else "train"


def proprio_rows_lr(T_L, T_R, g_L, g_R):
    n = len(T_L); v = np.zeros((n, PROPRIO_DIM))
    v[:, P_L_POSE] = T_to_pose9(inv_T(T_L[0])[None] @ T_L); v[:, P_R_POSE] = T_to_pose9(inv_T(T_R[0])[None] @ T_R)
    v[:, P_GRIP_L], v[:, P_GRIP_R] = g_L, g_R; v[:, P_T_LR] = T_to_pose9(inv_T(T_L) @ T_R); v[:, P_T_LR_VALID] = 1.0
    return v.astype(np.float32)


def gripper_at(ep, side, t_ns, servo):
    from ego_cart20.sources.hfold_export import gripper_binary
    if len(servo) > 10:
        b, _, info = gripper_binary(servo, side, t_ns, ep=ep); return b, np.ones(len(t_ns), bool), "servo"
    from ego_cart20.sources.jaw_visual import visual_gripper
    b, v, _ = visual_gripper(ep, side, t_ns); return np.asarray(b, float), np.asarray(v, bool), "visual"


RENDER = None          # set by --wrist-render c922: (C922Renderer, tilt T)
TCP_SHIFT = None       # set by --tcp-shift-x: 4x4 translation along contract x


def build(ep):
    from ego_cart20.sources.hfold_export import read_mcap
    fm, gr = read_mcap(ep); cz = np.load(ep / "derived/body_track/canonical.npz", allow_pickle=True)
    t0 = int(min(fm["left_wrist"][0, 1], fm["right_wrist"][0, 1], cz["t_ns"][0]))
    S = {s: np.load(ep / f"derived/{FUSION}/smoothed_{s}.npz") for s in ("left", "right")}
    lo = max(max(S[s]["t"][S[s]["full_6dof"]][0] for s in S), (fm["head_depth"][0, 1] - t0) / 1e9)
    hi = min(min(S[s]["t"][S[s]["full_6dof"]][-1] for s in S), (fm["head_depth"][-1, 1] - t0) / 1e9)
    t = lo + ACTION_DT_S * np.arange(int(np.floor((hi - lo) / ACTION_DT_S)) + 1); t_ns = (t0 + t * 1e9).astype(np.int64)
    out = dict(t=t, valid=np.ones(len(t), bool), src={})
    for s in ("left", "right"):
        f = S[s]["full_6dof"].astype(bool); Tsrc = S[s]["T_W_tcp"][f]
        if RENDER is not None and not ROBOTIZED_TCP_MAPPED:
            Tsrc = Tsrc @ RENDER[1]                         # contract TCP follows the tilted virtual wrist camera
                                                            # ([2026-10-16] skipped for ego_cart20/robotize outputs: they already carry it)
        if TCP_SHIFT is not None:
            Tsrc = Tsrc @ TCP_SHIFT                         # origin moved along contract x (e.g. X-VLA flange-equivalent)
        T, _, v, _ = resample_pose(S[s]["t"][f], Tsrc, np.zeros(f.sum()), t, max_gap=MAX_SRC_GAP_S)
        g, gv, src = gripper_at(ep, s, t_ns, gr[s]); out[f"T_{s}"] = T; out[f"g_{s}"] = g; out["valid"] &= v & gv; out["src"][s] = src
    for k, (cam, _) in CAMS.items():
        idx, res = nearest_frame(fm[cam][:, 1].astype(np.float64), t_ns.astype(np.float64)); out[f"img_{k}"] = fm[cam][idx, 0]; out["valid"] &= res <= MAX_IMG_RESIDUAL_S * 1e9
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--repo-id", required=True)
    ap.add_argument("--episodes", nargs="+", required=True); ap.add_argument("--split", choices=["train", "val", "all", "both"], default="both")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--wrist-render", choices=["raw", "c922"], default="raw", help="c922: re-render wrists as the reBot C922 (datasets/wrist_c922.py)")
    ap.add_argument("--wrist-tilt-deg", type=float, default=None)
    ap.add_argument("--fusion-dir", default="umi_fusion",
                    help="[2026-10-15] smoother output dir under derived/: umi_fusion_rcb3fix = data_v003 camera<-IMU rotation + tips-fixed r_c "
                         "(old umi_fusion had TCP 25-36 cm off the jaw tips)")
    ap.add_argument("--tcp-shift-x", type=float, default=0.0,
                    help="move the TCP origin along contract x (m); -0.136 = X-VLA Piper-flange equivalent (0.136 m behind)")
    a = ap.parse_args()
    global RENDER, TCP_SHIFT, FUSION, ROBOTIZED_TCP_MAPPED
    FUSION = a.fusion_dir
    rm = [json.loads(p.read_text()) for e in a.episodes[:1] for p in [pathlib.Path(e).expanduser() / "derived" / FUSION / "robotize_meta.json"] if p.exists()]
    if rm:
        ROBOTIZED_TCP_MAPPED = bool(rm[0].get("tcp_map_applied"))
        META.update(robotized=dict(mode=rm[0]["mode"], orientation=rm[0]["orientation"], scale_xyz=rm[0]["scale_xyz"], time_scale=rm[0].get("time_scale"),
                                   frame=rm[0]["frame"], tcp_map=rm[0]["tcp_map"], tool=rm[0]["tool"]))
    if FUSION != "umi_fusion":
        META.update(pose_source=f"umi_fusion batch smoother (Orbbec anchor + IMU + MASt3R), derived/{FUSION}: camera<-IMU R_cb data_v003 + camera->body lever r_c fixed to the jaw-tape measurement tips_v001 (TCP within ~5 cm of the jaw tips in the Orbbec check)")
    if a.tcp_shift_x:
        TCP_SHIFT = np.eye(4); TCP_SHIFT[0, 3] = a.tcp_shift_x
        META.update(tcp_point=f"UMI TCP shifted {a.tcp_shift_x:+.3f} m along contract x (X-VLA flange-equivalent for pretrain->UMI FT)")
    if a.wrist_render == "c922":
        from wrist_c922 import C922Renderer, tcp_tilt_T
        r = C922Renderer(a.wrist_tilt_deg); RENDER = (r, tcp_tilt_T(r.tilt))
        CAMS["left_wrist"] = ("left_wrist", (320, 240)); CAMS["right_wrist"] = ("right_wrist", (320, 240))
        META.update(wrist_render=f"c922 640x480 (robot_right_wrist_c922_v001 K+dist), virtual tilt {r.tilt:+.1f} deg, resized 320x240 (no stretch)",
                    tcp_axes=f"contract TCP aligned to the VIRTUAL C922 wrist camera (T_W_tcp @ tcp_tilt_T({r.tilt:+.1f}))")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    base = pathlib.Path(a.out).expanduser(); roots = {s: base.parent / f"{base.name}_{s}" for s in ("train", "val")} if a.split == "both" else {"one": base}
    for r in roots.values():
        if r.exists():
            if not a.overwrite: sys.exit(f"{r} exists (pass --overwrite)")
            shutil.rmtree(r)
    feats = {f"observation.images.{k}": dict(dtype="video", shape=(wh[1], wh[0], 3), names=["height", "width", "channels"]) for k, (_, wh) in CAMS.items()}
    feats.update({"observation.state": dict(dtype="float32", shape=(PROPRIO_DIM,), names=list(PROPRIO_NAMES)),
                  "action": dict(dtype="float32", shape=(20,), names=ABS_NAMES), "pose_abs": dict(dtype="float32", shape=(20,), names=ABS_NAMES)})
    dss, man = {}, []
    for e in a.episodes:
        ep = pathlib.Path(e).expanduser().resolve(); key = f"{ep.parent.name}/{ep.name}"; row = dict(episode_id=key, split=split_of(key), status="", reject_reason="")
        dkey = row["split"] if a.split == "both" else "one"
        if a.split in ("train", "val") and row["split"] != a.split: continue
        try:
            meta = json.loads((ep / "episode_meta.json").read_text())
            if meta.get("exclude_from_training"): raise ValueError("excluded (exclude_from_training)")
            for f in (f"derived/{FUSION}/smoothed_left.npz", f"derived/{FUSION}/smoothed_right.npz", "derived/body_track/canonical.npz"):
                if not (ep / f).exists(): raise FileNotFoundError(f"missing {f}")
            b = build(ep); segs = valid_segments(b["valid"], int(MIN_SECONDS / ACTION_DT_S))
            row.update(grid_frames=len(b["t"]), valid_frames=int(b["valid"].sum()), n_segments=len(segs), frames_written=int(sum(y - x for x, y in segs)),
                       grip_src_L=b["src"]["left"], grip_src_R=b["src"]["right"], umi_tags=json.dumps(meta.get("umi_tags", {})))
            if not segs: raise ValueError("no valid segment >= 4 s")
            caps = {k: cv2.VideoCapture(str(ep / f"{cam}.mp4")) for k, (cam, _) in CAMS.items()}
            cache = {k: {} for k in CAMS}
            def frame(k, i):                                       # sequential-friendly decode with a small cache
                c = cache[k]
                if i not in c:
                    cap = caps[k]; pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    if i < pos or i > pos + 30: cap.set(cv2.CAP_PROP_POS_FRAMES, i); pos = i
                    while pos <= i:
                        ok, im = cap.read(); pos += 1
                        if not ok: raise IOError(f"{k}: frame {i} unreadable")
                        if RENDER is not None and k != "global":
                            im = RENDER[0](k.split("_")[0], im)            # 1080p fisheye -> 640x480 C922 view
                        c[pos - 1] = cv2.cvtColor(cv2.resize(im, CAMS[k][1], interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                    for old in [x for x in c if x < i - 5]: del c[old]
                return c[i]
            if dkey not in dss:
                dss[dkey] = LeRobotDataset.create(a.repo_id + ("_" + dkey if dkey != "one" else ""), fps=int(round(1 / ACTION_DT_S)), features=feats,
                                                  root=roots[dkey], robot_type="handumi_x2", use_videos=True, image_writer_threads=0, streaming_encoding=True, encoder_threads=2)
            ds = dss[dkey]
            for x, y in segs:
                TL, TR, gL, gR = b["T_left"][x:y], b["T_right"][x:y], b["g_left"][x:y], b["g_right"][x:y]
                st = proprio_rows_lr(TL, TR, gL, gR); act = abs_action_rows(TL, TR, gL, gR)
                pa = np.concatenate([T_to_pose9(TL), T_to_pose9(TR), gL[:, None], gR[:, None]], -1).astype(np.float32)
                for j, i in enumerate(range(x, y)):
                    fr = {f"observation.images.{k}": frame(k, int(b[f"img_{k}"][i])) for k in CAMS}
                    fr.update({"observation.state": st[j], "action": act[j], "pose_abs": pa[j], "task": META["instruction"]}); ds.add_frame(fr)
                ds.save_episode()
            row.update(status="converted")
        except Exception as ex:
            row.update(status="rejected", reject_reason=f"{type(ex).__name__}: {ex}")
        man.append(row); print(f"{key} {row['status']} {row['reject_reason']} frames={row.get('frames_written')} grip={row.get('grip_src_L')}/{row.get('grip_src_R')}", flush=True)
    for d in dss.values(): d.finalize()
    cols = sorted({k for r in man for k in r}, key=lambda k: (k not in ("episode_id", "status", "reject_reason"), k))
    for dkey, r in roots.items():
        r.mkdir(parents=True, exist_ok=True)
        with open(r / "conversion_manifest.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows([x for x in man if dkey == "one" or x["split"] == dkey])
        (r / "source_meta.json").write_text(json.dumps(dict(META, converter="umi_to_lerobot.py", contract_version=__import__("contracts").CONTRACT_VERSION), indent=1))
    print("done", sum(r["status"] == "converted" for r in man), "converted,", sum(r["status"] == "rejected" for r in man), "rejected")


if __name__ == "__main__":
    main()
