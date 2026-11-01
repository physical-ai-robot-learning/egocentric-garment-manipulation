#!/usr/bin/env python3
"""[2026-10-22] AprilTag-rig UMI laundry-folding takes (ego_collector tag_rig_record --proto task --wrist) -> LeRobotDataset in the Agent A
contract, same layout as datasets/umi_to_lerobot.py (20 Hz grid, absolute next-frame action targets, contract proprio, 3 cameras),
shared by the Multi-Task DiT (training/train_dit.py) and ACT (training/train_act.py) launchers.

Inputs per take (READ-ONLY; nothing under ego_collector/datasets is written):
  meta.json                 t_start_monotonic_ns + protocol; every entry whose text starts with '자유:' is one demo = one episode
  t_ns.npy, color_%05d.jpg  fixed Orbbec RGB (~15 Hz, 1280x720)                                   -> observation.images.global
  wrist/, wrist_left/       UMI fisheye wrist cams (~30 Hz) + umi_jaw.npz (g_cont 0 closed .. 1 open, valid, ev_t)
  umi_traj_{right,left}.npz scripts/umi_tcp_traj.py: jaw-tip TCP p_B, rig rotation R_B_tcp (robot base frame B), ntags

ARM ASSIGNMENT (prep stage, per take; written to <derived>/<take>/assignment.json):
  The rig labels in umi_traj_<side>.npz come from the tag ids (R 200-202 / L 210-212) + their IMU.  On 2026-10-22 these labels are
  SWAPPED w.r.t. the robot (right-labelled rig at base +y = robot LEFT side) and w.r.t. the wrist cameras (cam 'wrist' moves with rig
  'left').  So the converter never trusts labels:
    robot arm  := sign of the rig's median base-frame y on tag-seen frames (+y = LEFT arm, reBot base convention)
    wrist cam  := the camera whose optical-flow magnitude correlates best with that rig's angular speed (2x2 test, margin required)
    gripper    := umi_jaw.npz of that camera
    orientation:= R_B_E = R_B_tcp @ R_U_E(rig label)  (configs/calibration/umi_gripper_align[_left]_v001.json, as umi_replay_bi does)
  The arm-labelled trajectories are copied to <derived>/<take>/umi_traj_<arm>.npz and the user's robotizer
  (scripts/umi_robotize_bi.py --clear 0 --sep 0.06 --min-reach 0) is run THERE -> umi_traj_<arm>_robot.npz.  Poses = robotized p_B.
  (Running the robotizer on the raw labels pushes each hand ~23 cm across the midline on these takes -> those _robot.npz are invalid.)

Per 20 Hz frame (grid inside the demo window clipped to the recorded data):
  observation.images.{global,left_wrist,right_wrist}  nearest frame (<= 50 ms global / 40 ms wrist), all 384x216 (16:9, ONE shape: DiT)
  observation.state  contract proprio 30D (episode-anchor poses, grippers, T_LR valid = 1: both rigs are in the same base frame B)
  action             ABSOLUTE next-frame target [L pose9 | R pose9 | gL | gR]; training builds inv(T(t)) @ T(t+k)
  pose_abs           current-frame absolute [L pose9 | R pose9 | gL | gR]
  quality            [tag_seen_L, tag_seen_R, robotize_L, robotize_R, jaw_valid_L, jaw_valid_R, jaw_event_L, jaw_event_R]
  weight             suggested per-frame loss weight: 1.0 good | 0.5 medium (robotize 1 or a rig without tags) |
                     0.1 bad (robotize 2, or jaw keypoints not detected within +-0.5 s of a jaw open/close event).  NOT used by the
                     stock DiT / ACT losses yet; stored so a weighted loss / frame filter can be added without re-conversion.
  src_index          [global, left_wrist, right_wrist] source frame indices; img_residual_ms likewise (provenance / QA)
Frames inside a no-tag run > 1 s (either arm) are MASKED (episode split there, like any invalid run).
Frames whose pose bracket > 150 ms, jaw sample > 60 ms away or image residual above tolerance are invalid; an episode is split at
invalid runs (segments < --min-seconds dropped; same rule as umi_to_lerobot).
Split: by EPISODE (take/demo key), seeded shuffle, 80/10/10 over ALL candidate demos -> <out>_{train,val,test}; meta/split.json in
each root + episode_table.csv.  --clean-min-tag 0.6 keeps only episodes whose per-arm tag-seen fraction is >= 0.6 (split unchanged).
usage: tag_rig_to_lerobot.py --out data/lerobot/tagrig_laundry_v1 --takes TAKE_DIR ... [--clean-min-tag 0.6] [--max-demos N]
"""
import argparse, csv, json, pathlib, shutil, subprocess, sys, time

import cv2, numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(pathlib.Path.home() / "laundry_system"))
from xvla_core import abs_action_rows, nearest_frame, resample_pose, valid_segments  # noqa: E402
from umi_to_lerobot import ABS_NAMES, proprio_rows_lr  # noqa: E402
from contracts.action_contract import ACTION_DT_S, T_to_pose9, make_T  # noqa: E402
from contracts.observation_contract import PROPRIO_DIM, PROPRIO_NAMES  # noqa: E402

EGO = pathlib.Path.home() / "ego_collector"
EGO_PY = EGO / ".venv/bin/python"
ALIGN = {"right": EGO / "configs/calibration/umi_gripper_align_v001.json", "left": EGO / "configs/calibration/umi_gripper_align_left_v001.json"}
ROBOTIZE_ARGS = ["--clear", "0", "--sep", "0.06", "--min-reach", "0"]
TASK = "laundry folding"
IMG_WH = (384, 216)
CAMS = ("global", "left_wrist", "right_wrist")
MAX_SRC_GAP_S, MAX_JAW_GAP_S, MIN_SECONDS = 0.15, 0.06, 5.0      # QA 5: demo window < 5 s -> drop
MIN_SEG_S = 2.0                   # 10-22: masked pieces >= 2 s (40 frames > the 32-step chunk) are kept as LeRobot episodes
IMG_TOL_S = {"global": 0.05, "left_wrist": 0.04, "right_wrist": 0.04}
JAW_EVENT_WIN_S = 0.5
GRIP_RAMP_S = 0.1                 # binary jaw state -> linear ramp of 0.1 s at each transition (box filter of the step)
TAG_GAP_MASK_S = 1.0              # QA 1: a no-tag run > 1 s -> those frames weight 0.1
MIN_GOOD_FRAC = 0.3               # QA 1 (user 10-22: 0.3, was 0.6): drop the episode when < 30 % of its active frames are valid and outside long tag gaps
FLICKER_TOGGLES_PER_S = 2
IDLE_SPEED_MPS, IDLE_MARGIN_S = 0.02, 1.0   # QA 6: idle head/tail trim (fixed 45 s demos end ~20 s idle with the UMIs on the table)         # QA 3: > 2 jaw state toggles inside any 1 s window -> flag
# 10-22 reviewer rule (fused pose quality, per episode); jump = neighbour-inconsistent step, contact = +-0.5 s around grasp (close) events: TRAIN (weight 1.0) / DOWNWEIGHT (0.3) / EXCLUDE
JUMP_MM, JUMP_DEG, JUMP_BIG_MM = 30.0, 15.0, 100.0          # 20 Hz frame-to-frame after a 3-frame position smoothing
CONTACT_DOWN_MM, CONTACT_EXCL_MM = 60.0, 150.0              # 10-22 auto-decision: grasp-instant robotize displacement median 136 mm (midline y-compression, --sep 0.06); 60 mm excl would drop 69/75
JUMPS_EXCL = 3                                              # >= 3 jumps (or any > 100 mm) -> exclude; 1-2 -> downweight
COVERAGE_WARN, GAP_WARN_S = 0.95, 2.0
DOWN_W = 0.3
PAIR_MARGIN = 0.15
QUALITY_NAMES = ["tag_seen_L", "tag_seen_R", "robotize_L", "robotize_R", "jaw_valid_L", "jaw_valid_R", "jaw_event_L", "jaw_event_R"]
META = dict(data_source="UMI", source_robot="HANDUMI_TAGRIG_X2", skill="LAUNDRY_FOLDING (full task; not a contracts.skill_contract.Skill yet)",
            instruction=TASK, pose_source="AprilTag rig (Orbbec) + IMU rotation fill (scripts/umi_tcp_traj.py) -> umi_robotize_bi.py "
            f"{' '.join(ROBOTIZE_ARGS)} (robot_feasible_tcp; p_B_human kept in the source only)",
            pose_frame="reBot base B (configs/calibration/robot_cam_v001.json), both arms; T_LR valid",
            tcp_point="UMI jaw-tip touch TCP (umi_tag_rig_v001 tcp_in_rig_m)",
            tcp_axes="reBot gripper EEF axes E: R_B_E = R_B_tcp(rig) @ R_U_E (umi_gripper_align[_left]_v001), i.e. the orientation "
                     "umi_replay_bi commands; NOT the camera-aligned dataset-TCP axes of the older UMIalign sets",
            T_LR_valid=True, gripper="action/state gripper = BINARY umi_jaw `state` (1 = open, 0 = closed) with a 0.1 s linear ramp at transitions; continuous g_cont stored in gripper_cont only. Deploy: > 0.5 open, else full close",
            images=f"global = Orbbec RGB, wrists = raw UMI fisheye; all resized to {IMG_WH[0]}x{IMG_WH[1]} (16:9, no stretch)",
            qa="automatic per episode (episode_table.csv qa_decision / qa_flags / reject_reason): tag gap > 1 s -> weight 0.1; < 60 % good frames -> drop; "
               "< 5 s -> drop; idle head/tail (both arms < 2 cm/s, 1 s smoothing, 1 s margin) trimmed; no open->close event / flicker > 2 toggles per s -> flag; left arm median y must be > 0 (else LR_SWAP flag). "
               "IK / joint-speed / workspace-limit feasibility is NOT checked here (umi_replay_bi --offline does that); robotize class 2 (>= 50 mm) -> weight 0.1",
            episode_rule=dict(jump=f"frame-to-frame > {JUMP_MM:.0f} mm or > {JUMP_DEG:.0f} deg at 20 Hz (3-frame position smoothing), active frames",
                              exclude=f">= {JUMPS_EXCL} jumps, any jump > {JUMP_BIG_MM:.0f} mm, or contact (gripper closed) robotize displacement >= {CONTACT_EXCL_MM:.0f} mm",
                              downweight=f"1-2 jumps or contact robotize displacement >= {CONTACT_DOWN_MM:.0f} mm -> weight {DOWN_W} (used in the loss)",
                              warn=f"fused coverage < {COVERAGE_WARN * 100:.0f} %, longest gap >= {GAP_WARN_S} s"),
            weight_column="`weight` = EPISODE weight (1.0 / 0.3) multiplied into the per-sample loss (training/tagrig_common.py); `weight_frame` = per-frame suggestion (not used)",
            rate_hz=1.0 / ACTION_DT_S, weight_rule="1.0 good | 0.5 robotize 1 or rig without tags | 0.1 robotize 2, tag gap > 1 s, or jaw undetected within 0.5 s of a jaw event")


# ---------------------------------------------------------------- prep: arm assignment + robotize (derived copy)
def _ang_speed(z):
    t = z["t_ns"].astype(np.int64); R = z["R_B_tcp"]
    dR = np.einsum("nji,njk->nik", R[:-1], R[1:]); a = np.arccos(np.clip((np.trace(dR, axis1=1, axis2=2) - 1) / 2, -1, 1))
    return t[1:], a / np.maximum(np.diff(t) / 1e9, 1e-3)


def _flow_mag(cam_dir, t_lo, t_hi, step=2):
    t = np.load(cam_dir / "t_ns.npy"); sel = np.flatnonzero((t >= t_lo) & (t <= t_hi))[::step]; prev, out, used = None, [], []
    for i in sel:
        im = cv2.imread(str(cam_dir / f"wrist_{i:05d}.jpg"))
        if im is None: continue   # 10-27: 152818 lost ~70 wrist jpgs (recorder drop) -> skip, flow is a median over the window
        g = cv2.cvtColor(cv2.resize(im, (160, 90)), cv2.COLOR_BGR2GRAY)
        out.append(0.0 if prev is None else float(np.median(np.linalg.norm(cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 2, 9, 2, 5, 1.1, 0), axis=2))))
        prev = g; used.append(i)
    return t[used][1:], np.array(out[1:])


def demos_of(take):
    m = json.loads((take / "meta.json").read_text()); t0 = int(m["t_start_monotonic_ns"])
    return [dict(idx=i, t0_ns=t0 + int(p["t0"] * 1e9), t1_ns=t0 + int((p["t0"] + p["dur"]) * 1e9), ended_early=bool(p.get("ended_early")))
            for i, p in enumerate([p for p in m["protocol"] if p["text"].startswith("자유:")])]


def assign_arms(take):
    """-> dict(arm -> rig label, arm -> camera dir name, diagnostics)"""
    Z = {s: np.load(take / f"umi_traj_{s}.npz") for s in ("right", "left")}
    ymed = {s: float(np.median(Z[s]["p_B"][Z[s]["ntags"] > 0, 1])) for s in Z}
    if np.sign(ymed["right"]) == np.sign(ymed["left"]): raise ValueError(f"rigs on the same side of the base (median y {ymed})")
    arm_of_rig = {s: ("left" if ymed[s] > 0 else "right") for s in Z}
    # per-demo 2x2 correlation of wrist optical-flow magnitude vs rig angular speed; decide by a vote over ALL demos
    # (124754: single 40 s windows were ambiguous, but crossed > straight in 10/10 demos)
    A = {s: _ang_speed(Z[s]) for s in Z}; C = {}; votes = []
    for d in demos_of(take):
        c1 = {}
        for c in ("wrist", "wrist_left"):
            tc, fl = _flow_mag(take / c, d["t0_ns"], d["t1_ns"], step=3)
            for s in Z:
                ta, a = A[s]; m = (tc >= ta[0]) & (tc <= ta[-1])
                if m.sum() > 50: c1[(s, c)] = float(np.corrcoef(fl[m], np.interp(tc[m], ta, a))[0, 1])
        if len(c1) == 4:
            votes.append((c1[("right", "wrist_left")] + c1[("left", "wrist")]) - (c1[("right", "wrist")] + c1[("left", "wrist_left")]))
            for k, v in c1.items(): C.setdefault(k, []).append(v)
    votes = np.array(votes); C = {k: float(np.mean(v)) for k, v in C.items()}
    agree = max(np.mean(votes > 0), np.mean(votes < 0)) if len(votes) else 0.0
    if len(votes) < 2 or agree < 0.8: raise ValueError(f"camera<->rig pairing ambiguous: votes {np.round(votes, 2).tolist()}")
    straight, crossed = (0.0, 1.0) if np.mean(votes > 0) >= 0.8 else (1.0, 0.0)
    cam_of_rig = {"right": "wrist", "left": "wrist_left"} if straight > crossed else {"right": "wrist_left", "left": "wrist"}
    out = dict(rig_of_arm={arm_of_rig[s]: s for s in Z}, cam_of_arm={arm_of_rig[s]: cam_of_rig[s] for s in Z},
               rig_median_y_m=ymed, flow_corr={f"{s}|{c}": round(v, 3) for (s, c), v in C.items()},
               rig_labels_swapped=bool(arm_of_rig["right"] == "left"), cam_labels_swapped=bool(cam_of_rig["right"] != "wrist"),
               pairing_votes_crossed_minus_straight=np.round(votes, 3).tolist(), pairing_agreement=float(agree))
    if {*out["rig_of_arm"]} != {"left", "right"}: raise ValueError(f"bad assignment {out}")
    # 10-22 user decision: the assignment is automatic for all future takes, so the two independent checks must agree:
    # y-sign (rig -> arm) and optical flow (rig -> camera) must put the 'wrist_left' camera on the robot LEFT arm (cameras are
    # labelled by the hand that wears them).  Any disagreement = something changed physically -> stop, never guess.
    if out["cam_of_arm"] != {"left": "wrist_left", "right": "wrist"}:
        raise SystemExit(f"[{take.name}] ASSIGNMENT CONFLICT: y-sign says rig_of_arm={out['rig_of_arm']} but optical flow pairs "
                         f"cam_of_arm={out['cam_of_arm']} (corr {out['flow_corr']}); check the rigs/cameras before converting")
    return out


ROBOTIZE = "ik_v1"  # "ik_v1" = datasets/robotize_ik.py (user option B, 10-22; FINAL) | "sep" = legacy umi_robotize_bi --sep 0.06 (never for training)
GRASP_OK_MM, GRASP_WARN_MM, GRASP_W = 20.0, 50.0, 0.5   # grasp-window gate (robotized vs fused TCP, max per window): <20 ok, 20-50 weight 0.5, >50 mask
IK_FAIL_MM, ARM_GAP_M = 10.0, 0.12
POSE_SRC = ""      # "" = tag-only umi_traj_<rig>.npz | "_fused" = MASt3R + AprilTag gated fusion umi_traj_<rig>_fused.npz (final build)
GAP_KEYS = ("fused_valid", "valid", "ok")   # fused files: if one of these per-sample masks exists it replaces ntags > 0 for gap masking


IK_WORKERS = 6
IK_PY = pathlib.Path.home() / "xvla-mac/bin/python"


def prep_take(take, derived_root, force=False):
    d = derived_root / (take.name + POSE_SRC); aj = d / "assignment.json"
    if not (aj.exists() and not force and all((d / f"umi_traj_{a}.npz").exists() for a in ("left", "right"))):
        d.mkdir(parents=True, exist_ok=True)
        asg = assign_arms(take)
        for arm, rig in asg["rig_of_arm"].items():
            z = dict(np.load(take / f"umi_traj_{rig}{POSE_SRC}.npz", allow_pickle=True)); z["source_rig_label"] = np.array(rig)
            z["pose_src"] = np.array(POSE_SRC or "tag_only")
            np.savez(d / f"umi_traj_{arm}.npz", **z)
        shutil.copy(take / "segments.json", d / "segments.json")
        asg.update(take=str(take), pose_src=POSE_SRC or "tag_only", created=time.strftime("%Y-%m-%d %H:%M:%S"))
        aj.write_text(json.dumps(asg, indent=1, ensure_ascii=False))
    asg = json.loads(aj.read_text())
    if ROBOTIZE == "sep" and not all((d / f"umi_traj_{a}_robot.npz").exists() for a in ("left", "right")):     # legacy, never for training
        r = subprocess.run([str(EGO_PY), str(EGO / "scripts/umi_robotize_bi.py"), str(d), *ROBOTIZE_ARGS], capture_output=True, text=True)
        if r.returncode: raise RuntimeError(f"robotize failed: {r.stderr[-500:]}")
        (d / "robotize.log").write_text(r.stdout)
    if ROBOTIZE == "ik_v1":                                  # datasets/robotize_ik.py per demo (skips demos already done)
        r = subprocess.run([str(IK_PY), str(HERE / "robotize_ik.py"), str(take), str(d), "--workers", str(IK_WORKERS)], capture_output=True, text=True)
        (d / "robotize_ik.log").open("a").write(r.stdout + r.stderr[-2000:])
        if r.returncode: raise RuntimeError(f"robotize_ik failed: {r.stderr[-500:]}")
    asg["robotize"] = ROBOTIZE; return asg


# ---------------------------------------------------------------- per-demo grid
def build_demo(take, dtake, asg, demo):
    tc = np.load(take / "t_ns.npy").astype(np.int64)
    trajs = {a: np.load(dtake / (f"umi_traj_{a}.npz" if ROBOTIZE == "ik_v1" else f"umi_traj_{a}_robot.npz"), allow_pickle=True) for a in ("left", "right")}
    IK = None
    if ROBOTIZE == "ik_v1":
        f = dtake / "ik_v1" / f"demo{demo['idx']:02d}.npz"
        if not f.exists(): raise FileNotFoundError(f"ik_v1 robotization missing ({f.name})")
        IK = np.load(f)
    jaws = {a: (take / asg["cam_of_arm"][a] / "umi_jaw.npz") for a in ("left", "right")}
    for a, p in jaws.items():
        if not p.exists(): raise FileNotFoundError(f"missing {p.parent.name}/umi_jaw.npz ({a} arm)")
    jaws = {a: np.load(p) for a, p in jaws.items()}
    for a, J in jaws.items():
        if "g_is_aperture" not in J.files or int(J["g_is_aperture"]) != 1: raise ValueError(f"{a} arm jaw npz has g_is_aperture != 1 (not an aperture label)")
    tcam = {"global": tc, "left_wrist": np.load(take / asg["cam_of_arm"]["left"] / "t_ns.npy").astype(np.int64),
            "right_wrist": np.load(take / asg["cam_of_arm"]["right"] / "t_ns.npy").astype(np.int64)}
    lo = max(demo["t0_ns"], *(int(trajs[a]["t_ns"][0]) for a in trajs), tc[0]); hi = min(demo["t1_ns"], *(int(trajs[a]["t_ns"][-1]) for a in trajs), tc[-1])
    if hi - lo < MIN_SECONDS * 1e9: raise ValueError(f"demo window inside the recording only {(hi - lo) / 1e9:.1f} s (take truncated)")
    g_ns = lo + (np.arange(int((hi - lo) / 1e9 / ACTION_DT_S) + 1) * ACTION_DT_S * 1e9).astype(np.int64)
    tg = (g_ns - lo) / 1e9
    out = dict(t=tg, t_ns=g_ns, valid=np.ones(len(g_ns), bool), window_s=(hi - lo) / 1e9)
    for a, s in (("left", "L"), ("right", "R")):
        z = trajs[a]; ts = (z["t_ns"].astype(np.int64) - lo) / 1e9
        k, _ = nearest_frame(ts, tg)
        if IK is not None:                                  # robot-feasible label = FK(q) of the ik_v1 joint path (already reBot EEF axes)
            ti = (IK["t_ns"].astype(np.int64) - lo) / 1e9
            T, _, v, _ = resample_pose(ti, make_T(IK[f"{a}_R"], IK[f"{a}_p"]), np.zeros(len(ti)), tg, max_gap=MAX_SRC_GAP_S)
            ki, _ = nearest_frame(ti, tg)
            Ph = np.stack([np.interp(tg, ti, IK[f"{a}_p_h"][:, j]) for j in range(3)], 1)
            out[f"Ph_{a}"] = Ph; out[f"ikmm_{s}"] = IK[f"{a}_ik_mm"][ki]; out[f"lim_{s}"] = IK[f"{a}_lim"][ki]; out[f"grasp_{s}"] = IK[f"{a}_grasp"][ki]
            out["gap_m"] = IK["gap"][ki]
        else:
            T_src = make_T(z["R_B_tcp"] @ np.array(json.loads(ALIGN[str(z["source_rig_label"])].read_text())["R_U_E"]), z["p_B"])
            T, _, v, _ = resample_pose(ts, T_src, np.zeros(len(ts)), tg, max_gap=MAX_SRC_GAP_S)
        J = jaws[a]; tj = (J["t_ns"].astype(np.int64) - lo) / 1e9
        g_cont = np.clip(np.interp(tg, tj, J["g_cont"].astype(float)), 0, 1); kj, rj = nearest_frame(tj, tg)
        fine = tg[:, None] + np.linspace(-GRIP_RAMP_S / 2, GRIP_RAMP_S / 2, 21)[None]       # box filter of width 0.1 s = linear ramp
        g = J["state"][nearest_frame(tj, fine.ravel())[0]].reshape(fine.shape).astype(float).mean(1)
        st_b = J["state"][kj].astype(int); win = int(round(1.0 / ACTION_DT_S))
        tog = np.r_[0, np.abs(np.diff(st_b))]; tog_1s = np.convolve(tog, np.ones(win), "same") if len(tog) >= win else tog
        out[f"close_events_{s}"] = int(np.sum(np.diff(st_b) < 0)); out[f"flicker_{s}"] = bool(np.max(tog_1s, initial=0) > FLICKER_TOGGLES_PER_S)
        out[f"gc_{a}"] = g_cont
        ev = (np.asarray(J["ev_t"], np.int64) - lo) / 1e9 if "ev_t" in J.files else np.zeros(0)
        near_ev = np.zeros(len(tg), bool)
        for e in ev: near_ev |= np.abs(tg - e) <= JAW_EVENT_WIN_S
        out[f"T_{a}"] = T; out[f"g_{a}"] = g
        out[f"ymed_{s}"] = float(np.median(T[:, 1, 3]))
        gk = next((g for g in GAP_KEYS if g in z.files), None)        # "tracked" flag: fused validity mask if present, else tag seen
        if gk is None and "slam_ok" in z.files: vsrc = z["slam_ok"].astype(bool) | (z["ntags"] > 0)   # pre-final fused files
        else: vsrc = (z[gk] > 0) if gk else (z["ntags"] > 0)
        out[f"tag_{s}"] = vsrc[k].astype(np.float32)
        msk = (z["t_ns"] >= lo) & (z["t_ns"] <= hi); vv = vsrc[msk]; tt = z["t_ns"][msk] / 1e9
        x = np.r_[0, (~vv).astype(int), 0]; g0, g1 = np.flatnonzero(np.diff(x) == 1), np.flatnonzero(np.diff(x) == -1)
        out[f"cov_{s}"] = float(vv.mean()) if len(vv) else 0.0
        out[f"maxgap_{s}"] = float(max([tt[min(e, len(tt) - 1)] - tt[max(b_ - 1, 0)] for b_, e in zip(g0, g1)], default=0.0))
        if IK is not None: out[f"disp_{s}"] = np.linalg.norm(T[:, :3, 3] - out[f"Ph_{a}"], axis=1) * 1000
        else: out[f"disp_{s}"] = np.linalg.norm(z["p_B"] - z["p_B_human"], axis=1)[k] * 1000 if "p_B_human" in z.files else np.zeros(len(k))
        sg = z["seg_src"][k] if "seg_src" in z.files else np.zeros(len(k), int)
        out[f"segchg_{s}"] = np.r_[False, sg[1:] != sg[:-1]]; out[f"rob_{s}"] = (np.digitize(out[f"disp_{s}"], [GRASP_OK_MM, GRASP_WARN_MM]).astype(np.float32) if IK is not None else z["robotize_quality"][k].astype(np.float32))
        out[f"jv_{s}"] = J["valid"][kj].astype(np.float32); out[f"ev_{s}"] = near_ev.astype(np.float32)
        out["valid"] &= v & (rj <= MAX_JAW_GAP_S)
    for c in CAMS:
        idx, res = nearest_frame(tcam[c].astype(np.float64), g_ns.astype(np.float64)); out[f"img_{c}"] = idx; out[f"res_{c}"] = res / 1e6
        out["valid"] &= res <= IMG_TOL_S[c] * 1e9
    long_gap = np.zeros(len(tg), bool)
    for s in ("L", "R"):
        x = np.r_[0, (out[f"tag_{s}"] < 0.5).astype(int), 0]; st_, en_ = np.flatnonzero(np.diff(x) == 1), np.flatnonzero(np.diff(x) == -1)
        for p0, p1 in zip(st_, en_):
            if (p1 - p0) * ACTION_DT_S > TAG_GAP_MASK_S: long_gap[p0:p1] = True
    out["long_gap"] = long_gap
    sp = np.zeros(len(tg))
    for a in ("left", "right"):
        v = np.r_[0, np.linalg.norm(np.diff(out[f"T_{a}"][:, :3, 3], axis=0), axis=1)] / ACTION_DT_S
        sp = np.maximum(sp, np.convolve(v, np.ones(20) / 20, "same"))
    act = np.flatnonzero(sp > IDLE_SPEED_MPS); m = int(IDLE_MARGIN_S / ACTION_DT_S); keep = np.zeros(len(tg), bool)
    if len(act): keep[max(act[0] - m, 0): act[-1] + m + 1] = True
    out["idle_head_s"] = float(np.argmax(keep) * ACTION_DT_S) if keep.any() else out["window_s"]
    out["idle_tail_s"] = float((len(keep) - 1 - np.flatnonzero(keep)[-1]) * ACTION_DT_S) if keep.any() else 0.0
    out["valid_pre_mask"] = out["valid"].copy(); out["valid"] &= ~long_gap & keep      # QA 1: MASK (episode split there) -- stock losses ignore `weight`
    act_m = out["valid"]
    for a, s in (("left", "L"), ("right", "R")):
        P = out[f"Ph_{a}"] if f"Ph_{a}" in out else out[f"T_{a}"][:, :3, 3]; Ps = np.copy(P)      # jumps: on the FUSED (human) path; Ps[1:-1] = (P[:-2] + P[1:-1] + P[2:]) / 3
        dp = np.r_[0, np.linalg.norm(np.diff(Ps, axis=0), axis=1)] * 1000
        R = out[f"T_{a}"][:, :3, :3]; dR = np.einsum("nji,njk->nik", R[:-1], R[1:])
        dr = np.r_[0, np.degrees(np.arccos(np.clip((np.trace(dR, axis1=1, axis2=2) - 1) / 2, -1, 1)))]
        # 10-22: an ABSOLUTE 30 mm/step (0.6 m/s) is the p95 of normal folding motion (tag-only and fused alike, max 77 mm, smooth) ->
        # a jump is a step INCONSISTENT with its neighbours: |dp(t) - mean(dp(t-2..t+2) without t)| > 30 mm, or a rotation step > 15 deg
        v = np.diff(Ps, axis=0, prepend=Ps[:1]); vn = np.zeros_like(v)
        for o in (-2, -1, 1, 2): vn += np.roll(v, o, axis=0)
        dev = np.linalg.norm(v - vn / 4, axis=1) * 1000; dev[:3] = 0; dev[-3:] = 0
        jump = act_m & ((dev > JUMP_MM) | (dr > JUMP_DEG)); near_stitch = np.convolve(out[f"segchg_{s}"].astype(int), np.ones(5), "same") > 0
        out[f"jumps_{s}"] = int(jump.sum()); out[f"stitch_jumps_{s}"] = int((jump & near_stitch).sum()); out[f"jump_max_mm_{s}"] = float(dev[act_m].max()) if act_m.any() else 0.0
        # contact = the GRASP instant (+-0.5 s around open->close events): the jaw is closed ~90 % of the time, so "closed" != contact
        gb = (out[f"g_{a}"] > 0.5).astype(int); ce = np.flatnonzero(np.diff(gb) < 0) + 1; contact = np.zeros(len(gb), bool)
        for c in ce: contact[max(c - 10, 0): c + 11] = True
        contact &= act_m
        d = out[f"disp_{s}"]
        out[f"disp_mean_{s}"] = float(d[act_m].mean()) if act_m.any() else 0.0; out[f"disp_max_{s}"] = float(d[act_m].max()) if act_m.any() else 0.0
        out[f"disp_contact_max_{s}"] = float(d[contact].max()) if contact.any() else 0.0
    out["grasp_w"] = np.ones(len(tg), np.float32); out["grasp_total"] = out["grasp_fail"] = out["grasp_warn"] = 0; out["grasp_max_mm"] = []
    if IK is not None:
        for a, s in (("left", "L"), ("right", "R")):
            gw = out[f"grasp_{s}"].astype(bool); x = np.r_[0, gw.astype(int), 0]
            for g0_, g1_ in zip(np.flatnonzero(np.diff(x) == 1), np.flatnonzero(np.diff(x) == -1)):
                if not out["valid"][g0_:g1_].any(): continue           # grasp inside masked / idle frames: not counted
                mx = float(out[f"disp_{s}"][g0_:g1_].max()); out["grasp_total"] += 1; out["grasp_max_mm"].append(round(mx))
                if mx > GRASP_WARN_MM: out["grasp_fail"] += 1; out["valid"][g0_:g1_] = False          # mask the grasp segment (+-0.5 s)
                elif mx >= GRASP_OK_MM: out["grasp_warn"] += 1; out["grasp_w"][g0_:g1_] = np.minimum(out["grasp_w"][g0_:g1_], GRASP_W)
            out[f"ikfail_{s}"] = out[f"ikmm_{s}"] > IK_FAIL_MM
            out["valid"] &= ~out[f"ikfail_{s}"]                                                 # free space: mask only IK / limit failures
        out["gapfail"] = out["gap_m"] < ARM_GAP_M - 0.005; out["valid"] &= ~out["gapfail"]
    rob = np.maximum(out["rob_L"], out["rob_R"])
    bad = (rob >= 2) | long_gap | ((out["jv_L"] < 0.5) & (out["ev_L"] > 0)) | ((out["jv_R"] < 0.5) & (out["ev_R"] > 0))
    med = (rob >= 1) | (out["tag_L"] < 0.5) | (out["tag_R"] < 0.5)
    out["weight"] = np.where(bad, 0.1, np.where(med, 0.5, 1.0)).astype(np.float32)
    out["cam_dir"] = {"global": take, "left_wrist": take / asg["cam_of_arm"]["left"], "right_wrist": take / asg["cam_of_arm"]["right"]}
    return out


JAW_CALIB = EGO / "configs/calibration/jaw_kpt_calib_v002.json"
JAW_THRESH_EXPECTED = {"left": 191.0, "right": 150.0}     # coordinator 10-22: fixed v002 calibration


def check_jaw_calib():
    c = json.loads(JAW_CALIB.read_text()); got = {h: float(c[h]["threshold_px"]) for h in JAW_THRESH_EXPECTED}
    if got != JAW_THRESH_EXPECTED: sys.exit(f"jaw calibration {JAW_CALIB.name} thresholds {got} != expected {JAW_THRESH_EXPECTED}")
    return dict(file=str(JAW_CALIB), threshold_px=got)


def load_img(b, c, i):
    p = b["cam_dir"][c] / (f"color_{i:05d}.jpg" if c == "global" else f"wrist_{i:05d}.jpg")
    im = cv2.imread(str(p))
    for d in (1, -1, 2, -2, 3, -3):          # 10-27: a dropped wrist jpg -> nearest existing frame within 0.1 s (30 Hz)
        if im is not None: break
        q = p.with_name(p.name[:-9] + f"{i + d:05d}.jpg"); im = cv2.imread(str(q)) if q.exists() else None
    if im is None: raise IOError(f"unreadable {p}")
    return cv2.cvtColor(cv2.resize(im, IMG_WH, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


SPLIT_FRAC = (0.85, 0.08, 0.07)     # user 10-22 night: ~85/8/7 by episode, stratified across takes


def split_assign(keys, seed, frac=SPLIT_FRAC):
    """episode-level, seeded, STRATIFIED across takes: shuffle within each take, interleave takes round-robin (largest first), give the
    first n_val to val, the next n_test to test, the rest to train -> val/test draw from as many takes as possible"""
    keys = sorted(keys); n = len(keys); rng = np.random.default_rng(seed); by = {}
    for k in keys: by.setdefault(k.split("/")[0], []).append(k)
    for t in by: by[t] = [by[t][i] for i in rng.permutation(len(by[t]))]
    order, takes = [], sorted(by, key=lambda t: (-len(by[t]), t))
    while any(by.values()):
        for t in takes:
            if by[t]: order.append(by[t].pop(0))
    n_val = max(1, int(round(frac[1] * n))) if n >= 3 else 0; n_test = max(1, int(round(frac[2] * n))) if n >= 3 else 0
    return {k: ("val" if r < n_val else "test" if r < n_val + n_test else "train") for r, k in enumerate(order)}


def main():
    global TASK, TAG_GAP_MASK_S, MIN_GOOD_FRAC, POSE_SRC, ROBOTIZE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--takes", nargs="+", required=True)
    ap.add_argument("--derived", default=str(HERE.parent / "data/tag_rig_derived"), help="arm-labelled + robotized trajectories (prep stage)")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--clean-min-tag", type=float, default=0.0)
    ap.add_argument("--max-demos", type=int, default=0, help="per take, for quick tests"); ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--prep-only", action="store_true"); ap.add_argument("--force-prep", action="store_true")
    ap.add_argument("--task", default=TASK, help="task / instruction string stored per frame (DiT text conditioning)")
    ap.add_argument("--jaw-newer-than", default="", help="YYYY-mm-dd HH:MM: reject episodes whose umi_jaw.npz is older (stale labels)")
    ap.add_argument("--qa-only", default="", help="write only the pass-1 QA table to this CSV and exit")
    ap.add_argument("--pose-src", default="", choices=["", "_fused"], help="'' = tag-only umi_traj_<rig>.npz (smoke/audit); _fused = MASt3R+tag fusion (FINAL)")
    ap.add_argument("--robotize", default="ik_v1", choices=["ik_v1", "sep"]); ap.add_argument("--split-file", default="", help="frozen split json (meta/split.json of an earlier build) -> identical episode lists")
    ap.add_argument("--tag-gap-s", type=float, default=TAG_GAP_MASK_S); ap.add_argument("--min-good-frac", type=float, default=MIN_GOOD_FRAC)
    a = ap.parse_args()
    POSE_SRC = a.pose_src; ROBOTIZE = a.robotize
    TAG_GAP_MASK_S, MIN_GOOD_FRAC = a.tag_gap_s, a.min_good_frac
    TASK = a.task; META["instruction"] = TASK
    jaw_cal = check_jaw_calib()
    jaw_min = time.mktime(time.strptime(a.jaw_newer_than, "%Y-%m-%d %H:%M")) if a.jaw_newer_than else 0.0
    takes = [pathlib.Path(t).expanduser().resolve() for t in a.takes]; derived = pathlib.Path(a.derived).expanduser().resolve()
    asgs = {}
    for t in takes:
        try:
            asgs[t.name] = prep_take(t, derived, a.force_prep)
            g = asgs[t.name]; print(f"[prep] {t.name}: rig_of_arm {g['rig_of_arm']} cam_of_arm {g['cam_of_arm']} rig_swapped={g['rig_labels_swapped']} cam_swapped={g['cam_labels_swapped']} corr {g['flow_corr']}", flush=True)
        except Exception as ex:
            print(f"[prep] {t.name}: FAILED {type(ex).__name__}: {ex}", flush=True)
    if a.prep_only: return
    cands = [(t, d) for t in takes for d in demos_of(t)[: (a.max_demos or None)]]
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    base = pathlib.Path(a.out).expanduser(); roots = {s: base.parent / f"{base.name}_{s}" for s in ("train", "val", "test")}
    for r in roots.values():
        if r.exists() and not a.qa_only:
            if not a.overwrite: sys.exit(f"{r} exists (pass --overwrite)")
            shutil.rmtree(r)
    feats = {f"observation.images.{c}": dict(dtype="video", shape=(IMG_WH[1], IMG_WH[0], 3), names=["height", "width", "channels"]) for c in CAMS}
    feats.update({"observation.state": dict(dtype="float32", shape=(PROPRIO_DIM,), names=list(PROPRIO_NAMES)),
                  "action": dict(dtype="float32", shape=(20,), names=ABS_NAMES), "pose_abs": dict(dtype="float32", shape=(20,), names=ABS_NAMES),
                  "quality": dict(dtype="float32", shape=(len(QUALITY_NAMES),), names=QUALITY_NAMES), "weight": dict(dtype="float32", shape=(1,), names=["weight"]), "weight_frame": dict(dtype="float32", shape=(1,), names=["weight_frame"]),
                  "gripper_cont": dict(dtype="float32", shape=(2,), names=["g_cont_left", "g_cont_right"]),
                  "src_index": dict(dtype="int64", shape=(3,), names=list(CAMS)), "img_residual_ms": dict(dtype="float32", shape=(3,), names=list(CAMS))})
    dss, rows, built = {}, [], {}
    for t, d in cands:                                          # ---- pass 1: grid + QA (no frames written)
        key = f"{t.name}/demo{d['idx']:02d}"
        row = dict(episode_key=key, take=t.name, demo=d["idx"], split="", ended_early=d["ended_early"], status="", reject_reason="")
        try:
            if t.name not in asgs: raise ValueError("take prep failed")
            asg = asgs[t.name]; row.update(rig_of_left_arm=asg["rig_of_arm"]["left"], cam_of_left_arm=asg["cam_of_arm"]["left"])
            for cd in set(asg["cam_of_arm"].values()):
                jp = t / cd / "umi_jaw.npz"
                if jp.exists() and jp.stat().st_mtime < jaw_min: raise ValueError(f"stale {cd}/umi_jaw.npz (older than --jaw-newer-than)")
            b = build_demo(t, derived / (t.name + POSE_SRC), asg, d); segs = valid_segments(b["valid"], int(MIN_SEG_S / ACTION_DT_S)); w = b["valid"]
            pct = lambda x: round(float(np.mean(x[w])) * 100, 1) if w.any() else 0.0
            w = b["valid_pre_mask"]
            row.update(duration_s=round(b["window_s"], 2), grid_frames=len(b["t"]), valid_frames=int(w.sum()), n_segments=len(segs),
                       frames_written=int(sum(y - x for x, y in segs)),
                       tag_seen_L_pct=pct(b["tag_L"]), tag_seen_R_pct=pct(b["tag_R"]), jaw_valid_L_pct=pct(b["jv_L"]), jaw_valid_R_pct=pct(b["jv_R"]),
                       robotize2_L_pct=pct(b["rob_L"] >= 2), robotize2_R_pct=pct(b["rob_R"] >= 2), robotize1_L_pct=pct(b["rob_L"] == 1), robotize1_R_pct=pct(b["rob_R"] == 1),
                       open_frac_L=pct(b["g_left"] > 0.5), open_frac_R=pct(b["g_right"] > 0.5),
                       mean_weight=round(float(b["weight"][w].mean()), 3) if w.any() else 0.0, img_res_max_ms=round(float(max(b[f"res_{c}"][w].max() for c in CAMS)), 1) if w.any() else None)
            good = b["valid"]; flags = []
            row.update(idle_head_trim_s=round(b["idle_head_s"], 1), idle_tail_trim_s=round(b["idle_tail_s"], 1))
            act_frac = 1 - (b["idle_head_s"] + b["idle_tail_s"]) / max(b["window_s"], 1e-6)
            good_in_active = good.sum() / max(act_frac * len(good), 1)
            row.update(good_frac=round(float(good.mean()), 3), long_tag_gap_pct=round(float(b["long_gap"].mean()) * 100, 1),
                       close_events_L=b["close_events_L"], close_events_R=b["close_events_R"], ymed_L=round(b["ymed_L"], 3), ymed_R=round(b["ymed_R"], 3))
            if b["ymed_L"] <= 0 or b["ymed_R"] >= 0: flags.append(f"LR_SWAP? left arm median y {b['ymed_L']:+.3f}, right {b['ymed_R']:+.3f}")
            for s_ in ("L", "R"):
                if b[f"close_events_{s_}"] == 0: flags.append(f"no open->close event {s_}")
                if b[f"flicker_{s_}"]: flags.append(f"gripper flicker {s_} (> {FLICKER_TOGGLES_PER_S} toggles / 1 s)")
            if row["long_tag_gap_pct"] > 0: flags.append(f"tag gaps > 1 s on {row['long_tag_gap_pct']} % of frames -> masked")
            row["qa_flags"] = "; ".join(flags)
            segs = valid_segments(b["valid"], int(MIN_SEG_S / ACTION_DT_S)); good = b["valid"]       # after the grasp / IK / gap masks
            if not segs: raise ValueError(f"no valid segment >= {MIN_SEG_S:.0f} s")
            row["good_frac"] = round(float(good_in_active), 3)       # of the ACTIVE (non-idle) part
            if good_in_active < MIN_GOOD_FRAC: raise ValueError(f"only {good_in_active * 100:.0f} % of the active frames valid and outside > 1 s tag gaps (< {MIN_GOOD_FRAC * 100:.0f} %)")
            fm = {f"{k}_{s_}": b[f"{k}_{s_}"] for s_ in ("L", "R") for k in ("cov", "maxgap", "jumps", "stitch_jumps", "jump_max_mm", "disp_mean", "disp_max", "disp_contact_max")}
            row.update({k: round(float(v), 3) for k, v in fm.items()})
            jumps = fm["jumps_L"] + fm["jumps_R"]; cmax = max(fm["disp_contact_max_L"], fm["disp_contact_max_R"]); big = max(fm["jump_max_mm_L"], fm["jump_max_mm_R"]) > JUMP_BIG_MM
            if ROBOTIZE == "ik_v1":
                cmax = 0.0                                        # contact handled by the per-grasp gate below
                row.update(grasps=b["grasp_total"], grasp_warn=b["grasp_warn"], grasp_fail=b["grasp_fail"], grasp_max_mm=" ".join(map(str, b["grasp_max_mm"])),
                           min_link_gap_mm=round(float(b["gap_m"].min()) * 1000, 1), ik_fail_L_pct=round(float(b["ikfail_L"].mean()) * 100, 1),
                           ik_fail_R_pct=round(float(b["ikfail_R"].mean()) * 100, 1), lim_L_pct=round(float(b["lim_L"].mean()) * 100, 1), lim_R_pct=round(float(b["lim_R"].mean()) * 100, 1),
                           free_disp_p50_mm=round(float(np.median(np.r_[b["disp_L"][~b["grasp_L"].astype(bool)], b["disp_R"][~b["grasp_R"].astype(bool)]])), 1))
                if b["grasp_total"] and b["grasp_fail"] > b["grasp_total"] / 2:
                    row.update(fused_decision="EXCLUDE", episode_weight=0.0); row["qa_flags"] = "; ".join(flags)
                    raise ValueError(f"grasp gate EXCLUDE: {b['grasp_fail']}/{b['grasp_total']} grasps > {GRASP_WARN_MM:.0f} mm")
                if b["grasp_fail"]: flags.append(f"{b['grasp_fail']} grasp(s) > {GRASP_WARN_MM:.0f} mm masked")
                if b["grasp_warn"]: flags.append(f"{b['grasp_warn']} grasp(s) 20-50 mm -> weight {GRASP_W}")
            why = []
            if min(fm["cov_L"], fm["cov_R"]) < COVERAGE_WARN: flags.append(f"fused coverage {min(fm['cov_L'], fm['cov_R']) * 100:.0f} % < 95 %")
            if max(fm["maxgap_L"], fm["maxgap_R"]) >= GAP_WARN_S: flags.append(f"longest gap {max(fm['maxgap_L'], fm['maxgap_R']):.1f} s")
            if jumps >= JUMPS_EXCL or big or cmax >= CONTACT_EXCL_MM:
                why = [f"jumps {jumps}" if jumps >= JUMPS_EXCL else "", "jump > 100 mm" if big else "", f"contact robotize {cmax:.0f} mm" if cmax >= CONTACT_EXCL_MM else ""]
                row.update(fused_decision="EXCLUDE", episode_weight=0.0); row["qa_flags"] = "; ".join(flags)
                raise ValueError("fused-quality EXCLUDE: " + ", ".join(w_ for w_ in why if w_))
            if jumps > 0 or cmax >= CONTACT_DOWN_MM:
                row.update(fused_decision="DOWNWEIGHT", episode_weight=DOWN_W); flags.append(f"downweight {DOWN_W}: jumps {jumps}, contact robotize max {cmax:.0f} mm")
            else:
                row.update(fused_decision="TRAIN", episode_weight=1.0)
            row["qa_flags"] = "; ".join(flags); b["episode_weight"] = row["episode_weight"]
            if a.clean_min_tag and min(row["tag_seen_L_pct"], row["tag_seen_R_pct"]) < a.clean_min_tag * 100:
                raise ValueError(f"not clean: tag_seen L/R {row['tag_seen_L_pct']}/{row['tag_seen_R_pct']} % < {a.clean_min_tag * 100:.0f} %")
            row["status"] = "eligible"; built[key] = (t, b, segs)
        except Exception as ex:
            row.update(status="rejected", reject_reason=f"{type(ex).__name__}: {ex}", qa_decision="dropped")
        rows.append(row)
    if a.qa_only:
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(a.qa_only, "w", newline="") as fh: wr = csv.DictWriter(fh, fieldnames=cols); wr.writeheader(); wr.writerows(rows)
        print("qa-only:", sum(r["status"] == "eligible" for r in rows), "eligible of", len(rows)); return
    # ---- split over the ELIGIBLE episodes (80/10/10, seeded) or a frozen split file
    elig = [r["episode_key"] for r in rows if r["status"] == "eligible"]
    if a.split_file:
        fz = json.loads(pathlib.Path(a.split_file).read_text())["episodes"]; split = {k: s for s, ks in fz.items() for k in ks}
        missing = [k for k in split if k not in elig]
        if missing: print("WARNING frozen-split episodes not eligible now:", missing, flush=True)
    else:
        split = split_assign(elig, a.seed)
    for row in rows:
        key = row["episode_key"]; sp = split.get(key, ""); row["split"] = sp
        if row["status"] != "eligible": pass
        elif not sp: row.update(status="rejected", reject_reason="not in the frozen split", qa_decision="dropped")
        else:
          try:
            t, b, segs = built[key]
            if sp not in dss:
                dss[sp] = LeRobotDataset.create(f"local/{base.name}_{sp}", fps=int(round(1 / ACTION_DT_S)), features=feats, root=roots[sp], robot_type="handumi_tagrig_x2",
                                                use_videos=True, image_writer_threads=0, streaming_encoding=True, encoder_threads=2)
            ds = dss[sp]; eps = []
            for x, y in segs:
                TL, TR, gL, gR = b["T_left"][x:y], b["T_right"][x:y], b["g_left"][x:y], b["g_right"][x:y]
                st = proprio_rows_lr(TL, TR, gL, gR); act = abs_action_rows(TL, TR, gL, gR)
                pa = np.concatenate([T_to_pose9(TL), T_to_pose9(TR), gL[:, None], gR[:, None]], -1).astype(np.float32)
                q = np.stack([b[k][x:y] for k in ("tag_L", "tag_R", "rob_L", "rob_R", "jv_L", "jv_R", "ev_L", "ev_R")], -1).astype(np.float32)
                for j, i in enumerate(range(x, y)):
                    fr = {f"observation.images.{c}": load_img(b, c, int(b[f"img_{c}"][i])) for c in CAMS}
                    fr.update({"observation.state": st[j], "action": act[j], "pose_abs": pa[j], "quality": q[j], "weight": np.array([b["episode_weight"] * b["grasp_w"][i]], np.float32), "weight_frame": b["weight"][i:i + 1], "gripper_cont": np.array([b["gc_left"][i], b["gc_right"][i]], np.float32),
                               "src_index": np.array([b[f"img_{c}"][i] for c in CAMS], np.int64),
                               "img_residual_ms": np.array([b[f"res_{c}"][i] for c in CAMS], np.float32), "task": TASK})
                    ds.add_frame(fr)
                eps.append(ds.meta.total_episodes); ds.save_episode()
            row.update(status="converted", lerobot_episodes=" ".join(map(str, eps)),
                       qa_decision=row["fused_decision"])
          except Exception as ex:
            row.update(status="rejected", reject_reason=f"write {type(ex).__name__}: {ex}", qa_decision="dropped")
        print(f"{key} [{row['split']}] {row['status']} {row['reject_reason']} frames={row.get('frames_written')} tag L/R {row.get('tag_seen_L_pct')}/{row.get('tag_seen_R_pct')} "
              f"jaw L/R {row.get('jaw_valid_L_pct')}/{row.get('jaw_valid_R_pct')} open L/R {row.get('open_frac_L')}/{row.get('open_frac_R')} w {row.get('mean_weight')} | {row.get('qa_flags', '')}", flush=True)
    for d in dss.values(): d.finalize()
    cols = list(dict.fromkeys(k for r in rows for k in r))
    split_doc = dict(method=("FROZEN " + a.split_file) if a.split_file else "episode-level, seeded, stratified across takes over the QA-eligible demos (take/demoNN), 85/8/7",
                     seed=a.seed, clean_min_tag=a.clean_min_tag,
                     episodes={s: sorted(k for k, v in split.items() if v == s) for s in ("train", "val", "test")})
    meta = dict(META, pose_src=POSE_SRC or "tag_only", robotize=ROBOTIZE, grasp_gate=dict(ok_mm=GRASP_OK_MM, warn_mm=GRASP_WARN_MM, warn_weight=GRASP_W,
                rule="per grasp window (+-0.5 s around each jaw close): max |robotized - fused| < 20 ok, 20-50 weight 0.5, > 50 masked; > half of grasps masked -> drop; free space: mask only IK error > 10 mm; link gap < 0.115 m masked"), converter="tag_rig_to_lerobot.py", contract_version=__import__("contracts").CONTRACT_VERSION, takes=[t.name for t in takes],
                assignments=asgs, jaw_calibration=jaw_cal, quality_names=QUALITY_NAMES, img_tolerance_s=IMG_TOL_S, max_src_gap_s=MAX_SRC_GAP_S, tag_gap_mask_s=TAG_GAP_MASK_S, min_good_frac=MIN_GOOD_FRAC)
    for s, r in roots.items():
        r.mkdir(parents=True, exist_ok=True); (r / "meta").mkdir(exist_ok=True)
        with open(r / "episode_table.csv", "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=cols); wr.writeheader(); wr.writerows(rows)     # full table (all splits) in every root
        (r / "meta" / "split.json").write_text(json.dumps(dict(split_doc, this_split=s), indent=1))
        (r / "source_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
        shutil.copy(r / "episode_table.csv", r / "conversion_manifest.csv")              # name used by training/smoke_train.sh
    print("done", sum(r["status"] == "converted" for r in rows), "converted,", sum(r["status"] == "rejected" for r in rows), "rejected;",
          {s: sum(1 for r in rows if r["status"] == "converted" and r["split"] == s) for s in roots})


if __name__ == "__main__":
    main()
