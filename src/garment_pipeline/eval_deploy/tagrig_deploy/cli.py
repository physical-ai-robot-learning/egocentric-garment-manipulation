"""[2026-10-23] Closed-loop real-robot deploy of the tag-rig laundry policies on the bimanual reBot B601-DM:
  --policy act  act_tagrig_v1_1m 500k (default)          --policy dit  dit_tagrig_v1 (checkpoints/last; Multi-Task DiT, DDIM 10)
DRY-RUN by default (reads cameras + robot, runs everything, sends NOTHING).  --go sends; only with a person at the robot.

Receding horizon (user review 10-23): every plan = fresh images + measured q -> FK(q) = T(t_obs) -> chunk (32 x 20 Hz) -> COMMIT
only --commit steps (ACT default 1, DiT default 4) -> re-plan.  Inference runs in this planner loop while the 30 Hz Streamer thread
keeps streaming the last accepted segment (it never waits for the policy).  Latency re-anchoring: when the chunk is ready (t_ready),
the latest measured TCP T_now is read, k0 = floor((t_ready - t_obs) / 50 ms) steps of the chunk are already "in the past", and the
committed targets are T_now @ inv(dT_k0) @ dT_k, k = k0+1 .. k0+commit (dT_0 = I): the plan is always anchored on the newest TCP.
Stages per committed step and arm, each logged: candidate -> limiter -> IK -> safety -> accepted -> commit (30 Hz interpolated stream).
  limiter   [temporal ensemble, ACT] -> speed scale s (--slow-start-speed for --slow-start-s, then --speed) as a fraction of the
            step from T_now -> Cartesian clamp (--max-tcp-mps, --max-rot-dps) -> finger-corner table floor (robotize_ik corner_lift)
  IK        robotize_ik.Arm.free = umi_replay_bi Pink settings (bridge joint limits 0.5 deg inside, ori cost 0.1, 7.5 deg/step bound),
            seeded with the last accepted q; then the joint-speed limiter scales dq to <= --max-deg-s
  safety    (hard: STOP + HOLD) NaN, workspace box (training pose_abs range +- --ws-margin, per arm), IK error > 10 mm / 20 deg,
            joint limits, capsule link-link collision between the arms (clearance < --arm-clear; core.Collision, capsules because
            the URDF link meshes are not on this Mac), link capsules under the table, TCP below table + 5 mm, bridge reject /
            E-STOP / stale read / stale image.  Warning only: link-origin gap < 120 mm -> logged + speed x0.5 on the next plan.
STOP = br.estop() = the MIT bridge holds the measured pose of both arms.  Nothing moves afterwards, ever, except: key r + confirm y
(--go only) -> /clear_estop + slow joint ramp to READY (umi_replay_bi.ramp_both).  Keys (this terminal): SPACE = stop + hold,
q = quit (stop + hold first); after a stop: r = return to READY (asks y/n).  Gripper: chunk row --grip-step, > 0.5 open (g 1) else closed.
Dry-run: same pipeline, nothing sent, violations are logged and the run continues ("would STOP"); state gripper = the virtual command.
Global camera = laundry_system/robot/orbbec_server.py (sudo, OrbbecCapture 848x480 = the collector's settings) at :8092.
Wrist cameras = robot Arducam fisheyes by USB location (deploy/act_tagrig_cams.json; `cams` subcommand), fixed exposure + IMAGE_CTRL
of umi_jaw record.  Output: reports/act_deploy/<run>/{log.npz, summary.json}, Desktop mp4 (Orbbec projection + top-down + wrists).
    sudo ~/ego_collector/.venv/bin/python ~/laundry_system/robot/orbbec_server.py          # terminal 1 (global camera)
    .venv-mac/bin/python deploy/act_tagrig_deploy.py cams [--cover]                         # once after any USB replug
    .venv-mac/bin/python deploy/act_tagrig_deploy.py run [--policy dit] --secs 30           # DRY-RUN (robot at READY)
    .venv-mac/bin/python deploy/act_tagrig_deploy.py run [--policy dit] --go --secs 60      # sends commands (person at the robot)"""
from __future__ import annotations
import argparse

from garment_pipeline.eval_deploy.tagrig_deploy.bridge import go_ready
from garment_pipeline.eval_deploy.tagrig_deploy.cameras import cams_cmd, wave_cmd
from garment_pipeline.eval_deploy.tagrig_deploy.keys import KeysCtl
from garment_pipeline.eval_deploy.tagrig_deploy.rig import Rig
from garment_pipeline.eval_deploy.tagrig_deploy.session import check_run_args, run_session
from garment_pipeline.eval_deploy.tagrig_deploy.start_pose import start_cmd


    # 10-23 user: DiT plans disagree by ~16 mm (k8, p50) from plan to plan -> back-and-forth; the ensemble now applies to DiT too


def run_cmd(a):
    check_run_args(a)
    rig = Rig(a).start(); ctl = KeysCtl()
    try:
        summ = run_session(a, rig, ctl)
        if a.go and not getattr(a, "no_prompt", False):
            print("다음 동작은 사람만: r = READY 로 천천히 복귀 (확인 필요), q = 종료 (자세 유지)", flush=True)
            k = ctl.keys.wait("rq")
            if k == "r":
                print("정말 READY 로 복귀? 경로 주변을 비우고 y (다른 키 = 취소)", flush=True)
                if ctl.keys.wait("yn", timeout=30) == "y":
                    from visual_teleop.robot.bridge import HttpBridge
                    brl = HttpBridge(rig.url, live=True, timeout_s=0.5)
                    try: go_ready(brl, rig.arms, rig.qr)
                    finally: brl.close()
    finally: rig.close()
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    c_ = sub.add_parser("cams", help="USB location -> OpenCV index (brightness probe), Desktop snapshot"); c_.add_argument("--cover", action="store_true"); c_.add_argument("--candidates", default="0,2,3,4,5")
    w_ = sub.add_parser("wave", help="side-by-side wrist streams labelled by USB location; --confirm left=LOC right=LOC pins the mapping")
    w_.add_argument("--port", type=int, default=8761); w_.add_argument("--secs", type=float, default=0); w_.add_argument("--exposure", type=int, default=2)
    w_.add_argument("--confirm", nargs=2, metavar="SIDE=LOC"); w_.add_argument("--by", default="user")
    s_ = sub.add_parser("start", help="plan (dry) / move (--go) READY -> a TRAINING episode start pose (READY is outside the training TCP distribution)")
    s_.add_argument("--go", action="store_true"); s_.add_argument("--episode", type=int, default=-1, help="train episode index (default: medoid start)")
    s_.add_argument("--start-lift", type=float, default=0.03, help="m above the training start TCP (table safety)")
    s_.add_argument("--height-calib", action="store_true", help="11-01: shift the start TCP by the per-arm xy of deploy/height_calib.json"); s_.add_argument("--deg-s", type=float, default=10.0)
    s_.add_argument("--root", default="", help="10-28: training root under data/lerobot for the start pose (default tagrig_laundry_v1_fused_train)"); s_.add_argument("--assume-ready", action="store_true"); s_.add_argument("--label", default="", help="10-27: start at this label's first pose (ik_v1/demoNN.npz) instead of a training episode")
    r = sub.add_parser("run")
    r.add_argument("--policy", choices=["act", "dit", "label"], default="act"); r.add_argument("--label", default="", help="--policy label: robotize_ik ik_v1/demoNN.npz"); r.add_argument("--label-rate", type=float, default=0.2, help="--policy label: label time per wall second"); r.add_argument("--no-safety", action="store_true", help="10-27 user: collision/table/joint-limit/workspace STOPs only logged (label replay)"); r.add_argument("--no-prompt", action="store_true", help="10-27: no r/q prompt after --go (driven from a script; stop = bridge /estop)"); r.add_argument("--ckpt", default=None, help="default: ACT 500k / DiT checkpoints/last")
    r.add_argument("--go", action="store_true", help="명령을 보냄 (사람이 옆에 있을 때만)"); r.add_argument("--secs", type=float, default=60.0)
    r.add_argument("--commit", type=int, default=None, help="ACT 1 (1/2/4), DiT 4 (2/4)"); r.add_argument("--speed", type=float, default=0.5)
    r.add_argument("--slow-start-s", type=float, default=2.0); r.add_argument("--slow-start-speed", type=float, default=0.25)
    r.add_argument("--max-tcp-mps", type=float, default=0.25); r.add_argument("--max-rot-dps", type=float, default=90.0); r.add_argument("--max-deg-s", type=float, default=30.0)
    r.add_argument("--max-ik-mm", type=float, default=10.0); r.add_argument("--max-ik-deg", type=float, default=20.0); r.add_argument("--ws-margin", type=float, default=0.05)
    r.add_argument("--arm-clear", type=float, default=0.010, help="capsule link-link clearance (m) -> STOP"); r.add_argument("--warn-gap", type=float, default=0.12, help="link-origin gap (m) -> WARN + speed x0.5")
    r.add_argument("--preview-hz", type=float, default=5.0); r.add_argument("--preview-stride", type=int, default=4); r.add_argument("--preview-hold", action="store_true", help="future collision/table in the preview -> HOLD")
    r.add_argument("--grip-step", type=int, default=1, help="chunk row (1..32) whose gripper decides open/close"); r.add_argument("--init-grip", type=float, default=0.0, help="state gripper at start (1 open, 0 closed, -1 = observed)")
    r.add_argument("--temporal-ensemble", type=float, default=0.01, help="temporal ensembling of overlapping plans (ACT and DiT; 0 = off; 0.01 = ACT paper)")
    r.add_argument("--dit-samples", type=int, default=1, help="DiT: mean of K batched DDIM samples (1 = offline-eval sampler)")
    r.add_argument("--rtc", default="off", choices=["off", "guided", "blend"], help="10-29 Real-Time Chunking (look-ahead GO only): guided = RTC guidance (DiT), blend = gradient-free (X-VLA always)")
    r.add_argument("--no-lookahead", action="store_true", help="10-30: GO without look-ahead (run the whole commit, then observe + infer)")
    r.add_argument("--z-offset-mm", type=float, default=0.0, help="10-30: raise (+) / lower (-) the whole commanded trajectory once, at the first chunk")
    r.add_argument("--height-calib", action="store_true", help="11-01: add the per-arm table-height error from deploy/height_calib.json (robot-measured tags)")
    r.add_argument("--rtc-beta", type=float, default=5.0, help="RTC guidance weight clip (paper: 5)")
    r.add_argument("--dit-fixed-noise", type=int, default=1, help="DiT: 1 = same initial noise every plan (less jitter), 0 = seed per cycle")
    r.add_argument("--grip-at", choices=["first", "mid", "last", "max"], default="mid", help="which committed chunk row sets the jaw")
    r.add_argument("--grip-thr", type=float, default=0.5, help="jaw opens above this")
    r.add_argument("--grip-continuous", action="store_true", help="10-30: jaw opening = predicted P(open) 0..1 (no threshold)")
    r.add_argument("--grip-mode", default="", choices=["", "sigmoid", "relu"], help="10-30: sigmoid = 1/(1+exp(-gain (p-0.5))); relu = clip(gain (p-0.5), 0, 1)")
    r.add_argument("--grip-gain", type=float, default=10.0, help="sigmoid gain (larger = closer to the 0/1 threshold)")
    r.add_argument("--keep-going", type=int, default=1, help="--go: 1 = safety hits skip that plan (hold) instead of ending the run; drive dropouts are recovered (re-enable, clear latch, re-seed) and the run continues. Only Space / q / UI stop / heartbeat end it")
    r.add_argument("--max-ik-stop-mm", type=float, default=80.0, help="IK error above --max-ik-mm only limits (closest reachable pose); above this it STOPs")
    r.add_argument("--tip-clear", type=float, default=0.0, help="finger-corner floor vs the table (m); < 0 = fingers may press into / scrape the table")   # 10-23 user: -0.01 scraped too much
    r.add_argument("--tcp-table-margin", type=float, default=-0.005, help="hard STOP if the TCP goes below table_z + this (m)")
    r.add_argument("--resync-deg", type=float, default=6.0, help="--go: measured vs commanded joint gap that triggers a re-seed + hold (motor dropout)")
    r.add_argument("--max-resync", type=int, default=5, help="--go: stop after this many re-syncs in one run")
    r.add_argument("--stream-hz", type=float, default=30.0); r.add_argument("--exposure", type=int, default=2); r.add_argument("--wrist-align", action="store_true", help="apply act_tagrig_cams.json wrist_align {left: {...}, right: {...}} (copy from wrist_align_proposed, act_tagrig_domain_gap.py)")
    r.add_argument("--global-url", default="http://127.0.0.1:8092", help="orbbec_server (live Orbbec, sudo)")
    r.add_argument("--replay-global", default="", metavar="TAKE_DIR", help="TEST ONLY: recorded color_*.jpg instead of the live Orbbec (refused with --go)")
    r.add_argument("--assume-ready", action="store_true", help="dry-run without a live robot: state = FK(READY)")
    r.add_argument("--assume-start", action="store_true", help="dry-run without a live robot: state = FK(start_pose.json)")
    r.add_argument("--allow-start-pose", action="store_true", help="--go may start from start_pose.json (after `start --go`) instead of READY")
    r.add_argument("--ready-tol-deg", type=float, default=5.0); r.add_argument("--dit-steps", type=int, default=10, help="DiT DDIM steps")
    r.add_argument("--bridge-url", default=None, help="default: visual_teleop config robot.base_url (:8021); mock bridge e.g. http://127.0.0.1:8031")
    r.add_argument("--max-plan-age-s", type=float, default=0.4, help="observation -> commit older than this -> HOLD"); r.add_argument("--max-img-age-ms", type=float, default=250.0); r.add_argument("--print-every", type=int, default=10)
    r.add_argument("--device", default="mps"); r.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    {"cams": cams_cmd, "wave": wave_cmd, "run": run_cmd, "start": start_cmd}[a.cmd](a)
