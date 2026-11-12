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
import sys

from ..core.errors import DeployError
from .bridge import goReady
from .camera_commands import camsCmd, waveCmd
from .keys import KeysCtl
from .rig import Rig
from .run_loop import checkRunArgs, runSession
from .start_pose import startCmd

READY_CONFIRM_TIMEOUT_S = 30
READY_BRIDGE_TIMEOUT_S = 0.5

# 10-23 user: DiT plans disagree by ~16 mm (k8, p50) from plan to plan -> back-and-forth; the ensemble now applies to DiT too


def runCmd(args):
    checkRunArgs(args)
    tRig = Rig(args).start()
    tCtl = KeysCtl()
    try:
        tSummary = runSession(args, tRig, tCtl)
        if args.go and not getattr(args, "no_prompt", False):
            print("다음 동작은 사람만: r = READY 로 천천히 복귀 (확인 필요), q = 종료 (자세 유지)", flush=True)
            tKey = tCtl.mKeys.wait("rq")
            if tKey == "r":
                print("정말 READY 로 복귀? 경로 주변을 비우고 y (다른 키 = 취소)", flush=True)
                if tCtl.mKeys.wait("yn", timeout=READY_CONFIRM_TIMEOUT_S) == "y":
                    from laundry_shared.visual_teleop.robot.bridge import HttpBridge

                    tBridge = HttpBridge(tRig.mUrl, live=True, timeout_s=READY_BRIDGE_TIMEOUT_S)
                    try:
                        goReady(tBridge, tRig.mArms, tRig.mQReady)
                    finally:
                        tBridge.close()
    finally:
        tRig.close()
    return tSummary


def buildParser() -> argparse.ArgumentParser:
    """The deploy CLI (also the single source of the run defaults for the web UI and run_n_steps)."""
    tParser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tSub = tParser.add_subparsers(dest="cmd", required=True)
    tCams = tSub.add_parser("cams", help="USB location -> OpenCV index (brightness probe), Desktop snapshot")
    tCams.add_argument("--cover", action="store_true")
    tCams.add_argument("--candidates", default="0,2,3,4,5")
    tWave = tSub.add_parser(
        "wave", help="side-by-side wrist streams labelled by USB location; --confirm left=LOC right=LOC pins the mapping"
    )
    tWave.add_argument("--port", type=int, default=8761)
    tWave.add_argument("--secs", type=float, default=0)
    tWave.add_argument("--exposure", type=int, default=2)
    tWave.add_argument("--confirm", nargs=2, metavar="SIDE=LOC")
    tWave.add_argument("--by", default="user")
    tStart = tSub.add_parser(
        "start",
        help="plan (dry) / move (--go) READY -> a TRAINING episode start pose (READY is outside the training TCP distribution)",
    )
    tStart.add_argument("--go", action="store_true")
    tStart.add_argument("--episode", type=int, default=-1, help="train episode index (default: medoid start)")
    tStart.add_argument("--start-lift", type=float, default=0.03, help="m above the training start TCP (table safety)")
    tStart.add_argument(
        "--height-calib", action="store_true", help="11-01: shift the start TCP by the per-arm xy of deploy/height_calib.json"
    )
    tStart.add_argument("--deg-s", type=float, default=10.0)
    tStart.add_argument(
        "--root",
        default="",
        help="10-28: training root under data/lerobot for the start pose (default tagrig_laundry_v1_fused_train)",
    )
    tStart.add_argument("--assume-ready", action="store_true")
    tStart.add_argument(
        "--label", default="", help="10-27: start at this label's first pose (ik_v1/demoNN.npz) instead of a training episode"
    )
    tRun = tSub.add_parser("run")
    tRun.add_argument("--policy", choices=["act", "dit", "label"], default="act")
    tRun.add_argument("--label", default="", help="--policy label: robotize_ik ik_v1/demoNN.npz")
    tRun.add_argument("--label-rate", type=float, default=0.2, help="--policy label: label time per wall second")
    tRun.add_argument(
        "--no-safety",
        action="store_true",
        help="10-27 user: collision/table/joint-limit/workspace STOPs only logged (label replay)",
    )
    tRun.add_argument(
        "--no-prompt", action="store_true", help="10-27: no r/q prompt after --go (driven from a script; stop = bridge /estop)"
    )
    tRun.add_argument("--ckpt", default=None, help="default: ACT 500k / DiT checkpoints/last")
    tRun.add_argument("--go", action="store_true", help="명령을 보냄 (사람이 옆에 있을 때만)")
    tRun.add_argument("--secs", type=float, default=60.0)
    tRun.add_argument("--commit", type=int, default=None, help="ACT 1 (1/2/4), DiT 4 (2/4)")
    tRun.add_argument("--speed", type=float, default=0.5)
    tRun.add_argument("--slow-start-s", type=float, default=2.0)
    tRun.add_argument("--slow-start-speed", type=float, default=0.25)
    tRun.add_argument("--max-tcp-mps", type=float, default=0.25)
    tRun.add_argument("--max-rot-dps", type=float, default=90.0)
    tRun.add_argument("--max-deg-s", type=float, default=30.0)
    tRun.add_argument("--max-ik-mm", type=float, default=10.0)
    tRun.add_argument("--max-ik-deg", type=float, default=20.0)
    tRun.add_argument("--ik-hold-mm", type=float, default=15.0,
                      help="11-07: IK pinned at a joint limit and farther than this from the target -> that arm holds (mm)")
    tRun.add_argument("--ik-hold-deg", type=float, default=30.0, help="11-07: the same for the orientation error (deg)")
    tRun.add_argument("--ws-margin", type=float, default=0.05)
    tRun.add_argument("--arm-clear", type=float, default=0.010, help="capsule link-link clearance (m) -> STOP")
    tRun.add_argument("--warn-gap", type=float, default=0.12, help="link-origin gap (m) -> WARN + speed x0.5")
    tRun.add_argument("--preview-hz", type=float, default=5.0)
    tRun.add_argument("--preview-stride", type=int, default=4)
    tRun.add_argument("--preview-hold", action="store_true", help="future collision/table in the preview -> HOLD")
    tRun.add_argument("--grip-step", type=int, default=1, help="chunk row (1..32) whose gripper decides open/close")
    tRun.add_argument("--init-grip", type=float, default=0.0, help="state gripper at start (1 open, 0 closed, -1 = observed)")
    tRun.add_argument(
        "--temporal-ensemble",
        type=float,
        default=0.01,
        help="temporal ensembling of overlapping plans (ACT and DiT; 0 = off; 0.01 = ACT paper)",
    )
    tRun.add_argument("--dit-samples", type=int, default=1, help="DiT: mean of K batched DDIM samples (1 = offline-eval sampler)")
    tRun.add_argument(
        "--rtc",
        default="off",
        choices=["off", "guided", "blend"],
        help="10-29 Real-Time Chunking (look-ahead GO only): guided = RTC guidance (DiT), blend = gradient-free (X-VLA always)",
    )
    tRun.add_argument(
        "--no-lookahead", action="store_true", help="10-30: GO without look-ahead (run the whole commit, then observe + infer)"
    )
    tRun.add_argument(
        "--z-offset-mm",
        type=float,
        default=0.0,
        help="10-30: raise (+) / lower (-) the whole commanded trajectory once, at the first chunk",
    )
    tRun.add_argument(
        "--height-calib",
        action="store_true",
        help="11-01: add the per-arm table-height error from deploy/height_calib.json (robot-measured tags)",
    )
    tRun.add_argument("--rtc-beta", type=float, default=5.0, help="RTC guidance weight clip (paper: 5)")
    tRun.add_argument(
        "--dit-fixed-noise", type=int, default=1, help="DiT: 1 = same initial noise every plan (less jitter), 0 = seed per cycle"
    )
    tRun.add_argument(
        "--grip-at", choices=["first", "mid", "last", "max", "row"], default="mid", help="which committed chunk row sets the jaw (row = every row, in time)"
    )
    tRun.add_argument("--grip-thr", type=float, default=0.5, help="jaw opens above this")
    tRun.add_argument("--latency-match", action="store_true", help="send-ahead + capture-time state from latency.json (UMI)")
    tRun.add_argument("--contact-floor", action="store_true", help="raise an arm's floor where it pressed on the table (torque)")
    tRun.add_argument("--contact-gap-mm", type=float, default=8.0, help="contact: measured TCP this far above the commanded one")
    tRun.add_argument("--contact-tau-nm", type=float, default=2.0, help="contact: pitch-joint torque this far off its free baseline")
    tRun.add_argument("--grip-continuous", action="store_true", help="10-30: jaw opening = predicted P(open) 0..1 (no threshold)")
    tRun.add_argument(
        "--grip-mode",
        default="",
        choices=["", "sigmoid", "relu"],
        help="10-30: sigmoid = 1/(1+exp(-gain (p-0.5))); relu = clip(gain (p-0.5), 0, 1)",
    )
    tRun.add_argument("--grip-gain", type=float, default=10.0, help="sigmoid gain (larger = closer to the 0/1 threshold)")
    tRun.add_argument(
        "--keep-going",
        type=int,
        default=1,
        help="--go: 1 = safety hits skip that plan (hold) instead of ending the run; drive dropouts are recovered (re-enable, clear latch, "
        "re-seed) and the run continues. Only Space / q / UI stop / heartbeat end it",
    )
    tRun.add_argument(
        "--max-ik-stop-mm",
        type=float,
        default=80.0,
        help="IK error above --max-ik-mm only limits (closest reachable pose); above this it STOPs",
    )
    # 10-23 user: -0.01 scraped too much
    tRun.add_argument(
        "--tip-clear",
        type=float,
        default=0.0,
        help="finger-corner floor vs the table (m); < 0 = fingers may press into / scrape the table",
    )
    tRun.add_argument("--tcp-table-margin", type=float, default=-0.005, help="hard STOP if the TCP goes below table_z + this (m)")
    tRun.add_argument(
        "--resync-deg",
        type=float,
        default=6.0,
        help="--go: measured vs commanded joint gap that triggers a re-seed + hold (motor dropout)",
    )
    tRun.add_argument("--max-resync", type=int, default=5, help="--go: stop after this many re-syncs in one run")
    tRun.add_argument("--stream-hz", type=float, default=30.0)
    tRun.add_argument("--exposure", type=int, default=2)
    tRun.add_argument(
        "--wrist-align",
        action="store_true",
        help="apply act_tagrig_cams.json wrist_align {left: {...}, right: {...}} (copy from wrist_align_proposed, act_tagrig_domain_gap.py)",
    )
    tRun.add_argument("--global-url", default="http://127.0.0.1:8092", help="orbbec_server (live Orbbec, sudo)")
    tRun.add_argument(
        "--replay-global",
        default="",
        metavar="TAKE_DIR",
        help="TEST ONLY: recorded color_*.jpg instead of the live Orbbec (refused with --go)",
    )
    tRun.add_argument("--assume-ready", action="store_true", help="dry-run without a live robot: state = FK(READY)")
    tRun.add_argument("--assume-start", action="store_true", help="dry-run without a live robot: state = FK(start_pose.json)")
    tRun.add_argument(
        "--allow-start-pose",
        action="store_true",
        help="--go may start from start_pose.json (after `start --go`) instead of READY",
    )
    tRun.add_argument("--ready-tol-deg", type=float, default=5.0)
    tRun.add_argument("--dit-steps", type=int, default=10, help="DiT DDIM steps")
    tRun.add_argument(
        "--bridge-url",
        default=None,
        help="default: visual_teleop config robot.base_url (:8021); mock bridge e.g. http://127.0.0.1:8031",
    )
    tRun.add_argument("--max-plan-age-s", type=float, default=0.4, help="observation -> commit older than this -> HOLD")
    tRun.add_argument("--max-img-age-ms", type=float, default=250.0)
    tRun.add_argument("--print-every", type=int, default=10)
    tRun.add_argument("--device", default="mps")
    tRun.add_argument("--no-video", action="store_true")
    return tParser


COMMANDS = {"cams": camsCmd, "wave": waveCmd, "run": runCmd, "start": startCmd}


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    try:
        COMMANDS[tArgs.cmd](tArgs)
    except DeployError as tExc:
        print(tExc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
