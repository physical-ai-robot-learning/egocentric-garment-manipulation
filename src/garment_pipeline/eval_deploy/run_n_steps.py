#!/usr/bin/env python3
"""[2026-10-28] Run exactly N policy steps on the real robot from the current pose (user: "여기서 3스텝만 가보자"), through the SAME
run_session step mode as the UI ([한 스텝 실행]): the policy plans continuously, each step token sends ONE committed segment
(--commit steps), the next token is issued only after the previous one was consumed + --gap seconds; after N executed steps
the run stops (bridge /estop = hold the measured pose).  All safety checks of run_session stay on.
    DEPLOY_FAKE_WRIST=right .venv-mac/bin/python deploy/run_n_steps.py --ckpt <dir> --n 3 [--commit 4 --speed 0.5]"""
import argparse, pathlib, sys, threading, time

HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import act_tagrig_deploy as dep  # noqa: E402
from act_tagrig_ui import dep_parser  # noqa: E402


class Ctl:
    def __init__(self, n, gap):
        self.n, self.gap = n, gap; self.lock = threading.Lock(); self.step_n = 0; self.steps_done = 0; self.stop_req = None; self.snap = None
        self.step_mode = True; self.t_last = None

    def take_step(self):
        with self.lock:
            if self.step_n <= 0: return False
            self.step_n -= 1; self.steps_done += 1; self.t_last = time.monotonic(); print(f"[n_steps] 스텝 {self.steps_done}/{self.n} 실행", flush=True); return True

    def poll(self):
        with self.lock:
            if self.stop_req: return self.stop_req
            if self.steps_done >= self.n and self.step_n == 0:
                if time.monotonic() - self.t_last > self.gap: self.stop_req = f"{self.n} 스텝 완료"
            elif self.step_n == 0 and (self.t_last is None or time.monotonic() - self.t_last > self.gap): self.step_n = 1
        return self.stop_req

    def publish(self, snap): self.snap = snap


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--policy", default="dit"); ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--commit", type=int, default=4); ap.add_argument("--speed", type=float, default=0.5); ap.add_argument("--gap", type=float, default=2.0)
    ap.add_argument("--dit-steps", type=int, default=4); ap.add_argument("--max-img-age-ms", type=float, default=600.0); ap.add_argument("--max-plan-age-s", type=float, default=1.0); ap.add_argument("--warmup", type=float, default=4.0, help="s of planning (preview only) before step 1")
    a = ap.parse_args()
    ra = dep_parser().parse_args(["run"]); ra.policy = a.policy; ra.ckpt = a.ckpt; ra.commit = a.commit; ra.speed = a.speed; ra.slow_start_speed = a.speed
    ra.secs = 0.0; ra.dit_steps = a.dit_steps; ra.go = True; ra.step_mode = True; ra.allow_start_pose = True
    ra.max_img_age_ms = a.max_img_age_ms; ra.max_plan_age_s = a.max_plan_age_s     # step mode: the robot is still while waiting
    rig = dep.Rig(ra).start(); ctl = Ctl(a.n, a.gap); ctl.t_last = time.monotonic() + a.warmup - a.gap      # first step after the warm-up
    S = dep.run_session(ra, rig, ctl)
    print("[n_steps] 끝:", (S or {}).get("stop_reason"), "| run", (S or {}).get("run"), flush=True); rig.close()


if __name__ == "__main__":
    main()
