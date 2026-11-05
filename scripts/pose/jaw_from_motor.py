#!/usr/bin/env python3
"""[2026-10-29] HandUMI jaw label from the Feetech jaw servo (tag_rig_record gripper_motor_{left,right}.npz, 100 Hz, torque off) instead
of the wrist-camera jaw detector: writes <take>/<wrist dir>/umi_jaw.npz in the format the dataset builders read (umi_jaw infer +
jaw_aperture): t_ns, g_cont (0 closed .. 1 open, continuous), p_open (= g_cont), state (1 open / 0 closed, hysteresis + dwell),
valid, ev_t / ev_kind, g_is_aperture = 1.  The continuous motor position is kept as motor_pos.

Sides: wrist = RIGHT hand (camera 0x03110, motor 5B3D0480821), wrist_left = LEFT hand (0x03140, 5B3D0497471) -- the 10-29 15:57
physical test.  Fixed per-side calibration from the 7 takes of 10-29 evening (pooled p2 / p98 of the motor position):
  left  closed -850 (jaws together; closed ON the cloth reads -700..-800) .. open +200 (rises with the jaw opening)
  right closed 1830 .. open 1340 (mirrored: falls with the jaw opening; the right jaw only reached ~1340-1450, stiff linkage)
usage: jaw_from_motor.py <take_dir> [...] [--hi 0.55 --lo 0.35 --dwell 0.1]"""
import argparse, json, pathlib
import numpy as np

CAL = {"left": (-850.0, 200.0), "right": (1830.0, 1340.0)}          # (closed, open) motor ticks
DIRS = {"right": "wrist", "left": "wrist_left"}
SHIFT_TOL = 300.0          # ticks; a re-plug / power cycle moves the whole reading (10-30: +2150), normal closed drift is < 50


def hysteresis(t_ns, g, hi, lo, dwell_s):
    st = np.zeros(len(g), np.int64); s = int(g[0] > (hi + lo) / 2); cand = None; ev = []
    for i, v in enumerate(g):
        want = 1 if v > hi else 0 if v < lo else s
        if want != s:
            if cand is None: cand = i
            if (t_ns[i] - t_ns[cand]) / 1e9 >= dwell_s:
                s = want; st[cand:i] = s; ev.append((int(t_ns[cand]), "open" if s else "close")); cand = None
        else: cand = None
        st[i] = s
    return st, ev


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("takes", nargs="+"); ap.add_argument("--hi", type=float, default=0.55)
    ap.add_argument("--lo", type=float, default=0.35); ap.add_argument("--dwell", type=float, default=0.1); a = ap.parse_args()
    for take in map(pathlib.Path, a.takes):
        for side, d in DIRS.items():
            z = np.load(take / f"gripper_motor_{side}.npz"); t = z["host_ns"].astype(np.int64); pos = z["pos"].astype(np.float64)
            c, o = CAL[side]
            # 10-30 take 172433: the right servo was re-plugged and its whole reading moved by ~+2150 ticks (closed 1830 -> ~3985, same
            # ~450-tick stroke and direction).  The jaw rests CLOSED most of a take, so the take median is the closed reading: if it is
            # more than SHIFT_TOL away from the calibrated closed value, the calibration is shifted by that offset for this take.
            shift = float(np.median(pos)) - c
            if abs(shift) > SHIFT_TOL: c, o = c + shift, o + shift; print(f"{take.name} {side}: servo reading shifted by {shift:+.0f} ticks -> cal closed {c:.0f} open {o:.0f}")
            else: shift = 0.0
            # 11-04 12:xx: the LEFT jaw was re-assembled between takes 105052 and 123741 -> its reading now FALLS when opening (was rising)
            # with a ~600-tick stroke.  The jaw rests closed (median), so the open side is the side with the long tail: if that side is
            # opposite to the calibrated direction, take open = the 1st/99th percentile on the observed side for this take.
            med, p1, p99 = np.percentile(pos, [50, 1, 99]); obs = 1.0 if (p99 - med) > (med - p1) else -1.0
            if obs != np.sign(o - c):
                o = float(p99 if obs > 0 else p1); print(f"{take.name} {side}: jaw direction FLIPPED vs calibration -> cal closed {c:.0f} open {o:.0f} (stroke {abs(o - c):.0f})")
            g = np.clip((pos - c) / (o - c), 0.0, 1.0).astype(np.float32)
            st, ev = hysteresis(t, g, a.hi, a.lo, a.dwell)
            out = take / d; out.mkdir(exist_ok=True)
            np.savez(out / "umi_jaw.npz", t_ns=t, g_cont=g, p_open=g, state=st, valid=np.ones(len(t), np.int8), hand=side,
                     model="feetech_sts3215_motor", g_is_aperture=np.int8(1), motor_pos=pos.astype(np.float32),
                     ev_t=np.array([e[0] for e in ev], np.int64), ev_kind=np.array([e[1] for e in ev]),
                     cal=json.dumps(dict(closed=c, open=o, shift=shift, hi=a.hi, lo=a.lo, dwell_s=a.dwell)))
            print(f"{take.name} {side}: {len(t)} samples, open frac {st.mean():.2f}, events {len(ev)} "
                  f"({sum(e[1] == 'open' for e in ev)} open / {sum(e[1] == 'close' for e in ev)} close)", flush=True)


if __name__ == "__main__":
    main()
