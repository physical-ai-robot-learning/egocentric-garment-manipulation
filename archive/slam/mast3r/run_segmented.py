#!/usr/bin/env python3
"""[2026-10-21] Quality-triggered RESTART of MASt3R-SLAM (GPT plan; UMI T-shirt take lost tracking at frame 327 of 4089 and the official
loop never left RELOC). Runs run_perframe.py on consecutive SEGMENTS of one frame folder:
  - a segment starts at s and covers at most --max-len frames
  - after it runs, the first stretch of >= --lost-run consecutive non-tracking frames (RELOC / lost) ends it at `cut`
  - the next segment starts at s + cut - --overlap (good frames shared for stitching); if the segment died almost immediately
    (cut < --min-good) the next start skips ahead by --skip frames
  - a segment that tracked to its end: next start = end - --overlap
Each segment = its own SLAM world (own Sim3 to the robot base later). Frames are symlinked, the official code is not modified.
Writes runs/<tag>/segs/<s>_<e>.csv and runs/<tag>/segs/manifest.json [{start, end, cut, csv}].
    python run_segmented.py <tag> [--max-len 360 --lost-run 15 --overlap 45 --min-good 30 --skip 15]"""
import argparse, csv, json, os, pathlib, subprocess, sys, time, types

BASE = pathlib.Path(os.environ.get("M3_BASE", "/srv/data/johann/trackA_mast3r_pose_v1"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--max-len", type=int, default=360); ap.add_argument("--lost-run", type=int, default=15)
    ap.add_argument("--overlap", type=int, default=45); ap.add_argument("--min-good", type=int, default=30); ap.add_argument("--skip", type=int, default=15); ap.add_argument("--seg-timeout", type=float, default=600); a = ap.parse_args()
    src = BASE / "frames" / a.tag; frames = sorted((src / "rgb").glob("*.png")); n = len(frames)
    out = BASE / "runs" / a.tag / "segs"; out.mkdir(parents=True, exist_ok=True); man = []; s = 0; t0 = time.time()
    while s < n - a.min_good:
        e = min(n, s + a.max_len); sd = BASE / "frames" / f"{a.tag}__seg" / f"{s:06d}_{e:06d}"; (sd / "rgb").mkdir(parents=True, exist_ok=True)
        for k, f in enumerate(frames[s:e]):
            l = sd / "rgb" / f"{k:06d}.png"
            if not l.exists(): l.symlink_to(f)
        oc = out / f"{s:06d}_{e:06d}.csv"
        # 10-21 (4090, torch 2.5.1): run_perframe finishes and writes the CSV, then dies / hangs at exit on a shared-memory unlink
        # (CudaIPC). Output goes to a log FILE (a captured pipe filled up and blocked the writer); once the CSV + its .json exist the
        # process gets 15 s to exit and is then killed. Hard cap --seg-timeout.
        if oc.exists(): oc.unlink()
        oj = oc.with_suffix(".json")
        if oj.exists(): oj.unlink()
        lg = open(out / f"{s:06d}_{e:06d}.log", "w")
        pr = subprocess.Popen([sys.executable, str(BASE / "run_perframe.py"), "--frames", str(sd / "rgb"), "--calib", str(src / "calib.yaml"), "--subsample", "1", "--seed", "0", "--out", str(oc)],
                              cwd=str(BASE), stdout=lg, stderr=subprocess.STDOUT, start_new_session=True)
        t1 = time.time(); done_at = None
        while pr.poll() is None:
            time.sleep(1)
            if done_at is None and oc.exists() and oj.exists(): done_at = time.time()
            if (done_at and time.time() - done_at > 15) or time.time() - t1 > a.seg_timeout:
                try: os.killpg(pr.pid, 9)
                except Exception: pass
                pr.wait(); break
        lg.close(); r = types.SimpleNamespace(returncode=pr.returncode, stderr=open(out / f"{s:06d}_{e:06d}.log").read()[-800:] if not oc.exists() else "")
        st = [row["state"] for row in csv.DictReader(open(oc))] if oc.exists() else []
        good = [x in ("INIT", "TRACKING") for x in st]; cut = len(good); run = 0
        for i, g in enumerate(good):
            run = 0 if g else run + 1
            if run >= a.lost_run: cut = i - run + 1; break
        man.append(dict(start=s, end=e, cut=cut, good=int(sum(good[:cut])), csv=str(oc.relative_to(BASE)), rc=r.returncode))
        print(f"seg {s}-{e}: tracked {cut} frames ({sum(good[:cut])} good) rc={r.returncode}  [{time.time() - t0:.0f}s]", flush=True)
        if not oc.exists(): print(r.stderr[-800:], flush=True)
        if cut < a.min_good: s += a.skip
        elif cut < e - s: s = max(s + cut - a.overlap, s + a.skip)       # 10-21 fix: cut == overlap restarted at the same frame forever
        else: s = e - a.overlap if e < n else n
    (out / "manifest.json").write_text(json.dumps(dict(tag=a.tag, n=n, args=vars(a), segments=man), indent=1))
    cov = set()
    for m in man: cov.update(range(m["start"], m["start"] + m["cut"]))
    print(f"SEG_DONE {a.tag}: {len(man)} segments, tracked coverage {len(cov)}/{n} = {len(cov) / n * 100:.0f}%", flush=True)


if __name__ == "__main__":
    main()
