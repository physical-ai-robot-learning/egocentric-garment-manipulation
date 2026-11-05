#!/usr/bin/env python3
"""[2026-11-05] Gold labeling page for grasp events (held / empty / uncertain), BLIND to the v0 prediction.

Builds one short clip per selected close event (that hand's wrist cam, close - 1 s .. open + 1 s, capped at 7 s, 15 fps; a bar shows
the jaw CLOSED state from the motor only) and serves a page on 127.0.0.1:PORT: keys 1 held, 2 empty, 3 uncertain, 0 skip, <- back.
Labels -> DIR/gold_events.csv (latest per event).
usage: grasp_gold_ui.py DIR [PORT]      (DIR = grasp_label_v0 output with events.csv)
"""
import csv, http.server, json, os, pathlib, random, sys, urllib.parse
import cv2, numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); import grasp_label_v0 as G  # noqa: E402

D = pathlib.Path(sys.argv[1]).expanduser(); PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8098
CL = D / "clips"; CL.mkdir(exist_ok=True); GOLD = D / "gold_events.csv"
N_HELD, N_UNK, SEED = 65, 75, 1105


def select():
    ev = [r for r in csv.DictReader(open(D / "events.csv")) if r["rest"] == "0"]
    rnd = random.Random(SEED)
    if os.environ.get("GOLD_ALL") == "1": sel = list(ev)      # r2+: events.csv is already the sample (uniform random), take it as is
    else:
        by = {k: [r for r in ev if r["label"] == k] for k in ("held", "empty", "unknown")}
        sel = by["empty"] + rnd.sample(by["held"], min(N_HELD, len(by["held"]))) + rnd.sample(by["unknown"], min(N_UNK, len(by["unknown"])))
    rnd.shuffle(sel); return [dict(id=f"{r['video'].rsplit('_', 1)[0]}_{r['event']}", video=r["video"], event=r["event"], t_close_s=float(r["t_close_s"]), closed_s=float(r["closed_s"])) for r in sel]


def clip(e):
    out = CL / f"{e['id']}.mp4"
    if out.exists(): return
    take = "_".join(e["video"].split("_")[:2]); d = int(e["video"].split("_")[2][1:]) - 1; t0, _ = G.demo_window(take, d)
    cam = dict(G.CAMS)[e["event"][0]]; cd = G.TR / take / cam; tw = np.load(cd / "t_ns.npy"); J = np.load(cd / "umi_jaw.npz")
    # 11-05 user: "jaw close가 한번만 있게" -> cut the clip at the previous open / next close so only THIS close is in it
    ev = list(zip(J["ev_t"].tolist(), J["ev_kind"].tolist())); te_g = t0 + e["t_close_s"] * 1e9
    te = min((t for t, k in ev if k == "close"), key=lambda t: abs(t - te_g)); tn = te + e["closed_s"] * 1e9
    prev_open = max([t for t, k in ev if k == "open" and t < te], default=te - 1e10); next_close = min([t for t, k in ev if k == "close" and t > te], default=tn + 1e10)
    a = max(te - 1.0e9, prev_open + 0.15e9); b = min(tn + 1.0e9, next_close - 0.15e9, a + 7e9)
    tmp = out.with_suffix(".raw.mp4"); vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (640, 400))
    for t in np.arange(a, b, 1e9 / 15):
        i = int(np.searchsorted(tw, t).clip(0, len(tw) - 1)); im = cv2.imread(str(cd / f"wrist_{i:05d}.jpg"))
        im = np.zeros((360, 640, 3), np.uint8) if im is None else cv2.resize(im, (640, 360))
        closed = te <= t < tn
        bar = np.full((40, 640, 3), (0, 160, 255) if closed else (90, 90, 90), np.uint8)
        cv2.putText(bar, ("JAW CLOSED" if closed else "jaw open") + f"   {(t - a) / 1e9:4.1f}s", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0) if closed else (255, 255, 255), 2)
        vw.write(np.vstack([im, bar]))
    vw.release()
    import subprocess
    subprocess.run(["/opt/homebrew/bin/ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "24", "-movflags", "+faststart", str(out)], check=True); tmp.unlink()


def load_gold():
    return {r["id"]: r for r in csv.DictReader(open(GOLD))} if GOLD.exists() else {}


def save_gold(g):
    with open(GOLD, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "video", "event", "gold"]); w.writeheader(); w.writerows(g.values())


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>Grasp gold labels</title><style>
body{font-family:-apple-system,sans-serif;background:#111;color:#eee;margin:0;padding:16px;text-align:center}
video{width:min(960px,100%);background:#000} .b{font-size:20px;padding:14px 22px;margin:6px;border:0;border-radius:8px;cursor:pointer}
#h{background:#3c3} #e{background:#e44;color:#fff} #u{background:#b7b} #s{background:#777;color:#fff} .info{margin:8px;color:#aaa}
</style></head><body><div class=info id=prog></div><video id=v autoplay loop muted playsinline></video>
<div><button class=b id=h onclick="lab('held')">1 잡힘</button><button class=b id=e onclick="lab('empty')">2 헛잡기</button>
<button class=b id=u onclick="lab('__L3__')">3 __T3__</button><button class=b id=s onclick="lab('skip')">0 건너뛰기</button></div>
<div class=info>주황 띠 = 집게 닫힘(모터). 닫힌 동안 천이 집게 사이에 물려 있었나? &nbsp; ← 이전 &nbsp; 스페이스 = 다시 재생 &nbsp; 속도: <a href=# onclick="sp(0.5)">0.5x</a> <a href=# onclick="sp(1)">1x</a></div>
<div class=info id=last></div>
<script>
let E=[],G={},i=0,rate=1;
async function init(){const r=await (await fetch('/api/state')).json();E=r.events;G=r.gold;i=E.findIndex(e=>!G[e.id]);if(i<0)i=E.length;show()}
function sp(x){rate=x;document.getElementById('v').playbackRate=x;return false}
function show(){const n=Object.keys(G).length;document.getElementById('prog').textContent=`${Math.min(i+1,E.length)} / ${E.length}   (라벨 ${n})`;
 if(i>=E.length){document.getElementById('v').removeAttribute('src');document.getElementById('prog').textContent+='  — 끝! 고마워요';return}
 const v=document.getElementById('v');v.src='/clips/'+E[i].id+'.mp4';v.playbackRate=rate;document.getElementById('last').textContent=E[i].id+(G[E[i].id]?'  현재: '+G[E[i].id]:'')}
async function lab(g){if(i>=E.length)return;const e=E[i];await fetch('/api/label',{method:'POST',body:JSON.stringify({id:e.id,video:e.video,event:e.event,gold:g})});G[e.id]=g;i++;show()}
document.onkeydown=k=>{if(k.key=='1')lab('held');else if(k.key=='2')lab('empty');else if(k.key=='3')lab('__L3__');else if(k.key=='0')lab('skip');
 else if(k.key=='ArrowLeft'){i=Math.max(0,i-1);show()}else if(k.key==' '){k.preventDefault();const v=document.getElementById('v');v.currentTime=0;v.play()}};
init();
</script></body></html>"""


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, body, ctype, code=200):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        p = urllib.parse.urlparse(self.path).path
        if p == "/": return self._send(PAGE.replace("__L3__", os.environ.get("GOLD_L3", "uncertain")).replace("__T3__", os.environ.get("GOLD_T3", "애매함")).encode(), "text/html; charset=utf-8")
        if p == "/api/state": return self._send(json.dumps(dict(events=EV, gold={k: v["gold"] for k, v in load_gold().items()})).encode(), "application/json")
        if p.startswith("/clips/"):
            f = CL / pathlib.Path(p).name
            if f.exists(): return self._send(f.read_bytes(), "video/mp4")
        self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        r = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        g = load_gold(); g[r["id"]] = {k: r[k] for k in ("id", "video", "event", "gold")}; save_gold(g); self._send(b"{}", "application/json")


if __name__ == "__main__":
    EV = select()
    for n, e in enumerate(EV):
        clip(e)
        if n % 20 == 0: print(f"clips {n}/{len(EV)}", flush=True)
    json.dump(EV, open(D / "gold_selection.json", "w"), indent=1)
    print(f"READY http://127.0.0.1:{PORT}  ({len(EV)} events)", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
