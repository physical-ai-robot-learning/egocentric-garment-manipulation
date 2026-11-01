"""Jaw-tip keypoint labelling for the UMI wrist cameras (2026-10-22, after the camera re-angle; GPT/user: learn jaw geometry, not
open/closed). Two points per frame: tip A = left jaw tip in the image, tip B = right jaw tip; aperture g is computed later from
their distance. The binary ResNet (umi_jaw v003) failed on half-open left jaws (77 % on take 105200), so half-open frames are
over-sampled.

  select  pick frames of a take (both hands), draft the two tips from the dark jaw blobs, write datasets/jaw_kpt/<take>/<hand>/
          {frames/*.jpg (the jaw crop), labels.json}
  serve   browser tool at http://localhost:8765 : drag the two points, keys  Enter/→ = OK + next, ← = back, x = jaw not visible,
          s = swap A/B. Every change is saved to labels.json immediately.
    .venv/bin/python scripts/pose/jaw_kpt_label.py select datasets/tag_rig/20261022_105200 [--left 260 --right 140]
    .venv/bin/python scripts/pose/jaw_kpt_label.py serve datasets/jaw_kpt/20261022_105200"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CROP = (0.55, 1.00, 0.15, 0.85)        # y0, y1, x0, x1 of the wrist frame: both jaws sit bottom-centre since the 10-22 re-angle
DIRS = {"right": "wrist", "left": "wrist_left"}


def crop(im):
    h, w = im.shape[:2]; y0, y1, x0, x1 = CROP
    return im[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]


def draft_tips(c):
    """two dark blobs touching the bottom edge = the jaws; tip = top-most pixel of each. Returns [[xA,yA],[xB,yB]] in crop px or None"""
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY); s = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)[:, :, 1]
    m = ((g < 95) & (s < 90)).astype(np.uint8); m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m); H, W = m.shape
    bl = [j for j in range(1, n) if st[j, 1] + st[j, 3] >= H - 3 and st[j, 4] > 0.01 * H * W and st[j, 3] > 0.3 * H]
    if not bl: return None
    bl = sorted(bl, key=lambda j: -st[j, 4])[:2]
    tips = []
    for j in bl:
        ys, xs = np.where(lab == j); top = ys.min(); sel = xs[ys <= top + 3]; tips.append([float(np.median(sel)), float(top)])
    if len(tips) == 1:          # jaws touching -> one merged blob: split its top row region into two tips around the centre
        ys, xs = np.where(lab == bl[0]); band = ys <= ys.min() + 0.15 * (ys.max() - ys.min()); x = xs[band]
        tips = [[float(np.percentile(x, 15)), float(ys.min())], [float(np.percentile(x, 85)), float(ys.min())]]
    tips.sort(key=lambda p: p[0])
    return tips


def select(a):
    E = Path(a.take); out = ROOT / "datasets/jaw_kpt" / E.name; meta = json.loads((E / "meta.json").read_text())
    t0 = int(meta["t_start_monotonic_ns"]); win = [(t0 + int(p["t0"] * 1e9), t0 + int((p["t0"] + p["dur"]) * 1e9)) for p in meta["protocol"] if p["text"].startswith("자유:")]
    rng = np.random.default_rng(0)
    for hand, n in (("left", a.left), ("right", a.right)):
        d = E / DIRS[hand]; tw = np.load(d / "t_ns.npy"); inw = np.zeros(len(tw), bool)
        for lo, hi in win: inw |= (tw >= lo) & (tw <= hi)
        idx = np.where(inw)[0]
        p = None
        if (d / "umi_jaw.npz").exists():
            z = np.load(d / "umi_jaw.npz"); p = np.interp(tw[idx], z["t_ns"], z["p_open"])
        # over-sample the uncertain / half-open range and the transitions (p changes fast), then fill uniformly
        pick = []
        if p is not None:
            mid = idx[(p > 0.03) & (p < 0.97)]; dp = np.abs(np.gradient(p)); trans = idx[np.argsort(-dp)[:len(idx) // 20]]
            pick += list(rng.choice(mid, min(len(mid), n // 2), replace=False)) + list(rng.choice(trans, min(len(trans), n // 4), replace=False))
        rest = np.setdiff1d(idx, pick); pick += list(rng.choice(rest, max(0, n - len(set(pick))), replace=False))
        pick = sorted(set(int(i) for i in pick))[:n] if len(set(pick)) > n else sorted(set(int(i) for i in pick))
        pick = sorted(pick[:: max(1, len(pick) // n)])[:n]
        fd = out / hand / "frames"; fd.mkdir(parents=True, exist_ok=True); labels = []
        for i in pick:
            c = crop(cv2.imread(str(d / f"wrist_{i:05d}.jpg"))); fn = f"{i:05d}.jpg"; cv2.imwrite(str(fd / fn), c, [cv2.IMWRITE_JPEG_QUALITY, 92])
            tips = draft_tips(c)
            labels.append(dict(frame=i, file=fn, t_ns=int(tw[i]), tips=tips, visible=tips is not None, status="draft",
                               p_open_v003=float(np.interp(tw[i], z["t_ns"], z["p_open"])) if p is not None else None))
        (out / hand / "labels.json").write_text(json.dumps(dict(take=E.name, hand=hand, crop=CROP, labels=labels), indent=1))
        print(f"{hand}: {len(labels)} 프레임, 초안 있는 것 {sum(l['visible'] for l in labels)} -> {out / hand}")


PAGE = """<!doctype html><meta charset=utf-8><title>jaw tips</title>
<style>body{font:14px sans-serif;margin:10px;background:#222;color:#eee}canvas{border:1px solid #888;cursor:crosshair}
#bar{margin:6px 0}b{color:#ff0}</style>
<div id=bar></div><canvas id=c></canvas>
<div>드래그로 점 이동 · <b>Enter/→</b> 확인+다음 · <b>←</b> 이전 · <b>x</b> 집게 안 보임 · <b>s</b> A/B 바꾸기 · <b>h</b> 손 바꾸기</div>
<script>
let hand='left',L=null,i=0,drag=-1,img=new Image(),S=1;const cv=document.getElementById('c'),cx=cv.getContext('2d');
async function load(){L=await (await fetch('/labels/'+hand)).json();i=L.labels.findIndex(l=>l.status!='ok');if(i<0)i=0;show()}
function show(){const l=L.labels[i];img.onload=()=>{S=Math.min(1100/img.width,1.6);cv.width=img.width*S;cv.height=img.height*S;draw()};img.src='/img/'+hand+'/'+l.file;
 const done=L.labels.filter(x=>x.status=='ok').length;document.getElementById('bar').innerHTML=`<b>${hand}</b> ${i+1}/${L.labels.length} · 확인됨 ${done} · 상태 ${l.status}${l.visible?'':' · <b>안 보임</b>'}`}
function draw(){cx.drawImage(img,0,0,cv.width,cv.height);const l=L.labels[i];if(!l.tips||!l.visible)return;
 l.tips.forEach((p,k)=>{cx.beginPath();cx.arc(p[0]*S,p[1]*S,9,0,7);cx.fillStyle=k?'#0f0':'#f0f';cx.fill();cx.fillStyle='#000';cx.fillText(k?'B':'A',p[0]*S-3,p[1]*S+4)});
 cx.strokeStyle='#ff0';cx.beginPath();cx.moveTo(l.tips[0][0]*S,l.tips[0][1]*S);cx.lineTo(l.tips[1][0]*S,l.tips[1][1]*S);cx.stroke()}
async function save(){await fetch('/labels/'+hand+'/'+i,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(L.labels[i])})}
cv.onmousedown=e=>{const l=L.labels[i];const x=e.offsetX/S,y=e.offsetY/S;if(!l.tips){l.tips=[[x-20,y],[x+20,y]];l.visible=true}
 const d=l.tips.map(p=>Math.hypot(p[0]-x,p[1]-y));drag=d[0]<d[1]?0:1;l.tips[drag]=[x,y];draw()};
cv.onmousemove=e=>{if(drag<0)return;L.labels[i].tips[drag]=[e.offsetX/S,e.offsetY/S];draw()};
cv.onmouseup=()=>{if(drag>=0){drag=-1;save()}};
document.onkeydown=async e=>{const l=L.labels[i];
 if(e.key=='Enter'||e.key=='ArrowRight'){l.status='ok';await save();if(i<L.labels.length-1)i++;show()}
 else if(e.key=='ArrowLeft'){if(i>0)i--;show()}
 else if(e.key=='x'){l.visible=!l.visible;l.status='ok';await save();if(i<L.labels.length-1)i++;show()}
 else if(e.key=='s'&&l.tips){l.tips=[l.tips[1],l.tips[0]];await save();draw()}
 else if(e.key=='h'){hand=hand=='left'?'right':'left';load()}};
load();
</script>"""


def serve(a):
    from flask import Flask, jsonify, request, send_from_directory
    D = Path(a.dir).resolve(); app = Flask(__name__)   # absolute: send_from_directory resolves relative paths against the app root (scripts/)
    @app.get("/")
    def index(): return PAGE
    @app.get("/labels/<hand>")
    def get(hand): return jsonify(json.loads((D / hand / "labels.json").read_text()))
    @app.post("/labels/<hand>/<int:k>")
    def put(hand, k):
        p = D / hand / "labels.json"; j = json.loads(p.read_text()); j["labels"][k] = request.get_json(); p.write_text(json.dumps(j, indent=1)); return "ok"
    @app.get("/img/<hand>/<fn>")
    def img(hand, fn): return send_from_directory(D / hand / "frames", fn)
    print(f"라벨링 도구: http://localhost:{a.port}  (왼손부터, h 로 오른손)", flush=True); app.run(port=a.port)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select"); s.add_argument("take"); s.add_argument("--left", type=int, default=260); s.add_argument("--right", type=int, default=140)
    v = sub.add_parser("serve"); v.add_argument("dir"); v.add_argument("--port", type=int, default=8765)
    a = ap.parse_args(); {"select": select, "serve": serve}[a.cmd](a)


if __name__ == "__main__":
    main()
