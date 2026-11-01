#!/usr/bin/env python3
"""[2026-11-01] user "카메라 정렬 도구를 만들어줘": the Orbbec moved after the training takes (static background vs 10-26: -6,-13 px, +1.0 deg;
vs 10-30: -11,-4 px), so the policy's global view and robot_cam_v005 are stale.  Live web page that compares the CURRENT Orbbec frame
(the act_tagrig UI stream :8770/stream/global) with a REFERENCE = per-pixel median of the first frames of every training take of one
day (people / cloth / arms average out).  Shows a 50/50 blend, a red/cyan edge overlay and a flicker view, plus the similarity transform
live -> reference from ORB matches on the static scene (upper 62 % of the image: floor, far tables, room; the robots and the near table
are masked out) as numbers and Korean hand directions.  Nothing is written to any calibration; it only guides the physical re-aim.
  ~/laundry_dit_training/.venv-mac/bin/python deploy/cam_align.py      -> http://localhost:8772"""
from __future__ import annotations
import pathlib, threading, time, urllib.request
import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

TAKES = pathlib.Path.home() / "ego_collector/datasets/tag_rig"
SRC = "http://127.0.0.1:8770/stream/global"
REFS = {"20261030": "10-30 (E1000 학습 데이터, 집기)", "20261026": "10-26 (GOLD 학습 데이터 + 카메라 캘리브 v005)", "20261027": "10-27 (GOLD 학습 데이터)"}
MASK_Y = 0.62                      # static scene = rows above 62 % of the height (robots, near table, people's hands below)


def make_ref(day):
    ims = []
    for t in sorted(TAKES.glob(f"{day}_*")):
        for k in (0, 60):
            im = cv2.imread(str(t / f"color_{k:05d}.jpg"))
            if im is not None and im.shape[:2] == (480, 848): ims.append(im)
    if not ims: raise SystemExit(f"{day}: 참조 프레임 없음")
    return np.median(np.stack(ims), 0).astype(np.uint8), len(ims)


class Live(threading.Thread):
    """latest JPEG from the UI's MJPEG stream (reconnects)"""
    def __init__(self):
        super().__init__(daemon=True); self.jpg = None; self.t = 0.0

    def run(self):
        while True:
            try:
                r = urllib.request.urlopen(SRC, timeout=5); buf = b""
                while True:
                    buf += r.read(16384); i = buf.find(b"\xff\xd8"); j = buf.find(b"\xff\xd9", i)
                    if i >= 0 and j > i: self.jpg = buf[i:j + 2]; self.t = time.time(); buf = buf[j + 2:]
            except Exception:
                time.sleep(1.0)


def similarity(ref, cur, orb=cv2.ORB_create(4000)):
    """cur -> ref similarity (dx, dy px, rot deg, scale) from ORB matches in the static mask; None if too few inliers"""
    m = np.zeros(ref.shape[:2], np.uint8); m[:int(ref.shape[0] * MASK_Y)] = 255
    gr, gc = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    kr, dr = orb.detectAndCompute(gr, m); kc, dc = orb.detectAndCompute(gc, m)
    if dr is None or dc is None: return None
    mt = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(dc, dr), key=lambda x: x.distance)[:600]
    if len(mt) < 20: return None
    pc = np.float32([kc[x.queryIdx].pt for x in mt]); pr = np.float32([kr[x.trainIdx].pt for x in mt])
    H, inl = cv2.estimateAffinePartial2D(pc, pr, ransacReprojThreshold=2.0)
    if H is None or inl.sum() < 25: return None
    # report the shift at the image centre (where the table / tags are) rather than at the origin
    c = np.array([ref.shape[1] / 2, ref.shape[0] / 2]); d = H[:, :2] @ c + H[:, 2] - c
    return dict(dx=float(d[0]), dy=float(d[1]), rot=float(np.degrees(np.arctan2(H[1, 0], H[0, 0]))), scale=float(np.hypot(H[0, 0], H[1, 0])),
                inliers=int(inl.sum()), H=H)


def advice(s):
    """content must move by (dx, dy) to match the reference: content moves OPPOSITE to the camera's pan / tilt"""
    if s is None: return ["특징점 부족 -- 화면이 가려졌는지 확인"]
    out = []
    if abs(s["dx"]) > 1.5: out.append(f"카메라를 {'왼쪽' if s['dx'] > 0 else '오른쪽'}으로 돌리기 (좌우 {abs(s['dx']):.1f} px)")
    if abs(s["dy"]) > 1.5: out.append(f"카메라를 {'위로 들기' if s['dy'] > 0 else '아래로 숙이기'} (상하 {abs(s['dy']):.1f} px)")
    if abs(s["rot"]) > 0.15: out.append(f"카메라를 {'반시계 방향' if s['rot'] > 0 else '시계 방향'}으로 기울이기 ({abs(s['rot']):.2f}°)")   # rot > 0 = content must turn clockwise (y down)
    if abs(s["scale"] - 1) > 0.006: out.append(f"카메라를 {'뒤로 (멀리)' if s['scale'] < 1 else '앞으로 (가까이)'} ({(s['scale'] - 1) * 100:+.1f} %)")
    return out or ["정렬됨 (모든 항목 허용 범위 안)"]


app = Flask(__name__); live = Live(); live.start()
ST = dict(day="20261030"); REF = {}; LAST = {}


def ref_img():
    if ST["day"] not in REF: REF[ST["day"]] = make_ref(ST["day"])
    return REF[ST["day"]][0]


def cur_img():
    if live.jpg is None: return None
    return cv2.imdecode(np.frombuffer(live.jpg, np.uint8), cv2.IMREAD_COLOR)


def view(mode):
    ref, cur = ref_img(), cur_img()
    if cur is None: return np.zeros_like(ref)
    if mode == "blend": out = cv2.addWeighted(ref, 0.5, cur, 0.5, 0)
    elif mode == "edges":
        er, ec = cv2.Canny(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), 60, 150), cv2.Canny(cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY), 60, 150)
        out = (cv2.cvtColor(cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) * 0.45).astype(np.uint8)
        out[er > 0] = (0, 0, 255); out[ec > 0] = (255, 255, 0); out[(er > 0) & (ec > 0)] = (255, 255, 255)
    elif mode == "ref": out = ref.copy()
    else: out = cur.copy()
    cv2.line(out, (0, int(out.shape[0] * MASK_Y)), (out.shape[1], int(out.shape[0] * MASK_Y)), (0, 200, 255), 1)
    return out


@app.route("/img/<mode>")
def img(mode):
    return Response(cv2.imencode(".jpg", view(mode), [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes(), mimetype="image/jpeg")


@app.route("/api/state")
def state():
    cur = cur_img(); s = similarity(ref_img(), cur) if cur is not None else None
    if s: s = {k: v for k, v in s.items() if k != "H"}
    return jsonify(day=ST["day"], refs=REFS, n_ref=REF[ST["day"]][1], live_age_s=round(time.time() - live.t, 1) if live.t else None, sim=s, advice=advice(s))


@app.route("/api/ref", methods=["POST"])
def set_ref():
    d = (request.json or {}).get("day")
    if d in REFS: ST["day"] = d; ref_img()
    return jsonify(ok=True, day=ST["day"])


PAGE = """<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>카메라 정렬</title><style>
:root{--bg:#fff;--fg:#111;--mut:#666;--ok:#0a7d32;--bad:#b3261e;--card:#f3f3f3}
@media (prefers-color-scheme:dark){:root{--bg:#141414;--fg:#eee;--mut:#9a9a9a;--ok:#5bd17f;--bad:#ff8a80;--card:#222}}
body{margin:0;padding:12px 16px;background:var(--bg);color:var(--fg);font:15px -apple-system,system-ui,sans-serif}
img{max-width:100%;border-radius:6px} .row{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start}
.card{background:var(--card);border-radius:8px;padding:10px 14px;min-width:260px} .big{font-size:22px;font-weight:600}
.ok{color:var(--ok)} .bad{color:var(--bad)} .mut{color:var(--mut);font-size:13px} button,select{font-size:14px;padding:4px 10px}
</style></head><body>
<h2 style="margin:4px 0 10px">Orbbec 카메라 정렬 (학습 화면에 맞추기)</h2>
<div class="row"><div><div style="margin-bottom:6px">
 보기 <button onclick="m='blend'">반반 겹침</button> <button onclick="m='edges'">윤곽선 (빨강 = 학습, 하늘 = 지금)</button>
 <button onclick="m='flick'">깜빡이기</button> <button onclick="m='cur'">지금</button> <button onclick="m='ref'">학습</button>
 &nbsp; 기준 <select id="day" onchange="fetch('/api/ref',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({day:this.value})})"></select></div>
 <img id="im" width="848" height="480"><div class="mut">주황 선 위쪽(바닥·먼 책상·방)만으로 계산합니다. 로봇과 앞 테이블은 제외.</div></div>
<div class="card"><div class="mut">지금 화면 → 학습 화면 (화면 중앙 기준)</div><div id="num" class="big">-</div>
 <div id="adv" style="margin-top:10px;line-height:1.7"></div><div id="meta" class="mut" style="margin-top:10px"></div></div></div>
<script>
let m='blend',fl=0;const im=document.getElementById('im');
setInterval(()=>{let v=m;if(m=='flick'){fl^=1;v=fl?'cur':'ref'}im.src='/img/'+v+'?'+Date.now()},m=='flick'?600:400);
async function poll(){try{const s=await (await fetch('/api/state')).json();const d=document.getElementById('day');
 if(!d.options.length)for(const k in s.refs){const o=document.createElement('option');o.value=k;o.textContent=s.refs[k];d.appendChild(o)}d.value=s.day;
 const x=s.sim;document.getElementById('num').innerHTML=x?`좌우 ${x.dx.toFixed(1)} px<br>상하 ${x.dy.toFixed(1)} px<br>회전 ${x.rot.toFixed(2)}°<br>배율 ${((x.scale-1)*100).toFixed(1)} %`:'계산 불가';
 const ok=s.advice.length==1&&s.advice[0].startsWith('정렬됨');document.getElementById('adv').innerHTML=s.advice.map(a=>`<div class="${ok?'ok':'bad'}">${a}</div>`).join('');
 document.getElementById('meta').textContent=`기준 프레임 ${s.n_ref}장 중앙값 · 특징점 일치 ${x?x.inliers:0} · 영상 지연 ${s.live_age_s}s`}catch(e){}}
setInterval(poll,700);poll();
</script></body></html>"""


@app.route("/")
def index():
    return PAGE


if __name__ == "__main__":
    ref_img(); print("카메라 정렬: http://localhost:8772", flush=True)
    app.run(host="0.0.0.0", port=8772, threaded=True)
