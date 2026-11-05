"""[2026-11-01] user "카메라 정렬 도구를 만들어줘": live web page (http://localhost:8772) that guides the physical re-aim of the Orbbec.

The Orbbec moved after the training takes (static background vs 10-26: -6,-13 px, +1.0 deg; vs 10-30: -11,-4 px), so the
policy's global view and robot_cam_v005 are stale. The page compares the CURRENT Orbbec frame (the act_tagrig UI stream
:8770/stream/global, or CAM_ALIGN_SRC) with a REFERENCE = per-pixel median of the first frames of every training take of one day
(people / cloth / arms average out): a 50/50 blend, a red/cyan edge overlay and a flicker view, plus the similarity transform
live -> reference from ORB matches on the static scene (upper 62 % of the image: floor, far tables, room; the robots and the near
table are masked out) as numbers and Korean hand directions. Nothing is written to any calibration.
Deploy stage: pre-session camera check.
    python -m laundry_deploy.tools.cam_align      -> http://localhost:8772   (CAM_ALIGN_SRC, CAM_ALIGN_DAY)
"""

from __future__ import annotations
import os
import sys
import threading
import time
import urllib.request

import cv2
import numpy as np

from laundry_common import paths as egopaths

TAKES = egopaths.TAG_RIG
# 11-03: or the orbbec_server single frame http://127.0.0.1:8092/frame
SRC = os.environ.get("CAM_ALIGN_SRC", "http://127.0.0.1:8770/stream/global")
DEFAULT_DAY = os.environ.get("CAM_ALIGN_DAY", "20261030")
REFS = {
    "20261030": "10-30 (E1000 학습 데이터, 집기)",
    "20261026": "10-26 (GOLD 학습 데이터 + 카메라 캘리브 v005)",
    "20261027": "10-27 (GOLD 학습 데이터)",
    "20261102": "11-02 (펼치기 149 학습 데이터)",
    "20261103": "11-03 오전 (13:00 전, 카메라 옮기기 전)",
}
# 11-03 user "예전으로 돌리자": the Orbbec was re-aimed during take 20261103_130430 -> that take and later ones are not
# reference material
REF_BEFORE = {"20261103": "20261103_130000"}
VIEW_WH = (848, 480)  # 11-02+ takes are 1280x720: everything is compared at 848x480
MASK_Y = 0.62  # static scene = rows above 62 % of the height (robots, near table, people's hands below)
REF_FRAMES = (0, 60)  # frames of every take that enter the reference median
ASPECT_TOL = 0.03  # a reference frame must have the 848:480 aspect within this
ORB_FEATURES = 4000
MAX_MATCHES = 600  # best cross-checked matches kept
MIN_MATCHES = 20
MIN_INLIERS = 25
RANSAC_PX = 2.0
SHIFT_TOL_PX = 1.5  # advice thresholds: below these the camera counts as aligned
ROT_TOL_DEG = 0.15
SCALE_TOL = 0.006
CANNY_LO = 60
CANNY_HI = 150
EDGE_DIM = 0.45  # grey background brightness of the edge view
JPEG_QUALITY = 80
URL_TIMEOUT_S = 5
RETRY_S = 1.0
POLL_S = 0.2  # single-JPEG source poll period
STREAM_CHUNK_BYTES = 16384
PORT = 8772


class CamAlignError(RuntimeError):
    """No reference frame for the requested day."""


def makeRef(day: str):
    """-> (per-pixel median of the day's reference frames at VIEW_WH, number of frames)."""
    tImgs = []
    for tTake in sorted(TAKES.glob(f"{day}_*")):
        if day in REF_BEFORE and tTake.name >= REF_BEFORE[day]:
            continue
        for tK in REF_FRAMES:
            tIm = cv2.imread(str(tTake / f"color_{tK:05d}.jpg"))
            if tIm is not None and abs(tIm.shape[1] / tIm.shape[0] - VIEW_WH[0] / VIEW_WH[1]) < ASPECT_TOL:
                tImgs.append(cv2.resize(tIm, VIEW_WH))
    if not tImgs:
        raise CamAlignError(f"{day}: 참조 프레임 없음")
    return np.median(np.stack(tImgs), 0).astype(np.uint8), len(tImgs)


class LiveGrabber(threading.Thread):
    """Latest JPEG (mJpg, wall time mT) from the UI's MJPEG stream (reconnects) or a single-JPEG endpoint (polled)."""

    def __init__(self, src: str = SRC) -> None:
        super().__init__(daemon=True)
        self.mSrc = src
        self.mJpg = None
        self.mT = 0.0

    def run(self) -> None:
        while True:
            if self.mSrc.endswith("/frame"):  # single-JPEG endpoint (orbbec_server): poll
                try:
                    self.mJpg = urllib.request.urlopen(self.mSrc, timeout=URL_TIMEOUT_S).read()
                    self.mT = time.time()
                except Exception:
                    time.sleep(RETRY_S)
                time.sleep(POLL_S)
                continue
            try:
                tResp = urllib.request.urlopen(self.mSrc, timeout=URL_TIMEOUT_S)
                tBuf = b""
                while True:
                    tBuf += tResp.read(STREAM_CHUNK_BYTES)
                    tI = tBuf.find(b"\xff\xd8")
                    tJ = tBuf.find(b"\xff\xd9", tI)
                    if tI >= 0 and tJ > tI:
                        self.mJpg = tBuf[tI : tJ + 2]
                        self.mT = time.time()
                        tBuf = tBuf[tJ + 2 :]
            except Exception:
                time.sleep(RETRY_S)


_ORB = []


def _orb():
    if not _ORB:
        _ORB.append(cv2.ORB_create(ORB_FEATURES))
    return _ORB[0]


def similarity(ref: np.ndarray, cur: np.ndarray, orb=None):
    """cur -> ref similarity (dx, dy px at the image centre, rot deg, scale, inliers, H) from ORB matches in the static mask;
    None if too few matches / inliers."""
    tOrb = _orb() if orb is None else orb
    tMask = np.zeros(ref.shape[:2], np.uint8)
    tMask[: int(ref.shape[0] * MASK_Y)] = 255
    tGr, tGc = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    tKr, tDr = tOrb.detectAndCompute(tGr, tMask)
    tKc, tDc = tOrb.detectAndCompute(tGc, tMask)
    if tDr is None or tDc is None:
        return None
    tMatches = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(tDc, tDr), key=lambda x: x.distance)[:MAX_MATCHES]
    if len(tMatches) < MIN_MATCHES:
        return None
    tPc = np.float32([tKc[tMatch.queryIdx].pt for tMatch in tMatches])
    tPr = np.float32([tKr[tMatch.trainIdx].pt for tMatch in tMatches])
    tH, tInl = cv2.estimateAffinePartial2D(tPc, tPr, ransacReprojThreshold=RANSAC_PX)
    if tH is None or tInl.sum() < MIN_INLIERS:
        return None
    # report the shift at the image centre (where the table / tags are) rather than at the origin
    tC = np.array([ref.shape[1] / 2, ref.shape[0] / 2])
    tD = tH[:, :2] @ tC + tH[:, 2] - tC
    return dict(
        dx=float(tD[0]),
        dy=float(tD[1]),
        rot=float(np.degrees(np.arctan2(tH[1, 0], tH[0, 0]))),
        scale=float(np.hypot(tH[0, 0], tH[1, 0])),
        inliers=int(tInl.sum()),
        H=tH,
    )


def advice(s) -> list[str]:
    """Content must move by (dx, dy) to match the reference: content moves OPPOSITE to the camera's pan / tilt."""
    if s is None:
        return ["특징점 부족 -- 화면이 가려졌는지 확인"]
    tOut = []
    if abs(s["dx"]) > SHIFT_TOL_PX:
        tOut.append(f"카메라를 {'왼쪽' if s['dx'] > 0 else '오른쪽'}으로 돌리기 (좌우 {abs(s['dx']):.1f} px)")
    if abs(s["dy"]) > SHIFT_TOL_PX:
        tOut.append(f"카메라를 {'위로 들기' if s['dy'] > 0 else '아래로 숙이기'} (상하 {abs(s['dy']):.1f} px)")
    if abs(s["rot"]) > ROT_TOL_DEG:  # rot > 0 = content must turn clockwise (y down)
        tOut.append(f"카메라를 {'반시계 방향' if s['rot'] > 0 else '시계 방향'}으로 기울이기 ({abs(s['rot']):.2f}°)")
    if abs(s["scale"] - 1) > SCALE_TOL:
        tOut.append(f"카메라를 {'뒤로 (멀리)' if s['scale'] < 1 else '앞으로 (가까이)'} ({(s['scale'] - 1) * 100:+.1f} %)")
    return tOut or ["정렬됨 (모든 항목 허용 범위 안)"]


class AlignState:
    """The selected reference day, the reference cache {day: (image, n frames)} and the live grabber."""

    def __init__(self, live: LiveGrabber, day: str = DEFAULT_DAY) -> None:
        self.mLive = live
        self.mDay = day
        self.mRefs = {}

    def refImg(self) -> np.ndarray:
        if self.mDay not in self.mRefs:
            self.mRefs[self.mDay] = makeRef(self.mDay)
        return self.mRefs[self.mDay][0]

    def curImg(self):
        if self.mLive.mJpg is None:
            return None
        tIm = cv2.imdecode(np.frombuffer(self.mLive.mJpg, np.uint8), cv2.IMREAD_COLOR)
        return None if tIm is None else cv2.resize(tIm, VIEW_WH)

    def view(self, mode: str) -> np.ndarray:
        tRef, tCur = self.refImg(), self.curImg()
        if tCur is None:
            return np.zeros_like(tRef)
        if mode == "blend":
            tOut = cv2.addWeighted(tRef, 0.5, tCur, 0.5, 0)
        elif mode == "edges":
            tEr = cv2.Canny(cv2.cvtColor(tRef, cv2.COLOR_BGR2GRAY), CANNY_LO, CANNY_HI)
            tEc = cv2.Canny(cv2.cvtColor(tCur, cv2.COLOR_BGR2GRAY), CANNY_LO, CANNY_HI)
            tOut = (cv2.cvtColor(cv2.cvtColor(tCur, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR) * EDGE_DIM).astype(np.uint8)
            tOut[tEr > 0] = (0, 0, 255)
            tOut[tEc > 0] = (255, 255, 0)
            tOut[(tEr > 0) & (tEc > 0)] = (255, 255, 255)
        elif mode == "ref":
            tOut = tRef.copy()
        else:
            tOut = tCur.copy()
        cv2.line(tOut, (0, int(tOut.shape[0] * MASK_Y)), (tOut.shape[1], int(tOut.shape[0] * MASK_Y)), (0, 200, 255), 1)
        return tOut


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


def createApp(state: AlignState):
    from flask import Flask, Response, jsonify, request

    tApp = Flask(__name__)

    @tApp.route("/img/<mode>")
    def img(mode):
        tJpg = cv2.imencode(".jpg", state.view(mode), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])[1].tobytes()
        return Response(tJpg, mimetype="image/jpeg")

    @tApp.route("/api/state")
    def apiState():
        tCur = state.curImg()
        tSim = similarity(state.refImg(), tCur) if tCur is not None else None
        if tSim:
            tSim = {tK: tV for tK, tV in tSim.items() if tK != "H"}
        tLive = state.mLive
        tAge = round(time.time() - tLive.mT, 1) if tLive.mT else None
        return jsonify(
            day=state.mDay, refs=REFS, n_ref=state.mRefs[state.mDay][1], live_age_s=tAge, sim=tSim, advice=advice(tSim)
        )

    @tApp.route("/api/ref", methods=["POST"])
    def setRef():
        tDay = (request.json or {}).get("day")
        if tDay in REFS:
            state.mDay = tDay
            state.refImg()
        return jsonify(ok=True, day=state.mDay)

    @tApp.route("/")
    def index():
        return PAGE

    return tApp


def main(argv: list[str] | None = None) -> int:
    tLive = LiveGrabber(SRC)
    tLive.start()
    tState = AlignState(tLive, DEFAULT_DAY)
    tApp = createApp(tState)
    try:
        tState.refImg()
    except CamAlignError as tErr:
        print(tErr, file=sys.stderr)
        return 1
    print("카메라 정렬: http://localhost:8772", flush=True)
    tApp.run(host="0.0.0.0", port=PORT, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
