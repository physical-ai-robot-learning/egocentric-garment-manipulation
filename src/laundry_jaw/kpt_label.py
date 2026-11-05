"""Jaw-tip keypoint labelling for the UMI wrist cameras (2026-10-22, after the camera re-angle; was scripts/pose/jaw_kpt_label.py).

Pipeline stage: jaw state (stage 2), training data of the jaw-tip keypoint model (`kpt_train`, then `aperture`).
GPT/user: learn jaw geometry, not open/closed. Two points per frame: tip A = left jaw tip in the image, tip B = right jaw tip;
aperture g is computed later from their distance. The binary ResNet (umi_jaw v003) failed on half-open left jaws (77 % on take
105200), so half-open frames are over-sampled.

  select  pick frames of a take (both hands), draft the two tips from the dark jaw blobs, write
          EGO_DATA_ROOT/datasets/jaw_kpt/<take>/<hand>/{frames/*.jpg (the jaw crop), labels.json}
  serve   browser tool at http://localhost:8765 : drag the two points, keys  Enter/→ = OK + next, ← = back, x = jaw not visible,
          s = swap A/B. Every change is saved to labels.json immediately.
    python -m laundry_jaw.kpt_label select datasets/tag_rig/20261022_105200 [--left 260 --right 140]
    python -m laundry_jaw.kpt_label serve datasets/jaw_kpt/20261022_105200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from laundry_common.paths import DATA_ROOT

JAW_KPT_DIR = DATA_ROOT / "datasets/jaw_kpt"
# y0, y1, x0, x1 of the wrist frame: both jaws sit bottom-centre since the 10-22 re-angle
CROP = (0.55, 1.00, 0.15, 0.85)
WRIST_DIRS = {"right": "wrist", "left": "wrist_left"}
NS_PER_S = 1e9
# Draft tips: a jaw is a dark (grey < 95), unsaturated (S < 90) blob ...
DARK_GREY_MAX = 95
DARK_SATURATION_MAX = 90
OPEN_KERNEL_PX = 7
# ... touching the bottom edge of the crop (within 3 px), larger than 1 % of the crop and taller than 30 % of it.
BOTTOM_TOUCH_PX = 3
MIN_BLOB_AREA_FRAC = 0.01
MIN_BLOB_HEIGHT_FRAC = 0.3
# Tip = median x of the blob pixels within 3 px of its top row.
TIP_ROW_BAND_PX = 3
# One merged blob (jaws touching): tips at the 15th / 85th x percentile of its top 15 %.
MERGED_TOP_FRAC = 0.15
MERGED_TIP_PERCENTILES = (15, 85)
# Frames whose classifier p_open is in (0.03, 0.97) count as half-open; the fastest-changing 1/20 of the frames as transitions.
HALF_OPEN_LO = 0.03
HALF_OPEN_HI = 0.97
TRANSITION_SHARE_DIVISOR = 20
SELECT_SEED = 0
FREE_PROTOCOL_PREFIX = "자유:"
FRAME_JPEG_QUALITY = 92
DEFAULT_LEFT_FRAMES = 260
DEFAULT_RIGHT_FRAMES = 140
DEFAULT_PORT = 8765


def crop(image: np.ndarray) -> np.ndarray:
    tH, tW = image.shape[:2]
    tY0, tY1, tX0, tX1 = CROP
    return image[int(tH * tY0):int(tH * tY1), int(tW * tX0):int(tW * tX1)]


def draftTips(cropImage: np.ndarray) -> list[list[float]] | None:
    """Two dark blobs touching the bottom edge = the jaws; tip = top-most pixel of each. [[xA, yA], [xB, yB]] in crop px, or None."""
    tGrey = cv2.cvtColor(cropImage, cv2.COLOR_BGR2GRAY)
    tSaturation = cv2.cvtColor(cropImage, cv2.COLOR_BGR2HSV)[:, :, 1]
    tMask = ((tGrey < DARK_GREY_MAX) & (tSaturation < DARK_SATURATION_MAX)).astype(np.uint8)
    tMask = cv2.morphologyEx(tMask, cv2.MORPH_OPEN, np.ones((OPEN_KERNEL_PX, OPEN_KERNEL_PX), np.uint8))
    tCount, tLabels, tStats, _ = cv2.connectedComponentsWithStats(tMask)
    tH, tW = tMask.shape
    tBlobs = [tJ for tJ in range(1, tCount)
              if tStats[tJ, 1] + tStats[tJ, 3] >= tH - BOTTOM_TOUCH_PX and tStats[tJ, 4] > MIN_BLOB_AREA_FRAC * tH * tW
              and tStats[tJ, 3] > MIN_BLOB_HEIGHT_FRAC * tH]
    if not tBlobs:
        return None
    tBlobs = sorted(tBlobs, key=lambda tJ: -tStats[tJ, 4])[:2]
    tTips = []
    for tJ in tBlobs:
        tYs, tXs = np.where(tLabels == tJ)
        tTop = tYs.min()
        tTopXs = tXs[tYs <= tTop + TIP_ROW_BAND_PX]
        tTips.append([float(np.median(tTopXs)), float(tTop)])
    if len(tTips) == 1:          # jaws touching -> one merged blob: split its top row region into two tips around the centre
        tYs, tXs = np.where(tLabels == tBlobs[0])
        tBand = tYs <= tYs.min() + MERGED_TOP_FRAC * (tYs.max() - tYs.min())
        tBandXs = tXs[tBand]
        tTips = [[float(np.percentile(tBandXs, tPct)), float(tYs.min())] for tPct in MERGED_TIP_PERCENTILES]
    tTips.sort(key=lambda tTip: tTip[0])
    return tTips


def _freeWindows(meta: dict) -> list[tuple[int, int]]:
    tStart = int(meta["t_start_monotonic_ns"])
    return [(tStart + int(tStep["t0"] * NS_PER_S), tStart + int((tStep["t0"] + tStep["dur"]) * NS_PER_S))
            for tStep in meta["protocol"] if tStep["text"].startswith(FREE_PROTOCOL_PREFIX)]


def _pickFrames(candidates: np.ndarray, pOpen: np.ndarray | None, count: int, rng: np.random.Generator) -> list[int]:
    """Over-sample the uncertain / half-open range and the transitions (p changes fast), then fill uniformly."""
    tPick = []
    if pOpen is not None:
        tHalfOpen = candidates[(pOpen > HALF_OPEN_LO) & (pOpen < HALF_OPEN_HI)]
        tRate = np.abs(np.gradient(pOpen))
        tTransitions = candidates[np.argsort(-tRate)[:len(candidates) // TRANSITION_SHARE_DIVISOR]]
        tPick += list(rng.choice(tHalfOpen, min(len(tHalfOpen), count // 2), replace=False))
        tPick += list(rng.choice(tTransitions, min(len(tTransitions), count // 4), replace=False))
    tRest = np.setdiff1d(candidates, tPick)
    tPick += list(rng.choice(tRest, max(0, count - len(set(tPick))), replace=False))
    tUnique = sorted(set(int(tI) for tI in tPick))
    tPick = tUnique[:count] if len(tUnique) > count else tUnique
    return sorted(tPick[::max(1, len(tPick) // count)])[:count]


def select(take: Path, framesPerHand: dict[str, int]) -> None:
    """Draft labels of `framesPerHand[hand]` frames per hand into JAW_KPT_DIR/<take>/<hand>/."""
    tOut = JAW_KPT_DIR / take.name
    tWindows = _freeWindows(json.loads((take / "meta.json").read_text()))
    tRng = np.random.default_rng(SELECT_SEED)
    for tHand, tCount in framesPerHand.items():
        tDir = take / WRIST_DIRS[tHand]
        tTimes = np.load(tDir / "t_ns.npy")
        tInWindow = np.zeros(len(tTimes), bool)
        for tLo, tHi in tWindows:
            tInWindow |= (tTimes >= tLo) & (tTimes <= tHi)
        tCandidates = np.where(tInWindow)[0]
        tPOpen = None
        if (tDir / "umi_jaw.npz").exists():
            tJaw = np.load(tDir / "umi_jaw.npz")
            tPOpen = np.interp(tTimes[tCandidates], tJaw["t_ns"], tJaw["p_open"])
        tPick = _pickFrames(tCandidates, tPOpen, tCount, tRng)
        tFramesDir = tOut / tHand / "frames"
        tFramesDir.mkdir(parents=True, exist_ok=True)
        tLabels = []
        for tI in tPick:
            tCrop = crop(cv2.imread(str(tDir / f"wrist_{tI:05d}.jpg")))
            tFile = f"{tI:05d}.jpg"
            cv2.imwrite(str(tFramesDir / tFile), tCrop, [cv2.IMWRITE_JPEG_QUALITY, FRAME_JPEG_QUALITY])
            tTips = draftTips(tCrop)
            tLabels.append(dict(frame=tI, file=tFile, t_ns=int(tTimes[tI]), tips=tTips, visible=tTips is not None, status="draft",
                                p_open_v003=float(np.interp(tTimes[tI], tJaw["t_ns"], tJaw["p_open"])) if tPOpen is not None else None))
        (tOut / tHand / "labels.json").write_text(json.dumps(dict(take=take.name, hand=tHand, crop=CROP, labels=tLabels), indent=1))
        print(f"{tHand}: {len(tLabels)} 프레임, 초안 있는 것 {sum(tLabel['visible'] for tLabel in tLabels)} -> {tOut / tHand}")


# The labelling page served at / (keys and drag handling client-side; every change is POSTed back at once).
LABEL_PAGE = """<!doctype html><meta charset=utf-8><title>jaw tips</title>
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


def serve(labelDir: Path, port: int) -> None:
    """Browser labelling tool on `port` over labelDir/<hand>/{labels.json, frames/}."""
    from flask import Flask, jsonify, request, send_from_directory
    tRoot = labelDir.resolve()   # absolute: send_from_directory resolves relative paths against the app root
    tApp = Flask(__name__)

    @tApp.get("/")
    def index():
        return LABEL_PAGE

    @tApp.get("/labels/<hand>")
    def getLabels(hand):
        return jsonify(json.loads((tRoot / hand / "labels.json").read_text()))

    @tApp.post("/labels/<hand>/<int:k>")
    def putLabel(hand, k):
        tPath = tRoot / hand / "labels.json"
        tAll = json.loads(tPath.read_text())
        tAll["labels"][k] = request.get_json()
        tPath.write_text(json.dumps(tAll, indent=1))
        return "ok"

    @tApp.get("/img/<hand>/<fn>")
    def image(hand, fn):
        return send_from_directory(tRoot / hand / "frames", fn)

    print(f"라벨링 도구: http://localhost:{port}  (왼손부터, h 로 오른손)", flush=True)
    tApp.run(port=port)


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tSub = tParser.add_subparsers(dest="cmd", required=True)
    tSelect = tSub.add_parser("select")
    tSelect.add_argument("take")
    tSelect.add_argument("--left", type=int, default=DEFAULT_LEFT_FRAMES)
    tSelect.add_argument("--right", type=int, default=DEFAULT_RIGHT_FRAMES)
    tServe = tSub.add_parser("serve")
    tServe.add_argument("dir")
    tServe.add_argument("--port", type=int, default=DEFAULT_PORT)
    return tParser


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    if tArgs.cmd == "select":
        select(Path(tArgs.take), {"left": tArgs.left, "right": tArgs.right})
    else:
        serve(Path(tArgs.dir), tArgs.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
