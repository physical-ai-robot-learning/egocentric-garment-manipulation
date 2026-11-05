"""Gold labeling page for grasp events (held / empty / uncertain), BLIND to any model prediction (was grasp_gold_ui.py).

Builds one short clip per selected close event of DIR/events.csv (a rule_v0 output or a mined round; that hand's wrist cam,
close - 1 s .. open + 1 s, cut at the previous open / next close so only THIS close is in it, at most 7 s, 15 fps; a bar shows the
jaw CLOSED state from the motor only) and serves a page on 127.0.0.1:PORT: keys 1 held, 2 empty, 3 uncertain, 0 skip, <- back,
space replay. Labels -> DIR/gold_events.csv (latest per event); the selection -> DIR/gold_selection.json.

Selection: non-rest events; by default every v0 'empty' + 65 random 'held' + 75 random 'unknown' (seed 1105), shuffled.
Environment: GOLD_ALL=1 takes every event of events.csv (rounds 2+: events.csv is already the sample); GOLD_L3 / GOLD_T3 set the
label / button text of key 3 (default uncertain / 애매함; round 4 used it for 'loose').

    python -m laundry_grasp.gold_ui DIR [PORT]
"""
from __future__ import annotations

import argparse
import csv
import http.server
import json
import os
import random
import sys
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .demos import camDir, demoWindow, handCam, nearestFrame, parseVideoName, wristImage
from .video import CLOSED_BGR, OPEN_BGR, RAW_SUFFIX, VIDEO_FPS, encodeH264, rawWriter, readFrame, singleCloseSpan

DEFAULT_PORT = 8098
N_HELD, N_UNKNOWN, SEED = 65, 75, 1105
CLIP_W, CLIP_H, CLIP_BAR_H = 640, 360, 40
CLIP_PAD_S, CLIP_GAP_S, CLIP_MAX_S = 1.0, 0.15, 7
CRF = 24
PROGRESS_EVERY = 20
GOLD_FIELDS = ["id", "video", "event", "gold"]
PAGE = Path(__file__).with_name("gold_ui_page.html")
HOST = "127.0.0.1"
NS_PER_S = 1e9


@dataclass(frozen=True)
class LabelRound:
    """one labeling round folder: events.csv in, clips/ + gold_selection.json + gold_events.csv out"""
    mDir: Path

    @property
    def mClips(self) -> Path:
        return self.mDir / "clips"

    @property
    def mGold(self) -> Path:
        return self.mDir / "gold_events.csv"

    def loadGold(self) -> dict:
        if not self.mGold.exists():
            return {}
        with open(self.mGold) as tFh:
            return {tR["id"]: tR for tR in csv.DictReader(tFh)}

    def saveGold(self, gold: dict) -> None:
        with open(self.mGold, "w", newline="") as tFh:
            tWriter = csv.DictWriter(tFh, fieldnames=GOLD_FIELDS)
            tWriter.writeheader()
            tWriter.writerows(gold.values())


def selectEvents(eventsCsv: Path, takeAll: bool) -> list[dict]:
    """the events to label (see module docstring), as {id, video, event, t_close_s, closed_s}"""
    with open(eventsCsv) as tFh:
        tEvents = [tR for tR in csv.DictReader(tFh) if tR["rest"] == "0"]
    tRnd = random.Random(SEED)
    if takeAll:
        tSel = list(tEvents)
    else:
        tBy = {tK: [tR for tR in tEvents if tR["label"] == tK] for tK in ("held", "empty", "unknown")}
        tSel = (tBy["empty"] + tRnd.sample(tBy["held"], min(N_HELD, len(tBy["held"])))
                + tRnd.sample(tBy["unknown"], min(N_UNKNOWN, len(tBy["unknown"]))))
    tRnd.shuffle(tSel)
    return [dict(id=f"{tR['video'].rsplit('_', 1)[0]}_{tR['event']}", video=tR["video"], event=tR["event"],
                 t_close_s=float(tR["t_close_s"]), closed_s=float(tR["closed_s"])) for tR in tSel]


def makeClip(event: dict, clipsDir: Path) -> None:
    """the single-close review clip of one event (kept when it exists)"""
    tOut = clipsDir / f"{event['id']}.mp4"
    if tOut.exists():
        return
    tTake, tDemo = parseVideoName(event["video"])
    tT0, _ = demoWindow(tTake, tDemo)
    tDir = camDir(tTake, handCam(event["event"]))
    tWrist = np.load(tDir / "t_ns.npy")
    tJaw = np.load(tDir / "umi_jaw.npz")
    tEvents = list(zip(tJaw["ev_t"].tolist(), tJaw["ev_kind"].tolist()))
    tGuess = tT0 + event["t_close_s"] * NS_PER_S
    tClose = min((tT for tT, tK in tEvents if tK == "close"), key=lambda tT: abs(tT - tGuess))
    tOpen = tClose + event["closed_s"] * NS_PER_S
    tA, tB = singleCloseSpan(tEvents, tClose, tOpen, CLIP_PAD_S, CLIP_GAP_S, CLIP_MAX_S)
    tRaw = tOut.with_suffix(RAW_SUFFIX)
    tWriter = rawWriter(tRaw, (CLIP_W, CLIP_H + CLIP_BAR_H))
    for tT in np.arange(tA, tB, NS_PER_S / VIDEO_FPS):
        tImage = readFrame(wristImage(tDir, nearestFrame(tWrist, tT)), (CLIP_W, CLIP_H))
        tClosed = tClose <= tT < tOpen
        tBar = np.full((CLIP_BAR_H, CLIP_W, 3), CLOSED_BGR if tClosed else OPEN_BGR, np.uint8)
        cv2.putText(tBar, ("JAW CLOSED" if tClosed else "jaw open") + f"   {(tT - tA) / NS_PER_S:4.1f}s", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0) if tClosed else (255, 255, 255), 2)
        tWriter.write(np.vstack([tImage, tBar]))
    tWriter.release()
    encodeH264(tRaw, tOut, CRF, faststart=True)


def pageHtml() -> bytes:
    tPage = PAGE.read_text()
    return tPage.replace("__L3__", os.environ.get("GOLD_L3", "uncertain")).replace("__T3__", os.environ.get("GOLD_T3", "애매함")).encode()


class GoldHandler(http.server.BaseHTTPRequestHandler):
    """/ page, /api/state (events + labels so far), /clips/<id>.mp4, POST /api/label; bound to a round by makeHandler"""
    mRound: LabelRound
    mEvents: list

    def log_message(self, *args) -> None:
        pass

    def _send(self, body: bytes, contentType: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", contentType)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        tPath = urllib.parse.urlparse(self.path).path
        if tPath == "/":
            return self._send(pageHtml(), "text/html; charset=utf-8")
        if tPath == "/api/state":
            tState = dict(events=self.mEvents, gold={tK: tV["gold"] for tK, tV in self.mRound.loadGold().items()})
            return self._send(json.dumps(tState).encode(), "application/json")
        if tPath.startswith("/clips/"):
            tFile = self.mRound.mClips / Path(tPath).name
            if tFile.exists():
                return self._send(tFile.read_bytes(), "video/mp4")
        self._send(b"not found", "text/plain", 404)

    def do_POST(self) -> None:
        tReq = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        tGold = self.mRound.loadGold()
        tGold[tReq["id"]] = {tK: tReq[tK] for tK in GOLD_FIELDS}
        self.mRound.saveGold(tGold)
        self._send(b"{}", "application/json")


def makeHandler(labelRound: LabelRound, events: list) -> type:
    return type("BoundGoldHandler", (GoldHandler,), dict(mRound=labelRound, mEvents=events))


def buildParser() -> argparse.ArgumentParser:
    tParser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    tParser.add_argument("dir", help="round folder with events.csv")
    tParser.add_argument("port", nargs="?", type=int, default=DEFAULT_PORT)
    return tParser


def main(argv: list[str] | None = None) -> int:
    tArgs = buildParser().parse_args(argv)
    tRound = LabelRound(Path(tArgs.dir).expanduser())
    tRound.mClips.mkdir(exist_ok=True)
    tEvents = selectEvents(tRound.mDir / "events.csv", os.environ.get("GOLD_ALL") == "1")
    for tN, tEvent in enumerate(tEvents):
        makeClip(tEvent, tRound.mClips)
        if tN % PROGRESS_EVERY == 0:
            print(f"clips {tN}/{len(tEvents)}", flush=True)
    with open(tRound.mDir / "gold_selection.json", "w") as tFh:
        json.dump(tEvents, tFh, indent=1)
    print(f"READY http://{HOST}:{tArgs.port}  ({len(tEvents)} events)", flush=True)
    http.server.ThreadingHTTPServer((HOST, tArgs.port), makeHandler(tRound, tEvents)).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
