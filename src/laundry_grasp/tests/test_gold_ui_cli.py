"""Labeling page routes, event selection, clip span, and the command lines (help, no work at import)."""
from __future__ import annotations

import csv
import importlib
import json
import threading
import urllib.error
import urllib.request

import pytest

from laundry_grasp import gold_ui, video

MODULES = ("events", "generate", "sidefile", "cv", "render", "gold_ui", "rule_v0")


def _eventsCsv(path, rows):
    with open(path, "w", newline="") as tFh:
        tW = csv.DictWriter(tFh, fieldnames=["video", "event", "t_close_s", "closed_s", "label", "rest"])
        tW.writeheader()
        tW.writerows(rows)


def test_selection_is_seeded_and_skips_rest(tmp_path):
    tRows = [dict(video="20990101_120000_d01_H0E0U0", event=f"L{tI}", t_close_s=tI, closed_s=1.0,
                  label=("held", "empty", "unknown")[tI % 3], rest=int(tI % 5 == 0)) for tI in range(1, 40)]
    _eventsCsv(tmp_path / "events.csv", tRows)
    tA = gold_ui.selectEvents(tmp_path / "events.csv", takeAll=False)
    assert tA == gold_ui.selectEvents(tmp_path / "events.csv", takeAll=False)
    assert {tE["id"] for tE in tA} <= {f"20990101_120000_d01_L{tI}" for tI in range(1, 40) if tI % 5}
    assert len(gold_ui.selectEvents(tmp_path / "events.csv", takeAll=True)) == sum(1 for tI in range(1, 40) if tI % 5)


def test_single_close_span():
    tEvents = [(0, "close"), (10 * 10**9, "open"), (12 * 10**9, "close"), (14 * 10**9, "open"), (14_100_000_000, "close")]
    # start = max(close - 1 s, previous open + 0.15 s) = 11 s; end = min(open + 1 s, next close - 0.15 s, start + 7 s) = 13.95 s
    assert video.singleCloseSpan(tEvents, 12 * 10**9, 14 * 10**9, 1.0, 0.15, 7) == (11e9, 13.95e9)
    tA, tB = video.singleCloseSpan([(0, "close")], 0, 20 * 10**9, 1.0, 0.15, 7)
    assert (tA, tB) == (-1e9, 6e9)                       # capped at 7 s


@pytest.fixture
def server(tmp_path):
    tRound = gold_ui.LabelRound(tmp_path)
    tRound.mClips.mkdir()
    (tRound.mClips / "x_L1.mp4").write_bytes(b"mp4")
    tEvents = [dict(id="x_L1", video="x_d01", event="L1", t_close_s=1.0, closed_s=2.0)]
    tServer = gold_ui.http.server.ThreadingHTTPServer(("127.0.0.1", 0), gold_ui.makeHandler(tRound, tEvents))
    tThread = threading.Thread(target=tServer.serve_forever, daemon=True)
    tThread.start()
    yield f"http://127.0.0.1:{tServer.server_address[1]}", tRound
    tServer.shutdown()


def test_routes(server, monkeypatch):
    tBase, tRound = server
    monkeypatch.setenv("GOLD_L3", "loose")
    tPage = urllib.request.urlopen(tBase + "/").read().decode()
    assert "lab('loose')" in tPage and "__L3__" not in tPage and "애매함" in tPage
    assert json.loads(urllib.request.urlopen(tBase + "/api/state").read()) == dict(events=[dict(
        id="x_L1", video="x_d01", event="L1", t_close_s=1.0, closed_s=2.0)], gold={})
    tReq = urllib.request.Request(tBase + "/api/label", method="POST",
                                  data=json.dumps(dict(id="x_L1", video="x_d01", event="L1", gold="held")).encode())
    assert urllib.request.urlopen(tReq).read() == b"{}"
    assert json.loads(urllib.request.urlopen(tBase + "/api/state").read())["gold"] == {"x_L1": "held"}
    assert tRound.mGold.read_text().splitlines() == ["id,video,event,gold", "x_L1,x_d01,L1,held"]
    assert urllib.request.urlopen(tBase + "/clips/x_L1.mp4").read() == b"mp4"
    with pytest.raises(urllib.error.HTTPError) as tErr:
        urllib.request.urlopen(tBase + "/clips/nothing.mp4")
    assert tErr.value.code == 404


TORCH_MODULES = ("generate", "sidefile", "cv", "render")   # import the ResNet-18 recipe (torch)


@pytest.mark.parametrize("name", MODULES)
def test_cli_help(name, capsys):
    if name in TORCH_MODULES:
        pytest.importorskip("torch")
    tModule = importlib.import_module(f"laundry_grasp.{name}")
    with pytest.raises(SystemExit) as tExit:
        tModule.main(["--help"])
    assert tExit.value.code == 0
    assert "usage" in capsys.readouterr().out
