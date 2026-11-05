"""v2 jaw segmentation, close intervals, demo windows and the v2 event table on synthetic jaw series."""
from __future__ import annotations

import numpy as np

from laundry_grasp import demos, events
from laundry_grasp.tests import synthetic

NS = 1_000_000_000


def _series(values, hz=100.0):
    tT = (np.arange(len(values)) * NS / hz).astype(np.int64)
    return tT, np.asarray(values, np.float32)


def test_segmentation_hysteresis_thresholds_and_dwell():
    # open, then closed (< 0.15), a partial re-open to 0.25 (inside the band: no change), closed, a 50 ms spike to 0.9 (shorter
    # than the 0.1 s dwell: no change), a real re-open for 0.3 s
    tG = [1.0] * 50 + [0.1] * 50 + [0.25] * 50 + [0.1] * 50 + [0.9] * 5 + [0.1] * 45 + [0.9] * 30 + [0.1] * 30
    tT, tG = _series(tG)
    tState, tEvents = events.segmentJaw(tT, tG)
    assert [tK for _, tK in tEvents] == ["close", "open", "close"]
    # state is back-filled from the first sample of the change
    assert tEvents[0][0] == tT[50] and tState[50] == 0 and tState[49] == 1
    assert tEvents[1][0] == tT[250]
    assert tState[150:200].tolist() == [0] * 50          # 0.25 sits between LO and HI: stays closed
    assert tState[200:205].tolist() == [0] * 5           # spike shorter than the dwell


def test_v1_thresholds_would_have_split_the_partial_reopen():
    # the 0.48 partial re-open that v1 (0.55 / 0.35) hid is a real open for v2 (0.30 / 0.15)
    tT, tG = _series([1.0] * 30 + [0.05] * 60 + [0.48] * 60 + [0.05] * 60 + [1.0] * 30)
    _, tEvents = events.segmentJaw(tT, tG)
    assert [tK for _, tK in tEvents] == ["close", "open", "close", "open"]


def test_close_intervals_overlap_and_rest():
    tEvents = [(10, "close"), (20, "open"), (30, "close"), (40, "close"), (50, "open"), (60, "close")]
    assert events.closeIntervals(tEvents, 99) == [(10, 20), (30, 50), (40, 50), (60, 99)]
    assert events.overlapsWindow(10, 20, 20, 30) and not events.overlapsWindow(10, 19, 20, 30)
    assert not events.overlapsWindow(31, 40, 20, 30)
    tT0, tT1 = 10 * NS, 20 * NS
    assert events.isRest(tT0 + int(0.4 * NS), tT0 + 2 * NS, tT0, tT1)       # closed within 0.5 s of the start
    assert events.isRest(tT0 + 2 * NS, tT1 - int(0.4 * NS), tT0, tT1)       # still closed 0.4 s before the end
    assert not events.isRest(tT0 + NS, tT0 + 2 * NS, tT0, tT1)


def test_demo_window_protocol_and_idle_trim(synthTakes):
    # demo 0 (protocol t0 0.5 s, 5 s) has no trim row; demo 1 (t0 6 s) is trimmed to [0.5, 5.0] s of the demo
    tA = synthetic.T_START_NS + int(0.5 * NS)
    assert demos.demoWindow(synthetic.TAKE, 0) == (tA, tA + 5 * NS)
    tB = synthetic.T_START_NS + 6 * NS
    assert demos.demoWindow(synthetic.TAKE, 1) == (tB + int(0.5 * NS), tB + 5 * NS)


def test_events_v2_table(synthTakes):
    tRows = events.eventsV2([f"{synthetic.TAKE}/demo00", f"{synthetic.TAKE}/demo01"])
    tR = [tRow for tRow in tRows if tRow["hand"] == "R" and tRow["demo"].endswith("demo00")]
    # right jaw: closed 1-3 s, 0.2 for 0.5 s (inside the v2 band? no: 0.2 > 0.15 and < 0.30 -> stays closed), closed until 5 s
    assert [(tRow["event"], tRow["closed_s"]) for tRow in tR] == [("R1", 4.0)]
    assert tR[0]["id"] == f"{synthetic.TAKE}_d01_R1" and tR[0]["cam"] == "wrist"
    assert list(tRows[0]) == ["id", "demo", "take", "cam", "hand", "event", "t_close_ns", "t_open_ns", "closed_s", "rest"]
    tL = [tRow for tRow in tRows if tRow["hand"] == "L"]
    assert {tRow["demo"] for tRow in tL} == {f"{synthetic.TAKE}/demo00"}   # the left close (2-6 s) ends before demo 1 starts
    assert all(tRow["rest"] in (0, 1) for tRow in tRows)


def test_demo_names():
    assert demos.parseDemoKey("20261104_150704/demo19") == ("20261104_150704", 19)
    assert demos.videoName("20261104_150704", 19) == "20261104_150704_d20"
    assert demos.parseVideoName("20261104_150704_d20_H0E2U19") == ("20261104_150704", 19)
    assert demos.handCam("L3") == "wrist_left" and demos.handCam("R") == "wrist"
