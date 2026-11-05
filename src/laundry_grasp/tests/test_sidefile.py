"""Side file: zero-order hold, frame-0 check, episode/key order and the written npy/json (parquet reading replaced by a frame table)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from laundry_grasp import demos, sidefile
from laundry_grasp.pgrasp import VERSION
from laundry_grasp.tests import synthetic

NS = 1_000_000_000


def test_zero_order_hold_is_causal():
    tTicks = np.array([0, 10, 20, 30], np.int64) * NS // 100
    tVals = np.array([0.1, 0.2, 0.3, 0.4], np.float32)
    tFrames = np.array([0, 5, 10, 19, 29, 30, 30 + 6], np.int64) * NS // 100
    assert sidefile.holdValues(tTicks, tVals, tFrames).tolist() == pytest.approx([0.1, 0.1, 0.2, 0.2, 0.3, 0.4, 0.4])


def test_hold_rejects_frames_outside_the_series():
    tTicks = np.array([100, 200], np.int64)
    with pytest.raises(sidefile.SideFileError):
        sidefile.holdValues(tTicks, np.zeros(2), np.array([50], np.int64))
    with pytest.raises(sidefile.SideFileError):
        sidefile.holdValues(tTicks, np.zeros(2), np.array([200 + NS], np.int64))     # > one 15 Hz tick after the last


def test_frame_zero_check():
    assert sidefile.checkFrameZero("k", "wrist", NS + NS // 10, NS) == pytest.approx(0.1)
    with pytest.raises(sidefile.SideFileError, match="episode/key order mismatch"):
        sidefile.checkFrameZero("k", "wrist", NS + NS // 2, NS)


def _set(root, synthTakes, keys, frameShift=0):
    """an aggregated set of the synthetic take's demos: frames = the wrist frames from the demo window start, 2 per tick"""
    tSet = root / "set"
    (tSet / "meta").mkdir(parents=True)
    (tSet / "meta/split.json").write_text(json.dumps({"episodes": {"train": keys[:1], "val": keys[1:], "test": []}}))
    tWrist = np.load(synthTakes / synthetic.TAKE / "wrist/t_ns.npy")
    tRows, tIndex = [], 0
    for tE, tKey in enumerate(keys):
        tT0, _ = demos.demoWindow(*demos.parseDemoKey(tKey))
        tFirst = int(np.searchsorted(tWrist, tT0)) + frameShift
        for tF in range(20):
            tRows.append(dict(index=tIndex, episode_index=tE, frame_index=tF, src_index=np.array([tIndex, tFirst + tF, tFirst + tF])))
            tIndex += 1
    (tSet / "meta/info.json").write_text(json.dumps(dict(total_frames=tIndex)))
    return tSet, pd.DataFrame(tRows)


def _pgraspDir(root, keys):
    tPg = root / "pg"
    (tPg / "demos").mkdir(parents=True)
    for tKey in keys:
        tT0, tT1 = demos.demoWindow(*demos.parseDemoKey(tKey))
        tTs = np.arange(tT0 - NS, tT1 + NS, NS / 15).astype(np.int64)
        np.savez(tPg / "demos" / f"{tKey.replace('/', '_')}.npz", version=VERSION, t_ns=tTs,
                 L_p=np.linspace(0, 1, len(tTs), dtype=np.float32), R_p=np.full(len(tTs), 0.25, np.float32))
    return tPg


def test_build_and_write(tmp_path, synthTakes, monkeypatch, capsys):
    tKeys = [f"{synthetic.TAKE}/demo00", f"{synthetic.TAKE}/demo01"]
    tSet, tDf = _set(tmp_path, synthTakes, tKeys)
    tPg = _pgraspDir(tmp_path, tKeys)
    monkeypatch.setattr(sidefile, "readFrames", lambda setDir: tDf)
    assert sidefile.main([str(tSet), str(tPg), "--check-only"]) == 0
    assert not list((tSet / "meta").glob("pgrasp_*"))
    assert sidefile.main([str(tSet), str(tPg)]) == 0
    tP = np.load(tSet / f"meta/pgrasp_{VERSION}.npy")
    assert tP.shape == (40, 2) and tP.dtype == np.float32
    assert np.all(tP[:, 1] == 0.25) and np.all(np.diff(tP[:20, 0]) >= 0)
    tMeta = json.loads((tSet / f"meta/pgrasp_{VERSION}.json").read_text())
    assert tMeta["columns"] == ["L", "R"] and tMeta["shape"] == [40, 2] and tMeta["source"] == str(tPg)
    assert "order verified" in capsys.readouterr().out


def test_episode_key_mismatch_is_caught(tmp_path, synthTakes, monkeypatch, capsys):
    tKeys = [f"{synthetic.TAKE}/demo00", f"{synthetic.TAKE}/demo01"]
    tSet, tDf = _set(tmp_path, synthTakes, tKeys, frameShift=30)        # frame 0 one second after the window start
    monkeypatch.setattr(sidefile, "readFrames", lambda setDir: tDf)
    assert sidefile.main([str(tSet), str(tmp_path), "--check-only"]) == 1
    assert "episode/key order mismatch" in capsys.readouterr().err
