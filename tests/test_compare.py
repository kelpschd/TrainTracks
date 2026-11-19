"""Tests for TrainTracks.compare."""
import numpy as np
import pandas as pd

from TrainTracks import compare


def test_blobs():
    a = pd.DataFrame({"t": [0, 1], "y": [5, 6], "x": [7, 8]})
    assert compare.compare_blobs(a, a.copy()).ok
    assert not compare.compare_blobs(a, a.iloc[::-1]).ok


def test_flow_bit_identical_and_different():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 4, 4, 2)).astype("float32")
    assert compare.compare_flow(a, a.copy()).ok
    b = a.copy(); b[1, 2, 2, 0] += 1e-6
    r = compare.compare_flow(a, b)
    assert r.ok and "1 values differ" in r.detail          # tiny rounding-level change is tolerated, but reported
    b[1, 2, 2, 0] += 1.0
    assert not compare.compare_flow(a, b).ok
    assert not compare.compare_flow(a, a[:2]).ok


def test_tracks_compared_by_uid_not_particle():
    a = pd.DataFrame({"particle": [1, 1, 2], "track_uid": ["u", "u", "v"]})
    b = pd.DataFrame({"particle": [9, 9, 4], "track_uid": ["v", "u", "u"]})  # renumbered
    assert compare.compare_tracks("t", a, a.assign(particle=[5, 5, 6])).ok
    assert not compare.compare_tracks("t", a, b.iloc[:2]).ok


def test_scores_match_on_uid_and_ignore_particle():
    a = pd.DataFrame({"particle": [1, 2], "track_uid": ["u", "v"], "span": [5, 9], "turn_incoherence": [0.5, 1.5],
                      "flagged": [False, True]})
    b = a.assign(particle=[7, 8]).iloc[::-1]
    assert compare.compare_scores(a, b).ok
    c = b.copy(); c["turn_incoherence"] = c["turn_incoherence"] + 0.1
    r = compare.compare_scores(a, c)
    assert not r.ok and "turn_incoherence" in r.detail
    assert not compare.compare_scores(a, a.assign(track_uid=["u", "z"])).ok


def test_metadata_flags_param_changes_but_only_notes_posthoc():
    a = {"n_tracks": 3, "detection_params": {"threshold": 1}, "posthoc_results": {"n_tracks_kept": 253}}
    b = {"n_tracks": 3, "detection_params": {"threshold": 1}, "posthoc_results": {"n_tracks_kept": 260}}
    r = compare.compare_metadata(a, b)
    assert r.ok and "posthoc_results differ" in r.detail
    assert not compare.compare_metadata(a, {**b, "detection_params": {"threshold": 2}}).ok
