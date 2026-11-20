"""Tests for TrainTracks.calibration and the repo's ground_truth/ folder."""
from pathlib import Path

import pandas as pd

from TrainTracks import calibration

GT_DIR = Path(__file__).resolve().parents[1] / "ground_truth"


def _zero_flow(t, y, x):
    return 0.0, 0.0


def _gt():
    straight = [(1, f, 0.0, 3.0 * f) for f in range(8)]
    zigzag_pts = [(0, 0), (0, 3), (3, 3), (3, 6), (6, 6), (6, 9), (9, 9), (9, 12)]
    zigzag = [(2, f, y, x) for f, (y, x) in enumerate(zigzag_pts)]
    short = [(3, f, 0.0, 3.0 * f) for f in range(3)]
    return pd.DataFrame(straight + zigzag + short, columns=["particle", "frame", "y", "x"])


def test_score_and_sweep_and_summary():
    feats = calibration.score_gt_movie(_gt(), _zero_flow, "m1", cutoff=1.4)
    assert set(feats["movie"]) == {"m1"} and len(feats) == 3
    assert feats.set_index("particle")["flagged"].to_dict() == {1: False, 2: True, 3: False}
    sweep = calibration.cutoff_sweep(feats, cutoffs=(1.0, 1.4, 1.6)).set_index("cutoff")
    assert sweep.loc[1.0, "n_flagged"] == 1 and sweep.loc[1.6, "n_flagged"] == 0
    assert (sweep["n_eligible"] == 2).all()  # the 3-point track is not eligible
    s = calibration.per_movie_summary(feats).iloc[0]
    assert (s.gt_tracks, s.eligible, s.flagged) == (3, 2, 1)


def test_cutoff_can_be_changed_without_rescoring():
    f_hi = calibration.score_gt_movie(_gt(), _zero_flow, "m1", cutoff=1.6)
    assert not f_hi["flagged"].any()


def test_repo_ground_truth_is_complete_and_clean():
    files = sorted(GT_DIR.glob("*_valid_tracks.csv"))
    assert len(files) == 32
    n_tracks = n_rows = 0
    for f in files:
        d = pd.read_csv(f)
        assert list(d.columns) == ["frame", "x", "y", "particle"], f.name
        assert not d.isna().any().any() and not d.duplicated(["particle", "frame"]).any(), f.name
        n_tracks += d["particle"].nunique()
        n_rows += len(d)
    assert (n_tracks, n_rows) == (294, 4040)
