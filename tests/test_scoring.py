"""Pin the turn_incoherence definition and the clean-track rule (TrainTracks.scoring)."""
import numpy as np
import pandas as pd

from TrainTracks import scoring
from TrainTracks.scoring import score_all, score_track, select_clean_tracks


def _zero_flow(t, y, x):
    return 0.0, 0.0


def _track(points, pid=1, t0=0):
    return pd.DataFrame(
        [(pid, t0 + i, y, x) for i, (y, x) in enumerate(points)],
        columns=["particle", "frame", "y", "x"],
    )


STRAIGHT = [(0, 3 * i) for i in range(8)]
ZIGZAG = [(0, 0), (0, 3), (3, 3), (3, 6), (6, 6), (6, 9), (9, 9), (9, 12)]  # 90 degree turn every step
PAUSEY = [(0, 0), (0, 3), (0, 3.1), (0, 6), (0, 6.1), (0, 9), (0, 9.1), (0, 12)]  # straight with pauses


def test_cutoffs_are_the_calibrated_values():
    assert scoring.TURN_FLAG_RADIANS == 1.4
    assert scoring.MIN_SPAN_FOR_SCORE == 5 and scoring.MIN_SPAN_KEEP == 5
    assert scoring.PAUSE_SPEED_PX_PER_FRAME == 0.5


def test_straight_track_has_zero_turn():
    f = score_track(_track(STRAIGHT), _zero_flow)
    assert f["turn_incoherence"] < 1e-6


def test_right_angle_zigzag_is_half_pi():
    f = score_track(_track(ZIGZAG), _zero_flow)
    assert abs(f["turn_incoherence"] - np.pi / 2) < 1e-6


def test_pauses_are_neutral_not_turns():
    """A straight track with pauses must NOT look incoherent (the all-steps definition
    this replaced would count the near-zero pause steps as random turns)."""
    f = score_track(_track(PAUSEY), _zero_flow)
    assert f["turn_incoherence"] < 1e-6
    assert f["n_pause_steps"] == 3 and f["n_motion_steps"] == 4


def test_score_all_flags_only_the_zigzag_and_adds_track_uid():
    df = pd.concat([_track(STRAIGHT, 1), _track(ZIGZAG, 2), _track(PAUSEY, 3)])
    feats = score_all(df, _zero_flow)
    assert feats.loc[2, "flagged"] and not feats.loc[1, "flagged"] and not feats.loc[3, "flagged"]
    assert feats["eligible"].all()
    assert feats["track_uid"].str.len().eq(40).all() and feats["track_uid"].is_unique


def test_short_tracks_are_not_eligible_or_flagged():
    short = _track(ZIGZAG[:4], 9)  # span 4 < MIN_SPAN_FOR_SCORE
    feats = score_all(short, _zero_flow)
    assert not feats.loc[9, "eligible"] and not feats.loc[9, "flagged"]


def test_select_clean_tracks_drops_flagged_and_short():
    df = pd.concat([_track(STRAIGHT, 1), _track(ZIGZAG, 2), _track(STRAIGHT[:3], 3)])
    feats = score_all(df, _zero_flow)
    keep, flagged, short = select_clean_tracks(df, feats)
    assert flagged == {2} and short == {3}
    assert sorted(keep.particle.unique()) == [1]
    assert list(keep.columns) == ["particle", "frame", "y", "x"]
