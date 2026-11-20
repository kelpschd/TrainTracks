"""Calibrating the cull against the validated ground-truth tracks.

The cull flags a track when it is eligible and ``turn_incoherence >= cutoff``
(``TrainTracks.scoring``). A safe cutoff leaves every known-good (ground-truth) track
unflagged. These helpers score GT tracks with the same scorer and summarise how a cutoff
behaves; ``scripts/calibrate_cull_on_gt.py`` runs them over all movies.

Pure numpy/pandas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from TrainTracks import scoring

DEFAULT_SWEEP = (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.57)


def score_gt_movie(gt_df: pd.DataFrame, flow_vec, movie: str, cutoff: float | None = None) -> pd.DataFrame:
    """Score one movie's GT tracks (columns particle, frame, y, x) -> one row per GT track.

    ``flagged`` here uses ``cutoff`` (default ``scoring.TURN_FLAG_RADIANS``) on eligible
    tracks, so the cutoff can be varied without re-scoring.
    """
    cutoff = scoring.TURN_FLAG_RADIANS if cutoff is None else cutoff
    feats = scoring.score_all(gt_df, flow_vec).reset_index()
    feats.insert(0, "movie", movie)
    feats["flagged"] = feats["eligible"] & (feats["turn_incoherence"] >= cutoff)
    return feats


def cutoff_sweep(feats: pd.DataFrame, cutoffs=DEFAULT_SWEEP) -> pd.DataFrame:
    """For each cutoff: how many eligible GT tracks (and what fraction) would be flagged."""
    elig = feats[feats["eligible"]]
    n = len(elig)
    rows = []
    for c in cutoffs:
        k = int((elig["turn_incoherence"] >= c).sum())
        rows.append({"cutoff": c, "n_eligible": n, "n_flagged": k, "frac_flagged": k / n if n else np.nan})
    return pd.DataFrame(rows)


def per_movie_summary(feats: pd.DataFrame) -> pd.DataFrame:
    """One row per movie: GT tracks, eligible, max turn among eligible, flagged at the cutoff used."""
    g = feats.groupby("movie")
    out = pd.DataFrame({
        "gt_tracks": g.size(),
        "eligible": g["eligible"].sum().astype(int),
        "max_turn": feats[feats["eligible"]].groupby("movie")["turn_incoherence"].max(),
        "flagged": g["flagged"].sum().astype(int),
    })
    return out.reset_index()
