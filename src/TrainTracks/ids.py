"""Deterministic track identifiers.

funtracks renumbers ``tracklet_id`` every time a GEFF is imported, so an integer
``particle`` is only meaningful within one import. Anything that needs to follow a
track across files (tracks.geff, flow_scores.csv, tracks_clean.geff, cotracking
tables) must join on ``track_uid``: a SHA-1 hex digest of the track's
frame/position sequence. It does not depend on import order.

All functions are pure pandas (no funtracks / zarr imports) so they are cheap to
import and easy to test.
"""

from __future__ import annotations

import hashlib

import pandas as pd

# Positions are rounded to 1e-6 px before hashing: far below localisation precision,
# but removes incidental float-serialisation differences between files.
_POSITION_SCALE = 1_000_000


def _payload(group: pd.DataFrame) -> list[tuple[int, int, int]]:
    g = group.sort_values("frame")[["frame", "y", "x"]]
    return [
        (
            int(r.frame),
            int(round(float(r.y) * _POSITION_SCALE)),
            int(round(float(r.x) * _POSITION_SCALE)),
        )
        for r in g.itertuples(index=False)
    ]


def track_uid_for_group(group: pd.DataFrame) -> str:
    """SHA-1 hex digest (40 chars) of one track's (frame, y, x) sequence."""
    return hashlib.sha1(repr(_payload(group)).encode("utf-8")).hexdigest()


def add_track_uids(track_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``track_df`` with a ``track_uid`` column.

    ``track_df`` needs columns ``particle, frame, y, x``.
    """
    df = track_df.copy()
    uid_map = {
        int(pid): track_uid_for_group(group)
        for pid, group in df.groupby("particle", sort=False)
    }
    df["track_uid"] = df["particle"].astype(int).map(uid_map)
    return df


def canonical_track_id(group: pd.DataFrame) -> int:
    """Deterministic 63-bit integer ID for a track (first 8 bytes of the SHA-1).

    NOTE: 63-bit integers are not exactly representable as float64, so IDs that pass
    through napari or pandas float columns can collide. Kept unchanged here so that
    consolidating code does not change any output; replaced in the track-ID fix step.
    """
    digest = hashlib.sha1(repr(_payload(group)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def assign_canonical_track_ids(track_df: pd.DataFrame) -> pd.DataFrame:
    """Replace ``particle`` with deterministic geometry-based integer IDs."""
    df = track_df.copy().sort_values(["particle", "frame"]).reset_index(drop=True)
    id_map = {
        int(old): canonical_track_id(group)
        for old, group in df.groupby("particle", sort=False)
    }
    df["particle"] = df["particle"].astype(int).map(id_map).astype("int64")
    return df.sort_values(["particle", "frame"]).reset_index(drop=True)
