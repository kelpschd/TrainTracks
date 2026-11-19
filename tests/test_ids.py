"""Tests for TrainTracks.ids (pure pandas; no tracking stack needed)."""
import numpy as np
import pandas as pd

from TrainTracks.ids import (
    add_track_uids,
    assign_canonical_track_ids,
    canonical_track_id,
    track_uid_for_group,
)


def _tracks():
    return pd.DataFrame(
        {
            "particle": [7, 7, 7, 3, 3],
            "frame": [2, 0, 1, 5, 4],  # deliberately unsorted
            "y": [12.0, 10.0, 11.0, 50.5, 50.0],
            "x": [22.0, 20.0, 21.0, 60.5, 60.0],
        }
    )


def test_uid_is_40_char_hex_and_order_independent():
    df = _tracks()
    g = df[df.particle == 7]
    uid = track_uid_for_group(g)
    assert len(uid) == 40 and int(uid, 16) >= 0
    assert uid == track_uid_for_group(g.sample(frac=1, random_state=0))


def test_uid_ignores_particle_renumbering():
    df = _tracks()
    renum = df.assign(particle=df.particle.map({7: 100, 3: 5}))
    a = add_track_uids(df).drop_duplicates("particle").set_index("track_uid").index
    b = add_track_uids(renum).drop_duplicates("particle").set_index("track_uid").index
    assert set(a) == set(b)


def test_uid_changes_when_a_point_moves():
    df = _tracks()
    g = df[df.particle == 7]
    moved = g.copy()
    moved.loc[moved.index[0], "y"] += 0.01
    assert track_uid_for_group(g) != track_uid_for_group(moved)


def test_uid_stable_below_1e6_px_rounding():
    g = _tracks().query("particle == 7")
    nudged = g.copy()
    nudged["y"] += 1e-9
    assert track_uid_for_group(g) == track_uid_for_group(nudged)


def test_add_track_uids_does_not_mutate_and_one_uid_per_particle():
    df = _tracks()
    out = add_track_uids(df)
    assert "track_uid" not in df.columns
    assert out.groupby("particle").track_uid.nunique().eq(1).all()
    assert out.track_uid.nunique() == 2


def test_canonical_ids_fit_63_bits_and_are_deterministic():
    df = _tracks()
    a = assign_canonical_track_ids(df)
    b = assign_canonical_track_ids(df.sample(frac=1, random_state=1))
    assert (a.particle.to_numpy() == b.particle.to_numpy()).all()
    assert a.particle.dtype == np.int64
    assert (a.particle >= 0).all() and (a.particle < 2**63).all()
    assert canonical_track_id(df[df.particle == 3]) in set(a.particle)
