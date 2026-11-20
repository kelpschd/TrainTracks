"""Tests for TrainTracks.tracks_io using fake funtracks node attributes."""
import pandas as pd
import pytest

from TrainTracks.tracks_io import solution_tracks_to_track_df, track_df_from_node_attrs


def _attrs():
    return pd.DataFrame(
        {
            "tracklet_id": [2, 2, 1, 1, 1],
            "t": [1, 0, 2, 0, 1],
            "pos_0": [5.0, 4.0, 9.0, 7.0, 8.0],
            "pos_1": [6.0, 5.0, 3.0, 1.0, 2.0],
        }
    )


def test_columns_sorting_and_dtypes():
    df = track_df_from_node_attrs(_attrs())
    assert list(df.columns) == ["particle", "frame", "y", "x", "track_uid"]
    assert df[["particle", "frame"]].values.tolist() == [[1, 0], [1, 1], [1, 2], [2, 0], [2, 1]]
    assert df.y.dtype == float and df.frame.dtype.kind == "i"
    assert df.groupby("particle").track_uid.nunique().eq(1).all()


def test_missing_attribute_error_names_source():
    with pytest.raises(ValueError, match=r"a\.geff: missing imported attributes \['pos_1'\]"):
        track_df_from_node_attrs(_attrs().drop(columns=["pos_1"]), "a.geff")


def test_solution_tracks_wrapper_uses_graph_attrs():
    class _R:
        def to_pandas(self):
            return _attrs()

    class _G:
        def node_attrs(self, unpack=True):
            return _R()

    class _SG:
        graph = _G()

    pd.testing.assert_frame_equal(solution_tracks_to_track_df(_SG()), track_df_from_node_attrs(_attrs()))
