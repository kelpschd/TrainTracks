"""Tests for the pure-pandas parts of TrainTracks.detection (no skimage needed)."""
import pandas as pd

from TrainTracks.detection import blobs_df_to_tyx, filter_dense_blobs


def _blobs(points):
    return pd.DataFrame(
        [{"frame": f, "x": x, "y": y, "size": 2.0} for f, x, y in points]
    )


def test_dense_bin_is_cleared_across_all_frames():
    # 5 detections in the same 5x5 px bin, on different frames -> all removed (>= 5)
    dense = [(f, 12.0, 12.0) for f in range(5)]
    sparse = [(0, 100.0, 100.0), (1, 101.0, 101.0)]
    out = filter_dense_blobs(_blobs(dense + sparse), bin_size=5, blob_filter=5)
    assert len(out) == 2 and set(out.x) == {100.0, 101.0}


def test_four_in_a_bin_survive_with_filter_five():
    out = filter_dense_blobs(_blobs([(f, 12.0, 12.0) for f in range(4)]), bin_size=5, blob_filter=5)
    assert len(out) == 4


def test_floor_binning_splits_neighbouring_bins():
    # x = 9.9 is in bin 5, x = 10.0 is in bin 10 -> 3 + 3 detections, neither bin reaches 5
    pts = [(f, 9.9, 7.0) for f in range(3)] + [(f, 10.0, 7.0) for f in range(3)]
    assert len(filter_dense_blobs(_blobs(pts), bin_size=5, blob_filter=5)) == 6


def test_filter_empty_input_passthrough():
    empty = pd.DataFrame()
    assert filter_dense_blobs(empty, bin_size=5, blob_filter=5) is empty


def test_blobs_df_to_tyx_rounds_and_orders_columns():
    out = blobs_df_to_tyx(_blobs([(3, 10.6, 20.4)]))
    assert list(out.columns) == ["t", "y", "x"]
    assert out.iloc[0].tolist() == [3, 20, 11]


def test_blobs_df_to_tyx_empty():
    out = blobs_df_to_tyx(pd.DataFrame())
    assert list(out.columns) == ["t", "y", "x"] and len(out) == 0
