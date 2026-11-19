"""End-to-end smoke test: synthetic blobs + flow -> candidate graph -> motile solve -> tracks.

Needs networkx, scikit-image, scipy and motile (skipped if any is missing).
"""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("networkx")
pytest.importorskip("skimage")
pytest.importorskip("scipy")
pytest.importorskip("motile")

from TrainTracks.ids import add_track_uids  # noqa: E402
from TrainTracks.tracking import (  # noqa: E402
    annotate_solution_track_ids,
    run_tracking,
    solution_to_track_df,
    track_metrics,
)

PARAMS = dict(
    dilation_radius=2,
    max_edge_distance=30.0,
    max_gap=2,
    gap_penalty=10.0,
    node_selected_weight=1.0,
    edge_selected_weight=3.0,
    edge_flow_attribute="flow_offset",
    edge_selected_constant=-72.0,
    node_appear_constant=2.0,
    max_parents=1,
    max_children=1,
)
T, Y, X = 6, 64, 64


def _movie(tracks):
    """tracks: list of (y0, x0); each moves +3 px in x per frame. Flow = the true motion (dx=3, dy=0)."""
    rows = [(t, y0, x0 + 3 * t) for (y0, x0) in tracks for t in range(T)]
    blobs = pd.DataFrame(rows, columns=["t", "y", "x"])
    flow = np.zeros((T - 1, Y, X, 2), dtype=np.float32)
    flow[..., 0] = 3.0  # OpenCV order: [..., 0] = dx, [..., 1] = dy
    return blobs, flow


def test_one_particle_becomes_one_track():
    blobs, flow = _movie([(20, 10)])
    sol, stats = run_tracking(blobs, flow, T, PARAMS)
    m = track_metrics(sol, 3)
    assert stats["n_candidate_nodes"] == T
    assert m["n_tracks"] == 1 and m["n_detections_tracked"] == T and m["n_edges"] == T - 1


def test_two_separate_particles_become_two_tracks_with_stable_uids():
    blobs, flow = _movie([(15, 10), (50, 10)])
    sol, _ = run_tracking(blobs, flow, T, PARAMS)
    assert track_metrics(sol, 3)["n_tracks"] == 2
    df = solution_to_track_df(sol)
    assert sorted(df["particle"].unique()) == [1, 2]
    uids = add_track_uids(df).groupby("particle")["track_uid"].first()
    # same input -> same uids on a second run
    sol2, _ = run_tracking(blobs, flow, T, PARAMS)
    uids2 = add_track_uids(solution_to_track_df(sol2)).groupby("particle")["track_uid"].first()
    assert list(uids) == list(uids2)


def test_annotate_writes_track_ids_on_every_node():
    blobs, flow = _movie([(15, 10), (50, 10)])
    sol, _ = run_tracking(blobs, flow, T, PARAMS)
    annotate_solution_track_ids(sol)
    ids = {d["track_id"] for _, d in sol.nodes(data=True)}
    assert ids == {1, 2} and all(d["tracklet_id"] == d["track_id"] for _, d in sol.nodes(data=True))
