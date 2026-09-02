# candidate_graph.py
#
# Input : ONE blob CSV (t, y, x), passed by the user.
# Output: - a solved track GEFF, named after the input CSV
#         - one appended row in a shared CSV log capturing the experimental
#           setup (input, params) and resulting track-continuity metrics.
#
# Detection is NOT part of this script -- blobs come from a separate detection
# script. Run this once per blob CSV; the log accumulates one row per run so you
# can compare tracking across detection thresholds.
#
# Usage:
#   python candidate_graph.py /path/to/blobs_0_00021.csv
#   python candidate_graph.py /path/to/blobs_0_00021.csv --note "baseline"

from __future__ import annotations

import re
import csv
import json
import argparse
from typing import Iterable, Any
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx
import scipy.spatial
import skimage.measure
from skimage.morphology import disk, binary_dilation
from tqdm.auto import tqdm

import zarr
import motile
import motile.costs
import motile.constraints
import geff


# ============================================================================
# CONFIG -- fixed inputs and tracking params (held constant across runs so the
# only thing that varies between logged rows is the input blob CSV / threshold)
# ============================================================================

# Flow (precomputed by optical_flow.py): (T, Y, X, 2)
FLOW_PATH = Path(
    "/Users/kelpschdj/Documents/DataTecnica/TTU/TrainTracks/flow.zarr"
)

# Where GEFF outputs and the log CSV go
OUTPUT_DIR = Path(
    "/Users/kelpschdj/Documents/DataTecnica/TTU/TrainTracks/tracking_runs"
)
LOG_CSV = OUTPUT_DIR / "tracking_log.csv"

# Fixed tracking params -- change only if you deliberately want to; every value
# is recorded in the log so any row is fully reproducible.
FIXED_PARAMS = dict(
    dilation_radius=5,          # disk radius for per-region flow averaging
    max_edge_distance=30.0,     # KDTree edge cutoff between adjacent frames
    node_selected_weight=1.0,   # motile NodeSelectedCost
    edge_selected_weight=3.0,   # motile EdgeSelectedCost
    edge_flow_attribute="flow_offset",
    edge_selected_constant=-36.0,
    node_appear_constant=2.0,   # motile NodeAppearCost
    max_parents=1,
    max_children=1,
)

# A track is "long" if it spans at least this many frames.
LONG_TRACK_MIN_FRAMES = 10


# ============================================================================
# Candidate graph  (logic unchanged from your original candidate_graph.py)
# ============================================================================

def build_candidate_graph(
    blobs_np: np.ndarray,
    flow_arr: np.ndarray,
    dilation_radius: int,
) -> nx.DiGraph:
    """blobs (t,y,x) + flow (T,Y,X,2) -> candidate DiGraph with flow attrs."""
    mask = np.zeros(flow_arr.shape[:-1], dtype=np.uint16)  # (T, Y, X)
    mask[blobs_np[:, 0], blobs_np[:, 1], blobs_np[:, 2]] = True

    dilated = np.zeros(mask.shape, dtype=np.uint16)
    footprint = disk(dilation_radius)
    for frame in range(mask.shape[0]):
        dilated[frame] = binary_dilation(mask[frame], footprint=footprint)

    cand_graph = nx.DiGraph()
    last_node_id = 0
    for frame in range(dilated.shape[0]):
        segmentation = skimage.measure.label(dilated[frame])
        flow_frame = flow_arr[frame]
        props = skimage.measure.regionprops(segmentation)
        last_node_id += len(props)

        for regionprop in props:
            node_id = int(regionprop.label)
            region = segmentation == node_id
            centroid = (float(regionprop.centroid[0]), float(regionprop.centroid[1]))
            flow = (
                float(np.mean(flow_frame[region][..., 1])),
                float(np.mean(flow_frame[region][..., 0])),
            )
            attrs = {"t": frame, "x": centroid[1], "y": centroid[0], "flow": flow}
            node_id += int(last_node_id)
            cand_graph.add_node(node_id, **attrs)

    return cand_graph


def _compute_node_frame_dict(cand_graph: nx.DiGraph) -> dict[int, list[Any]]:
    node_frame_dict: dict[int, list[Any]] = {}
    for node, data in cand_graph.nodes(data=True):
        node_frame_dict.setdefault(data["t"], []).append(node)
    return node_frame_dict


def _create_kdtree(cand_graph: nx.DiGraph, node_ids: Iterable[Any]) -> scipy.spatial.KDTree:
    positions = [[cand_graph.nodes[n]["x"], cand_graph.nodes[n]["y"]] for n in node_ids]
    return scipy.spatial.KDTree(positions)


def add_cand_edges(cand_graph: nx.DiGraph, max_edge_distance: float) -> None:
    """Connect nodes within max_edge_distance in adjacent frames (in place)."""
    print("Extracting candidate edges")
    node_frame_dict = _compute_node_frame_dict(cand_graph)
    frames = sorted(node_frame_dict.keys())
    if not frames:
        return
    prev_node_ids = node_frame_dict[frames[0]]
    prev_kdtree = _create_kdtree(cand_graph, prev_node_ids)

    for frame in tqdm(frames):
        if frame + 1 not in node_frame_dict:
            continue
        next_node_ids = node_frame_dict[frame + 1]
        next_kdtree = _create_kdtree(cand_graph, next_node_ids)
        matched = prev_kdtree.query_ball_tree(next_kdtree, max_edge_distance)
        for prev_id, next_idxs in zip(prev_node_ids, matched):
            for j in next_idxs:
                cand_graph.add_edge(prev_id, next_node_ids[j])
        prev_node_ids = next_node_ids
        prev_kdtree = next_kdtree


def add_flow_dist_attr(cand_graph: motile.TrackGraph) -> None:
    """Attach flow_offset to each edge: |(pos_u + flow_u) - pos_v|."""
    for edge in cand_graph.edges:
        u, v = edge
        node_u = cand_graph.nodes[u]
        node_v = cand_graph.nodes[v]
        pos_u = np.array([node_u["y"], node_u["x"]])
        pos_v = np.array([node_v["y"], node_v["x"]])
        predicted = pos_u + np.array(node_u["flow"])
        cand_graph.edges[edge]["flow_offset"] = float(np.linalg.norm(predicted - pos_v))


def solve_tracks(cand_graph: nx.DiGraph, params: dict) -> nx.DiGraph:
    """Run the motile solver with the fixed cost/constraint set; return nx graph."""
    if cand_graph.number_of_nodes() == 0:
        return nx.DiGraph()

    cand_trackgraph = motile.TrackGraph(cand_graph, frame_attribute="t")
    print("Calculating drift distances using optical flow...")
    add_flow_dist_attr(cand_trackgraph)

    solver = motile.Solver(cand_trackgraph)
    solver.add_cost(motile.costs.NodeSelectedCost(weight=params["node_selected_weight"]))
    solver.add_cost(
        motile.costs.EdgeSelectedCost(
            weight=params["edge_selected_weight"],
            attribute=params["edge_flow_attribute"],
            constant=params["edge_selected_constant"],
        )
    )
    solver.add_cost(motile.costs.NodeAppearCost(constant=params["node_appear_constant"]))
    solver.add_constraint(motile.constraints.MaxParents(params["max_parents"]))
    solver.add_constraint(motile.constraints.MaxChildren(params["max_children"]))

    solver.solve()
    return solver.get_selected_subgraph().to_nx_graph()


# ============================================================================
# Metrics + logging
# ============================================================================

def track_metrics(solution_nx: nx.DiGraph, long_min_frames: int) -> dict:
    """Continuity-focused metrics from a solved track graph.

    A 'track' = one weakly-connected component. Track length = frames spanned.
    """
    n_nodes = solution_nx.number_of_nodes()
    n_edges = solution_nx.number_of_edges()
    components = list(nx.weakly_connected_components(solution_nx))
    n_tracks = len(components)

    if n_tracks == 0:
        return dict(
            n_detections_tracked=0, n_edges=0, n_tracks=0,
            mean_track_len=0.0, median_track_len=0.0, max_track_len=0,
            n_long_tracks=0, frac_nodes_in_long=0.0, tracks_per_detection=0.0,
        )

    lengths = []
    nodes_in_long = 0
    for comp in components:
        frames = [solution_nx.nodes[n]["t"] for n in comp]
        span = max(frames) - min(frames) + 1
        lengths.append(span)
        if span >= long_min_frames:
            nodes_in_long += len(comp)
    lengths = np.array(lengths)

    return dict(
        n_detections_tracked=int(n_nodes),
        n_edges=int(n_edges),
        n_tracks=int(n_tracks),
        mean_track_len=float(np.mean(lengths)),
        median_track_len=float(np.median(lengths)),
        max_track_len=int(np.max(lengths)),
        n_long_tracks=int(np.sum(lengths >= long_min_frames)),
        frac_nodes_in_long=float(nodes_in_long / n_nodes),
        # Fragmentation proxy: fewer tracks per detection = longer, less-broken
        # tracks. If a lower detection threshold stitches tracklets, this should
        # drop/hold as detections rise; if it only adds noise, it rises.
        tracks_per_detection=float(n_tracks / n_nodes),
    )


def parse_threshold_from_name(csv_path: Path):
    """Pull the threshold out of names like blobs_0_00021.csv -> 0.00021.

    Returns float or None if it can't be parsed (logged as null, not an error).
    """
    m = re.search(r"blobs_([0-9]+(?:_[0-9]+)?)", csv_path.stem)
    if not m:
        return None
    try:
        return float(m.group(1).replace("_", "."))
    except ValueError:
        return None


def append_log_row(row: dict) -> None:
    """Append one row to LOG_CSV, writing the header once. Stable column order."""
    columns = [
        "timestamp", "input_csv", "output_geff", "threshold", "note",
        "n_blobs_input",
        # metrics
        "n_detections_tracked", "n_edges", "n_tracks",
        "mean_track_len", "median_track_len", "max_track_len",
        "n_long_tracks", "frac_nodes_in_long", "tracks_per_detection",
        "long_track_min_frames",
        # fixed params (flattened, so every row is self-describing)
        "dilation_radius", "max_edge_distance",
        "node_selected_weight", "edge_selected_weight",
        "edge_flow_attribute", "edge_selected_constant",
        "node_appear_constant", "max_parents", "max_children",
    ]
    write_header = not LOG_CSV.exists()
    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ============================================================================
# Driver
# ============================================================================

def run(blobs_csv: Path, note: str = "") -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load blobs (t, y, x) ---
    print(f"Loading blobs: {blobs_csv}")
    blobs_df = pd.read_csv(blobs_csv)
    blobs_np = blobs_df.to_numpy().astype(np.uint16)
    n_blobs = len(blobs_np)
    print(f"  {n_blobs} blobs loaded")

    # --- Load flow ---
    flow_root = zarr.open_group(FLOW_PATH, mode="r")
    flow_arr = np.array(flow_root["flow_raw"])  # (T, Y, X, 2)
    print(f"  flow shape: {flow_arr.shape}")

    # --- Candidate graph -> solve ---
    cand = build_candidate_graph(blobs_np, flow_arr, FIXED_PARAMS["dilation_radius"])
    add_cand_edges(cand, FIXED_PARAMS["max_edge_distance"])
    print(f"Candidate graph: {cand.number_of_nodes()} nodes, {cand.number_of_edges()} edges")
    solution_nx = solve_tracks(cand, FIXED_PARAMS)

    # --- Metrics ---
    metrics = track_metrics(solution_nx, LONG_TRACK_MIN_FRAMES)
    print(f"Tracks={metrics['n_tracks']}  long(>={LONG_TRACK_MIN_FRAMES})="
          f"{metrics['n_long_tracks']}  max_len={metrics['max_track_len']}  "
          f"tracks/det={metrics['tracks_per_detection']:.4f}")

    # --- Write GEFF named after the input CSV (no overwrite across thresholds) ---
    output_geff = OUTPUT_DIR / f"{blobs_csv.stem}.geff"
    if solution_nx.number_of_nodes() > 0:
        # overwrite=True so re-running the same CSV replaces its own GEFF rather
        # than erroring on the existing store.
        geff.write(solution_nx, str(output_geff), zarr_format=3, overwrite=True)
        print(f"Wrote GEFF: {output_geff}")
    else:
        print("Empty solution graph -- no GEFF written")

    # --- Append one log row ---
    threshold = parse_threshold_from_name(blobs_csv)
    row = dict(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        input_csv=str(blobs_csv),
        output_geff=str(output_geff),
        threshold=threshold,
        note=note,
        n_blobs_input=n_blobs,
        long_track_min_frames=LONG_TRACK_MIN_FRAMES,
        **metrics,
        **FIXED_PARAMS,
    )
    append_log_row(row)
    print(f"Logged run -> {LOG_CSV}")
    return row


def main():
    parser = argparse.ArgumentParser(description="Track one blob CSV and log the run.")
    parser.add_argument("blobs_csv", type=Path, help="Path to a blob CSV (columns t, y, x)")
    parser.add_argument("--note", default="", help="Optional free-text note for the log row")
    args = parser.parse_args()

    if not args.blobs_csv.exists():
        parser.error(f"Blob CSV not found: {args.blobs_csv}")

    run(args.blobs_csv, note=args.note)


if __name__ == "__main__":
    main()


# python scripts/candidate_graph.py /Users/kelpschdj/Documents/DataTecnica/TTU/TrainTracks/blobs/blobs_0_00021.csv --note "original threshold"
# python scripts/candidate_graph.py /Users/kelpschdj/Documents/DataTecnica/TTU/TrainTracks/blobs/blobs_0_00019.csv
# python scripts/candidate_graph.py /Users/kelpschdj/Documents/DataTecnica/TTU/TrainTracks/blobs/blobs_0_00017.csv
# python scripts/candidate_graph.py /Users/kelpschdj/Documents/DataTecnica/TTU/TrainTracks/blobs/blobs_0_00015.csv
