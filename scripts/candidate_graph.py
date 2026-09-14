# candidate_graph.py
#
# Input : ONE blob CSV (t, y, x), passed by the user.
# Output: - a unique run directory under tracking_runs/runs/<run_id>/ holding
#           the solved track GEFF and a metadata.json (all params + results)
#         - one appended row in a shared CSV log indexing every run
#
# Detection is NOT part of this script -- blobs come from a separate detection
# script. Run this once per blob CSV. Each run is fully isolated (its own
# directory + run_id), so sweeping gap params never overwrites a prior run.
#
# Usage:
#   python candidate_graph.py /path/to/blobs_0_00021.csv
#   python candidate_graph.py /path/to/blobs_0_00021.csv --note "maxgap2 pen20"

from __future__ import annotations

import csv
import json
import uuid
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
# only thing that varies between logged rows is the input blob CSV and whatever
# gap params you deliberately change here)
# ============================================================================

# Flow (precomputed by optical_flow.py): (T, Y, X, 2)
FLOW_PATH = Path(
    "/Users/kelpschdj/Documents/DataTecnica/TTU/TrainTracks/flow.zarr"
)

# Root for all tracking outputs. Each run gets its own subdir under runs/,
# and a shared CSV indexes every run for cross-run comparison.
OUTPUT_DIR = Path(
    "/Users/kelpschdj/Documents/DataTecnica/TTU/TrainTracks/tracking_runs"
)
RUNS_DIR = OUTPUT_DIR / "runs"        # per-run subdirectories live here
LOG_CSV = OUTPUT_DIR / "tracking_log.csv"  # one row per run (queryable index)

# Tracking params. Every value is recorded in the log so any row is fully
# reproducible. Edit + rerun to sweep; the gap params are the ones you'll tune.
FIXED_PARAMS = dict(
    dilation_radius=5,          # disk radius for per-region flow averaging
    max_edge_distance=30.0,     # KDTree edge cutoff between adjacent frames
    # --- gap-closing (skip edges) ---
    max_gap=2,                  # 1 = adjacent frames only (original behavior);
                                # 2 or 3 = allow skipping missing frames.
    gap_penalty=20.0,           # added to flow_offset per skipped frame, so a
                                # direct edge wins when one exists. 0 = skips
                                # free (over-merges); high = skips never chosen.
    # --- motile costs / constraints ---
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
# Candidate graph
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


def add_cand_edges(
    cand_graph: nx.DiGraph,
    max_edge_distance: float,
    max_gap: int = 1,
) -> None:
    """Connect nodes across frames within distance, in place.

    For each source frame f, connect to frames f+1 ... f+max_gap. The search
    radius scales with the gap (max_edge_distance * gap) because a cell unseen
    for `gap` frames can have travelled proportionally farther. Each edge stores
    its `gap` (1 = adjacent) so the cost can penalize skips.

    max_gap=1 reproduces the original adjacent-frame-only behavior exactly.
    """
    print(f"Extracting candidate edges (max_gap={max_gap})")
    node_frame_dict = _compute_node_frame_dict(cand_graph)
    frames = sorted(node_frame_dict.keys())
    if not frames:
        return

    # Cache one KDTree per frame (each frame is now a target for several
    # source frames, so building trees once avoids redundant work).
    kdtrees = {f: _create_kdtree(cand_graph, node_frame_dict[f]) for f in frames}

    for frame in tqdm(frames):
        src_ids = node_frame_dict[frame]
        src_tree = kdtrees[frame]
        for gap in range(1, max_gap + 1):
            target = frame + gap
            if target not in node_frame_dict:
                continue
            tgt_ids = node_frame_dict[target]
            tgt_tree = kdtrees[target]
            matched = src_tree.query_ball_tree(tgt_tree, max_edge_distance * gap)
            for src_id, tgt_idxs in zip(src_ids, matched):
                for j in tgt_idxs:
                    cand_graph.add_edge(src_id, tgt_ids[j], gap=gap)


def add_flow_dist_attr(cand_graph: motile.TrackGraph, gap_penalty: float) -> None:
    """Attach flow_offset to each edge.

    Base offset is |(pos_u + flow_u * gap) - pos_v|: the per-frame flow is
    extrapolated across the gap (constant-velocity assumption, fine for small
    gaps). A penalty of gap_penalty * (gap - 1) is added so a direct edge (gap
    1, no penalty) is preferred whenever it competes with a skip edge.
    """
    for edge in cand_graph.edges:
        u, v = edge
        node_u = cand_graph.nodes[u]
        node_v = cand_graph.nodes[v]
        gap = cand_graph.edges[edge].get("gap", 1)
        pos_u = np.array([node_u["y"], node_u["x"]])
        pos_v = np.array([node_v["y"], node_v["x"]])
        predicted = pos_u + np.array(node_u["flow"]) * gap
        base = float(np.linalg.norm(predicted - pos_v))
        cand_graph.edges[edge]["flow_offset"] = base + gap_penalty * (gap - 1)


def solve_tracks(cand_graph: nx.DiGraph, params: dict) -> nx.DiGraph:
    """Run the motile solver with the fixed cost/constraint set; return nx graph."""
    if cand_graph.number_of_nodes() == 0:
        return nx.DiGraph()

    cand_trackgraph = motile.TrackGraph(cand_graph, frame_attribute="t")
    print("Calculating drift distances using optical flow...")
    add_flow_dist_attr(cand_trackgraph, params["gap_penalty"])

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
            n_skip_edges=0, frac_skip_edges=0.0,
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

    # How many selected edges span a gap (frame difference > 1). Requires the
    # 't' attribute on both endpoints, which the solution graph carries.
    n_skip = 0
    for u, v in solution_nx.edges:
        du = solution_nx.nodes[u]["t"]
        dv = solution_nx.nodes[v]["t"]
        if abs(dv - du) > 1:
            n_skip += 1

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
        # tracks. Should drop as gap-closing stitches tracklets together.
        tracks_per_detection=float(n_tracks / n_nodes),
        # How much the solver actually used the skip edges you offered.
        n_skip_edges=int(n_skip),
        frac_skip_edges=float(n_skip / n_edges) if n_edges else 0.0,
    )


def make_run_id() -> str:
    """Timestamp + short random suffix, e.g. 20260914_142530_a3f1.

    Timestamp sorts chronologically and is readable at a glance; the 4-char
    suffix avoids collisions when two runs start in the same second.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:4]
    return f"{ts}_{suffix}"


def write_metadata_json(run_dir: Path, meta: dict) -> Path:
    """Write the authoritative per-run record as metadata.json."""
    path = run_dir / "metadata.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return path


def append_log_row(row: dict) -> None:
    """Append one row to LOG_CSV (the cross-run index), header once.

    This is a flat, queryable mirror of each run's metadata.json. The JSON is
    authoritative; the CSV is for scanning/comparing runs in one table.
    """
    columns = [
        "run_id", "timestamp", "input_csv", "output_geff", "note",
        "n_blobs_input",
        # metrics
        "n_detections_tracked", "n_edges", "n_tracks",
        "mean_track_len", "median_track_len", "max_track_len",
        "n_long_tracks", "frac_nodes_in_long", "tracks_per_detection",
        "n_skip_edges", "frac_skip_edges",
        "long_track_min_frames",
        # params (flattened, so every row is self-describing)
        "dilation_radius", "max_edge_distance",
        "max_gap", "gap_penalty",
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
    # Create this run's isolated directory up front.
    run_id = make_run_id()
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run ID: {run_id}")
    print(f"Run dir: {run_dir}")

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
    add_cand_edges(
        cand,
        FIXED_PARAMS["max_edge_distance"],
        max_gap=FIXED_PARAMS["max_gap"],
    )
    print(f"Candidate graph: {cand.number_of_nodes()} nodes, {cand.number_of_edges()} edges")
    solution_nx = solve_tracks(cand, FIXED_PARAMS)

    # --- Metrics ---
    metrics = track_metrics(solution_nx, LONG_TRACK_MIN_FRAMES)
    print(f"Tracks={metrics['n_tracks']}  long(>={LONG_TRACK_MIN_FRAMES})="
          f"{metrics['n_long_tracks']}  max_len={metrics['max_track_len']}  "
          f"tracks/det={metrics['tracks_per_detection']:.4f}  "
          f"skip_edges={metrics['n_skip_edges']} ({metrics['frac_skip_edges']:.1%})")

    # --- Write GEFF into the run directory (fixed name; the run_id disambiguates) ---
    output_geff = run_dir / "tracks.geff"
    if solution_nx.number_of_nodes() > 0:
        geff.write(solution_nx, str(output_geff), zarr_format=3, overwrite=True)
        print(f"Wrote GEFF: {output_geff}")
    else:
        print("Empty solution graph -- no GEFF written")

    # --- Assemble the run record (authoritative JSON + CSV index row) ---
    record = dict(
        run_id=run_id,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        input_csv=str(blobs_csv),
        output_geff=str(output_geff),
        note=note,
        n_blobs_input=n_blobs,
        long_track_min_frames=LONG_TRACK_MIN_FRAMES,
        **metrics,
        **FIXED_PARAMS,
    )

    json_path = write_metadata_json(run_dir, record)
    print(f"Wrote metadata: {json_path}")

    append_log_row(record)
    print(f"Indexed run -> {LOG_CSV}")
    return record


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


# python scripts/candidate_graph.py /Users/kelpschdj/Documents/DataTecnica/TTU/TrainTracks/blobs/blobs_0_00021.csv --note "maxgap2 pen20"
# python scripts/candidate_graph.py /Users/kelpschdj/Documents/DataTecnica/TTU/TrainTracks/blobs/blobs_0_00019.csv