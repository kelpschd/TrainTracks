#!/usr/bin/env python3
"""
Batch blob detection -> optical flow -> candidate graph -> Motile tracking.

This reproduces the TRACKING configuration from run 20260916_105301_9452:
    edge_selected_constant = -72
    max_gap                = 2
    gap_penalty            = 10
    dilation_radius        = 5
    max_edge_distance      = 30
    node_selected_weight   = 1
    edge_selected_weight   = 3
    node_appear_constant   = 2
    max_parents            = 1
    max_children           = 1

Detection is set to threshold 0.00017 because the reference run consumed
blobs_0_00017.csv. The remaining detection settings are those currently in
preprocess_blob_det.py.

Usage:
    python batch_pipeline.py /path/to/folder_of_zarr_movies
    python batch_pipeline.py /path/to/folder_of_zarr_movies --output-dir /path/to/output
    python batch_pipeline.py /path/to/folder_of_zarr_movies --overwrite

Expected movie layout:
    folder/
        movie_a.zarr/
        movie_b.zarr/
        ...

Each movie gets an isolated directory:
    output/<movie>/
        blobs.csv
        flow.zarr/
        tracks.geff
        metadata.json

A batch_summary.csv is also written at the output root.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import geff
import motile
import motile.constraints
import motile.costs
import networkx as nx
import numpy as np
import pandas as pd
import scipy.spatial
import skimage.measure
import zarr
from dask import compute, delayed
from dask.diagnostics import ProgressBar
from scipy import ndimage as ndi
from scipy.ndimage import median_filter
from skimage import filters, morphology
from skimage.feature import blob_log
from skimage.morphology import binary_dilation, disk
from skimage.exposure import equalize_adapthist


# ============================================================================
# Reference configuration
# ============================================================================

REFERENCE_RUN_ID = "20260916_105301_9452"
REFERENCE_RUN_NOTE = "gap_penalty sweep: ec-72.0 pen10.0 maxgap2"

ZARR_ARRAY_KEY = "s0"

DETECTION_PARAMS = dict(
    blob_channel=0,
    mask_channel=0,
    median_filter_size=10,
    apply_mask=False,
    min_object_size=400,
    mask_dilation_radius=5,
    threshold=0.00017,
    min_sigma=1,
    max_sigma=3,
    gaussian_sigma=1,
    use_gaussian_blur=False,
    normalize=False,
    bin_size=5,
    blob_filter=5,
)

FLOW_PARAMS = dict(
    kernel_size=(16, 16),
    clip_limit=0.01,
    pyr_scale=0.6,
    levels=5,
    winsize=15,
    iterations=8,
    poly_n=3,
    poly_sigma=0.7,
    flags=int(cv2.OPTFLOW_FARNEBACK_GAUSSIAN),
    temporal_smoothing_sigma=0.7,
)

TRACKING_PARAMS = dict(
    dilation_radius=5,
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

LONG_TRACK_MIN_FRAMES = 10


# ============================================================================
# Helpers
# ============================================================================


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def make_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:4]}"


def json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def write_json(path: Path, record: dict) -> None:
    with path.open("w") as f:
        json.dump(json_safe(record), f, indent=2)


def resolve_movie_dirs(input_dir: Path) -> list[Path]:
    movies = sorted(
        p for p in input_dir.iterdir()
        if p.name.endswith(".zarr") and p.is_dir()
    )
    if not movies:
        raise FileNotFoundError(f"No .zarr movie directories found in {input_dir}")
    return movies


# ============================================================================
# Blob detection
# ============================================================================


def preprocess_stack(
    img_stack: np.ndarray,
    blob_channel: int,
    mask_channel: int,
    median_filter_size: int,
    min_object_size: int,
    dilation_radius: int,
    apply_mask: bool,
) -> np.ndarray:
    if img_stack.ndim != 4:
        raise ValueError(f"Expected (T,C,Y,X); got {img_stack.shape}")
    if blob_channel >= img_stack.shape[1]:
        raise ValueError(
            f"blob_channel={blob_channel} but image has {img_stack.shape[1]} channels"
        )

    mean_proj = np.mean(img_stack[:, blob_channel], axis=0)
    background = median_filter(mean_proj, size=median_filter_size)

    preprocessed = np.stack([
        np.clip(frame - background, 0, None).astype(np.uint16)
        for frame in img_stack[:, blob_channel]
    ])

    if apply_mask:
        if mask_channel >= img_stack.shape[1]:
            raise ValueError(
                f"mask_channel={mask_channel} but image has {img_stack.shape[1]} channels"
            )
        ch_mask_proj = np.max(img_stack[:, mask_channel], axis=0)
        thresh = filters.threshold_otsu(ch_mask_proj)
        binary = ch_mask_proj > thresh
        filled = ndi.binary_fill_holes(binary)
        large = morphology.remove_small_objects(
            filled, min_size=min_object_size
        )
        dilated = morphology.binary_dilation(
            large, disk(dilation_radius)
        )
        preprocessed *= dilated.astype(np.uint16)

    return preprocessed


def detect_blobs_in_stack(
    image_stack: np.ndarray,
    threshold: float,
    min_sigma: float,
    max_sigma: float,
    gaussian_sigma: float,
    normalize: bool,
    use_gaussian_blur: bool,
) -> pd.DataFrame:
    @delayed
    def process_frame(t, frame):
        work = frame
        if normalize:
            work = (work - np.min(work)) / (
                np.max(work) - np.min(work) + 1e-8
            )
        if use_gaussian_blur:
            work = filters.gaussian(work, sigma=gaussian_sigma)
        blobs = blob_log(
            work,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            threshold=threshold,
        )
        return [
            {
                "frame": t,
                "x": x,
                "y": y,
                "size": sigma * np.sqrt(2),
            }
            for y, x, sigma in blobs
        ]

    tasks = [process_frame(t, frame) for t, frame in enumerate(image_stack)]
    with ProgressBar():
        results = compute(*tasks)
    positions = [entry for sublist in results for entry in sublist]
    return pd.DataFrame(positions)


def filter_dense_blobs(
    blob_df: pd.DataFrame,
    bin_size: int,
    blob_filter: int,
) -> pd.DataFrame:
    if len(blob_df) == 0:
        return blob_df

    rounded = blob_df.copy()
    rounded["x_rounded"] = (rounded["x"] // bin_size * bin_size).astype(int)
    rounded["y_rounded"] = (rounded["y"] // bin_size * bin_size).astype(int)

    counts = (
        rounded.groupby(["x_rounded", "y_rounded"])
        .size()
        .reset_index(name="count")
    )
    merged = rounded.merge(counts, on=["x_rounded", "y_rounded"])
    return merged[merged["count"] < blob_filter].reset_index(drop=True)


def blobs_df_to_tyx(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=["t", "y", "x"], dtype=int)
    return pd.DataFrame(
        {
            "t": df["frame"].astype(int),
            "y": np.round(df["y"]).astype(int),
            "x": np.round(df["x"]).astype(int),
        }
    )


# ============================================================================
# Optical flow
# ============================================================================


def compute_optical_flow(img_stack: np.ndarray, params: dict) -> np.ndarray:
    """
    Compute flow with shape (T-1, Y, X, 2), where flow[t] is frame t -> t+1.
    """
    if img_stack.ndim != 4:
        raise ValueError(f"Expected (T,C,Y,X); got {img_stack.shape}")
    if img_stack.shape[0] < 2:
        raise ValueError("Need at least 2 frames to compute optical flow")

    gray = img_stack[:, 0]
    min_int_val = gray.min()
    max_int_val = gray.max()
    denom = max_int_val - min_int_val
    if denom == 0:
        arr_norm = np.zeros_like(gray, dtype=np.uint8)
    else:
        arr_norm = (
            (gray - min_int_val) / denom * 255
        ).astype(np.uint8)

    kernel_size = list(params["kernel_size"])
    clip_limit = params["clip_limit"]
    prev = equalize_adapthist(
        arr_norm[0], kernel_size=kernel_size, clip_limit=clip_limit
    )
    prev = (prev * 255).astype(np.uint8)

    T, Y, X = arr_norm.shape
    flow = np.zeros((T - 1, Y, X, 2), dtype=np.float32)
    prev_flow = None

    for i in range(1, T):
        curr = equalize_adapthist(
            arr_norm[i], kernel_size=kernel_size, clip_limit=clip_limit
        )
        curr = (curr * 255).astype(np.uint8)

        current_flow = cv2.calcOpticalFlowFarneback(
            prev,
            curr,
            None,
            pyr_scale=params["pyr_scale"],
            levels=params["levels"],
            winsize=params["winsize"],
            iterations=params["iterations"],
            poly_n=params["poly_n"],
            poly_sigma=params["poly_sigma"],
            flags=params["flags"],
        )

        sigma = params["temporal_smoothing_sigma"]
        if prev_flow is not None:
            current_flow = sigma * current_flow + (1.0 - sigma) * prev_flow

        flow[i - 1] = current_flow.astype(np.float32)
        prev = curr
        prev_flow = current_flow

    return flow


def write_flow_zarr(flow: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(path, mode="w")
    arr = root.create_array(
        "flow_raw",
        shape=flow.shape,
        dtype=np.float32,
        chunks=(1, flow.shape[1], flow.shape[2], 2),
    )
    arr[:] = flow


# ============================================================================
# Candidate graph + solver
# ============================================================================


def build_candidate_graph(
    blobs_np: np.ndarray,
    flow_arr: np.ndarray,
    dilation_radius: int,
    n_frames: int,
) -> nx.DiGraph:
    """
    blobs (t,y,x) + flow (T-1,Y,X,2) -> candidate DiGraph.

    The final image frame has no outgoing flow by definition, so the final
    frame's node flow attribute is set to (0,0). This keeps all T image frames
    represented while preserving the correct flow convention: flow[t] is t->t+1.
    """
    if flow_arr.ndim != 4 or flow_arr.shape[-1] != 2:
        raise ValueError(f"Expected flow shape (T-1,Y,X,2); got {flow_arr.shape}")
    if n_frames != flow_arr.shape[0] + 1:
        raise ValueError(
            f"n_frames={n_frames} inconsistent with flow length {flow_arr.shape[0]}"
        )

    Y, X = flow_arr.shape[1], flow_arr.shape[2]
    mask = np.zeros((n_frames, Y, X), dtype=np.uint16)

    if len(blobs_np):
        if np.any(blobs_np[:, 0] < 0) or np.any(blobs_np[:, 0] >= n_frames):
            raise ValueError("Blob frame index out of bounds")
        if np.any(blobs_np[:, 1] < 0) or np.any(blobs_np[:, 1] >= Y):
            raise ValueError("Blob y index out of bounds")
        if np.any(blobs_np[:, 2] < 0) or np.any(blobs_np[:, 2] >= X):
            raise ValueError("Blob x index out of bounds")
        mask[blobs_np[:, 0], blobs_np[:, 1], blobs_np[:, 2]] = True

    dilated = np.zeros(mask.shape, dtype=np.uint16)
    footprint = disk(dilation_radius)
    for frame in range(n_frames):
        dilated[frame] = binary_dilation(mask[frame], footprint=footprint)

    cand_graph = nx.DiGraph()
    last_node_id = 0

    for frame in range(n_frames):
        segmentation = skimage.measure.label(dilated[frame])
        props = skimage.measure.regionprops(segmentation)

        # Only frames that can be a source for an outgoing edge have actual flow.
        flow_frame = flow_arr[frame] if frame < flow_arr.shape[0] else None

        for regionprop in props:
            node_id = int(regionprop.label) + last_node_id
            region = segmentation == regionprop.label
            centroid = (
                float(regionprop.centroid[0]),
                float(regionprop.centroid[1]),
            )

            if flow_frame is None:
                flow_vec = (0.0, 0.0)
            else:
                flow_vec = (
                    float(np.mean(flow_frame[region][..., 1])),
                    float(np.mean(flow_frame[region][..., 0])),
                )

            cand_graph.add_node(
                node_id,
                t=frame,
                x=centroid[1],
                y=centroid[0],
                flow=flow_vec,
            )

        # Advance the global offset only after this frame's nodes are created.
        last_node_id += len(props)

    if cand_graph.number_of_nodes() != len(cand_graph.nodes):
        raise RuntimeError("Candidate graph node count is internally inconsistent")

    return cand_graph


def _compute_node_frame_dict(cand_graph: nx.DiGraph) -> dict[int, list[int]]:
    frame_dict: dict[int, list[int]] = {}
    for node, data in cand_graph.nodes(data=True):
        frame_dict.setdefault(data["t"], []).append(node)
    return frame_dict


def _create_kdtree(cand_graph: nx.DiGraph, node_ids: list[int]) -> scipy.spatial.KDTree:
    positions = [[cand_graph.nodes[n]["x"], cand_graph.nodes[n]["y"]] for n in node_ids]
    return scipy.spatial.KDTree(positions)


def add_candidate_edges(
    cand_graph: nx.DiGraph,
    max_edge_distance: float,
    max_gap: int,
) -> None:
    node_frame_dict = _compute_node_frame_dict(cand_graph)
    frames = sorted(node_frame_dict.keys())
    if not frames:
        return

    kdtrees = {
        f: _create_kdtree(cand_graph, node_frame_dict[f])
        for f in frames
        if node_frame_dict[f]
    }

    for frame in frames:
        src_ids = node_frame_dict[frame]
        if not src_ids:
            continue
        src_tree = kdtrees[frame]

        for gap in range(1, max_gap + 1):
            target = frame + gap
            if target not in node_frame_dict:
                continue
            tgt_ids = node_frame_dict[target]
            if not tgt_ids:
                continue
            tgt_tree = kdtrees[target]
            matched = src_tree.query_ball_tree(
                tgt_tree, max_edge_distance * gap
            )
            for src_id, tgt_idxs in zip(src_ids, matched):
                for j in tgt_idxs:
                    cand_graph.add_edge(src_id, tgt_ids[j], gap=gap)


def add_flow_dist_attr(cand_graph: nx.DiGraph, gap_penalty: float) -> None:
    for u, v in cand_graph.edges:
        node_u = cand_graph.nodes[u]
        node_v = cand_graph.nodes[v]
        gap = cand_graph.edges[u, v].get("gap", 1)
        pos_u = np.array([node_u["y"], node_u["x"]])
        pos_v = np.array([node_v["y"], node_v["x"]])
        predicted = pos_u + np.array(node_u["flow"]) * gap
        base = float(np.linalg.norm(predicted - pos_v))
        cand_graph.edges[u, v]["flow_offset"] = (
            base + gap_penalty * (gap - 1)
        )


def solve_tracks(cand_graph: nx.DiGraph, params: dict) -> nx.DiGraph:
    if cand_graph.number_of_nodes() == 0:
        return nx.DiGraph()

    cand_trackgraph = motile.TrackGraph(cand_graph, frame_attribute="t")
    add_flow_dist_attr(cand_trackgraph, params["gap_penalty"])

    solver = motile.Solver(cand_trackgraph)
    solver.add_cost(
        motile.costs.NodeSelectedCost(
            weight=params["node_selected_weight"]
        )
    )
    solver.add_cost(
        motile.costs.EdgeSelectedCost(
            weight=params["edge_selected_weight"],
            attribute=params["edge_flow_attribute"],
            constant=params["edge_selected_constant"],
        )
    )
    solver.add_cost(
        motile.costs.NodeAppearCost(
            constant=params["node_appear_constant"]
        )
    )
    solver.add_constraint(
        motile.constraints.MaxParents(params["max_parents"])
    )
    solver.add_constraint(
        motile.constraints.MaxChildren(params["max_children"])
    )
    solver.solve()
    return solver.get_selected_subgraph().to_nx_graph()


# ============================================================================
# Metrics
# ============================================================================


def track_metrics(solution_nx: nx.DiGraph, long_min_frames: int) -> dict:
    n_nodes = solution_nx.number_of_nodes()
    n_edges = solution_nx.number_of_edges()
    components = list(nx.weakly_connected_components(solution_nx))
    n_tracks = len(components)

    if n_tracks == 0:
        return {
            "n_detections_tracked": 0,
            "n_edges": 0,
            "n_tracks": 0,
            "mean_track_len": 0.0,
            "median_track_len": 0.0,
            "max_track_len": 0,
            "n_long_tracks": 0,
            "frac_nodes_in_long": 0.0,
            "tracks_per_detection": 0.0,
            "n_skip_edges": 0,
            "frac_skip_edges": 0.0,
        }

    lengths = []
    nodes_in_long = 0
    for comp in components:
        frames = [solution_nx.nodes[n]["t"] for n in comp]
        span = max(frames) - min(frames) + 1
        lengths.append(span)
        if span >= long_min_frames:
            nodes_in_long += len(comp)

    lengths = np.asarray(lengths)
    n_skip = sum(
        1
        for u, v in solution_nx.edges
        if abs(solution_nx.nodes[v]["t"] - solution_nx.nodes[u]["t"]) > 1
    )

    return {
        "n_detections_tracked": int(n_nodes),
        "n_edges": int(n_edges),
        "n_tracks": int(n_tracks),
        "mean_track_len": float(np.mean(lengths)),
        "median_track_len": float(np.median(lengths)),
        "max_track_len": int(np.max(lengths)),
        "n_long_tracks": int(np.sum(lengths >= long_min_frames)),
        "frac_nodes_in_long": float(nodes_in_long / n_nodes),
        "tracks_per_detection": float(n_tracks / n_nodes),
        "n_skip_edges": int(n_skip),
        "frac_skip_edges": float(n_skip / n_edges) if n_edges else 0.0,
    }


# ============================================================================
# Per-movie pipeline
# ============================================================================


def process_movie(movie_path: Path, output_root: Path, overwrite: bool) -> dict:
    movie_id = movie_path.name[:-5] if movie_path.name.endswith(".zarr") else movie_path.stem
    run_id = make_run_id()
    movie_out = output_root / movie_id

    if movie_out.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {movie_out}. Use --overwrite to replace it."
            )
        shutil.rmtree(movie_out)
    movie_out.mkdir(parents=True, exist_ok=True)

    blobs_csv = movie_out / "blobs.csv"
    flow_zarr = movie_out / "flow.zarr"
    tracks_geff = movie_out / "tracks.geff"
    metadata_json = movie_out / "metadata.json"

    started = now_iso()
    print("\n" + "=" * 80)
    print(f"MOVIE: {movie_path.name}")
    print(f"RUN : {run_id}")
    print("=" * 80)

    root = zarr.open_group(str(movie_path), mode="r")
    img_stack = np.asarray(root[ZARR_ARRAY_KEY])
    if img_stack.ndim != 4:
        raise ValueError(
            f"{movie_path}: expected {ZARR_ARRAY_KEY} shape (T,C,Y,X), got {img_stack.shape}"
        )

    T, C, Y, X = img_stack.shape
    print(f"Image shape: {img_stack.shape}, dtype={img_stack.dtype}")

    # Detection
    print(f"Detecting blobs (threshold={DETECTION_PARAMS['threshold']})...")
    pre = preprocess_stack(
        img_stack,
        blob_channel=DETECTION_PARAMS["blob_channel"],
        mask_channel=DETECTION_PARAMS["mask_channel"],
        median_filter_size=DETECTION_PARAMS["median_filter_size"],
        min_object_size=DETECTION_PARAMS["min_object_size"],
        dilation_radius=DETECTION_PARAMS["mask_dilation_radius"],
        apply_mask=DETECTION_PARAMS["apply_mask"],
    )
    raw_blobs = detect_blobs_in_stack(
        pre,
        threshold=DETECTION_PARAMS["threshold"],
        min_sigma=DETECTION_PARAMS["min_sigma"],
        max_sigma=DETECTION_PARAMS["max_sigma"],
        gaussian_sigma=DETECTION_PARAMS["gaussian_sigma"],
        normalize=DETECTION_PARAMS["normalize"],
        use_gaussian_blur=DETECTION_PARAMS["use_gaussian_blur"],
    )
    filtered_blobs = filter_dense_blobs(
        raw_blobs,
        bin_size=DETECTION_PARAMS["bin_size"],
        blob_filter=DETECTION_PARAMS["blob_filter"],
    )
    blobs_df = blobs_df_to_tyx(filtered_blobs)
    blobs_df.to_csv(blobs_csv, index=False)
    print(f"Blobs: {len(raw_blobs)} raw -> {len(blobs_df)} after density filter")

    # Flow
    print("Computing optical flow...")
    flow_raw = compute_optical_flow(img_stack, FLOW_PARAMS)
    write_flow_zarr(flow_raw, flow_zarr)
    print(f"Flow shape: {flow_raw.shape} (explicitly T-1 transitions)")

    # Candidate graph and solve
    blobs_np = blobs_df[["t", "y", "x"]].to_numpy(dtype=np.int64)
    cand = build_candidate_graph(
        blobs_np,
        flow_raw,
        dilation_radius=TRACKING_PARAMS["dilation_radius"],
        n_frames=T,
    )
    add_candidate_edges(
        cand,
        max_edge_distance=TRACKING_PARAMS["max_edge_distance"],
        max_gap=TRACKING_PARAMS["max_gap"],
    )
    print(
        f"Candidate graph: {cand.number_of_nodes()} nodes, "
        f"{cand.number_of_edges()} edges"
    )

    solution_nx = solve_tracks(cand, TRACKING_PARAMS)
    metrics = track_metrics(solution_nx, LONG_TRACK_MIN_FRAMES)

    print(
        f"Solution: tracks={metrics['n_tracks']} | "
        f"tracked_nodes={metrics['n_detections_tracked']} | "
        f"median_span={metrics['median_track_len']:.1f} | "
        f"max_span={metrics['max_track_len']} | "
        f"long>={LONG_TRACK_MIN_FRAMES}={metrics['n_long_tracks']} | "
        f"skip_edges={metrics['n_skip_edges']} "
        f"({metrics['frac_skip_edges']:.1%})"
    )

    if solution_nx.number_of_nodes() > 0:
        geff.write(solution_nx, str(tracks_geff), zarr_format=3, overwrite=True)
    else:
        print("Warning: empty solution; no tracks.geff written")

    record = {
        "run_id": run_id,
        "reference_run_id": REFERENCE_RUN_ID,
        "reference_run_note": REFERENCE_RUN_NOTE,
        "timestamp_start": started,
        "timestamp_end": now_iso(),
        "input_zarr": str(movie_path),
        "zarr_array_key": ZARR_ARRAY_KEY,
        "output_dir": str(movie_out),
        "output_blobs_csv": str(blobs_csv),
        "output_flow_zarr": str(flow_zarr),
        "output_geff": str(tracks_geff),
        "image_shape": list(img_stack.shape),
        "n_blobs_raw": int(len(raw_blobs)),
        "n_blobs_input": int(len(blobs_df)),
        "long_track_min_frames": LONG_TRACK_MIN_FRAMES,
        **metrics,
        "detection_params": DETECTION_PARAMS,
        "flow_params": FLOW_PARAMS,
        "tracking_params": TRACKING_PARAMS,
    }
    write_json(metadata_json, record)
    print(f"Wrote: {metadata_json}")
    return record


# ============================================================================
# Batch driver
# ============================================================================


def write_summary(records: list[dict], path: Path) -> None:
    columns = [
        "run_id",
        "input_zarr",
        "n_blobs_raw",
        "n_blobs_input",
        "n_detections_tracked",
        "n_edges",
        "n_tracks",
        "mean_track_len",
        "median_track_len",
        "max_track_len",
        "n_long_tracks",
        "frac_nodes_in_long",
        "tracks_per_detection",
        "n_skip_edges",
        "frac_skip_edges",
        "output_geff",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the validated tracking configuration over a folder of Zarr movies."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Folder containing *.zarr movie directories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root (default: <input_dir>/tracking_batch)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing per-movie output directory",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch at the first failed movie",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        parser.error(f"Input directory not found: {input_dir}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_dir / "tracking_batch"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    movies = resolve_movie_dirs(input_dir)
    print(f"Found {len(movies)} Zarr movies")
    print(f"Output root: {output_dir}")
    print(f"Reference tracking run: {REFERENCE_RUN_ID}")
    print(json.dumps(json_safe(TRACKING_PARAMS), indent=2))
    print(f"Detection threshold: {DETECTION_PARAMS['threshold']}")

    records = []
    failures = []

    for movie in movies:
        try:
            records.append(process_movie(movie, output_dir, args.overwrite))
        except Exception as exc:
            print(f"\nFAILED: {movie.name}: {exc}", file=sys.stderr)
            traceback.print_exc()
            failures.append({"input_zarr": str(movie), "error": str(exc)})
            if args.stop_on_error:
                break

    summary_csv = output_dir / "batch_summary.csv"
    if records:
        write_summary(records, summary_csv)
        print(f"\nWrote batch summary: {summary_csv}")

    if failures:
        failures_json = output_dir / "batch_failures.json"
        write_json(failures_json, failures)
        print(f"Wrote failure log: {failures_json}", file=sys.stderr)

    print(
        f"\nBatch complete: {len(records)} succeeded, "
        f"{len(failures)} failed, {len(movies)} total"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
