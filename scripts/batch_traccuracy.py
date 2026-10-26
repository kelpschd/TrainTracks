#!/usr/bin/env python3
"""
Batch traccuracy validation for tracking_batch movie outputs.

For every movie, resolve the matching manually validated GT CSV:
    <movie>.zarr -> <GT_ROOT>/<movie>_valid_tracks.csv

Evaluate BOTH tracking outputs when present:
    tracks.geff       = raw Motile solver result
    tracks_clean.geff = post-hoc filtered result

Outputs:
    <batch_root>/traccuracy/
        traccuracy_summary_long.csv   # one row per movie/prediction
        traccuracy_summary_paired.csv # one row per movie, raw vs clean side-by-side
        traccuracy_errors.csv          # only when failures occur
        <movie>/traccuracy_vs_gt.csv   # detailed metrics for that movie

The paired summary is the main human-facing output. It includes:
    - raw and clean prediction sizes
    - raw and clean recall-side traccuracy metrics
    - clean-minus-raw deltas for those metrics
    - post-hoc filtering counts from metadata.json

Usage:
    python batch_traccuracy_v2.py /path/to/tracking_batch
    python batch_traccuracy_v2.py /path/to/tracking_batch --gt-root /path/to/tracking_results
    python batch_traccuracy_v2.py /path/to/tracking_batch --movie sg100_Well5_1018
    python batch_traccuracy_v2.py /path/to/tracking_batch --only-clean
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import networkx as nx
import pandas as pd

from traccuracy import TrackingGraph, run_metrics
from traccuracy.metrics import BasicMetrics, TrackOverlapMetrics
from traccuracy.matchers import PointMatcher
from funtracks.import_export import import_from_geff


DEFAULT_GT_ROOT = Path(
    "/Users/kelpschdj/Documents/DataTecnica/TTU/Data/Sphere/"
    "220725_i11w-hT-M33-I76_sg1035_d10sphere/tracking_results"
)
DEFAULT_MATCH_THRESHOLD = 15.0

# These are the metrics used in the existing traccuracy scripts as the
# informative GT->prediction / recall-side readout.
KEY_METRICS = [
    "Node Recall",
    "Edge Recall",
    "target_effectiveness",
    "track_fractions",
    "True Positive Nodes",
    "False Negative Nodes",
    "True Positive Edges",
    "False Negative Edges",
]


def _extract_results(result):
    if hasattr(result, "results"):
        return dict(result.results)
    if isinstance(result, dict):
        if "results" in result and isinstance(result["results"], dict):
            return dict(result["results"])
        return dict(result)
    raise TypeError(f"Unexpected metric result type: {type(result)}")


def movie_id_from_zarr(zarr_path: str | Path) -> str:
    name = Path(zarr_path).name
    return name[:-5] if name.endswith(".zarr") else Path(name).stem


def load_metadata(movie_dir: Path) -> dict:
    path = movie_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata.json: {path}")
    with path.open() as f:
        return json.load(f)


def gt_csv_to_nx(df: pd.DataFrame) -> nx.DiGraph:
    df = df.copy()
    df.columns = df.columns.str.strip()
    required = {"frame", "x", "y", "particle"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"GT CSV missing columns: {sorted(missing)}")

    g = nx.DiGraph()
    next_id = 0
    for _, group in df.sort_values(["particle", "frame"]).groupby("particle"):
        prev = None
        for _, row in group.iterrows():
            nid = next_id
            next_id += 1
            g.add_node(
                nid,
                t=int(row["frame"]),
                y=float(row["y"]),
                x=float(row["x"]),
            )
            if prev is not None:
                g.add_edge(prev, nid)
            prev = nid
    return g


def geff_to_nx(geff_path: Path) -> nx.DiGraph:
    sg = import_from_geff(
        directory=str(geff_path),
        node_name_map={"time": "t", "pos": ["y", "x"]},
    )
    attrs = sg.graph.node_attrs(unpack=True).to_pandas()
    required = {"tracklet_id", "t", "pos_0", "pos_1"}
    missing = required - set(attrs.columns)
    if missing:
        raise ValueError(
            f"{geff_path}: missing imported attributes: {sorted(missing)}"
        )

    df = pd.DataFrame(
        {
            "particle": attrs["tracklet_id"].astype(int),
            "frame": attrs["t"].astype(int),
            "y": attrs["pos_0"].astype(float),
            "x": attrs["pos_1"].astype(float),
        }
    ).sort_values(["particle", "frame"]).reset_index(drop=True)

    g = nx.DiGraph()
    next_id = 0
    for _, group in df.groupby("particle"):
        prev = None
        for _, row in group.sort_values("frame").iterrows():
            nid = next_id
            next_id += 1
            g.add_node(
                nid,
                t=int(row["frame"]),
                y=float(row["y"]),
                x=float(row["x"]),
            )
            if prev is not None:
                g.add_edge(prev, nid)
            prev = nid
    return g


def make_tracking_graph(g: nx.DiGraph, name: str) -> TrackingGraph:
    return TrackingGraph(
        g,
        frame_key="t",
        location_keys=("y", "x"),
        name=name,
    )


def run_accuracy(
    gt_nx: nx.DiGraph,
    pred_nx: nx.DiGraph,
    gt_name: str,
    pred_name: str,
    match_threshold: float,
) -> dict:
    # traccuracy mutates annotations in-place; always construct fresh wrappers.
    gt_graph = make_tracking_graph(gt_nx.copy(), gt_name)
    pred_graph = make_tracking_graph(pred_nx.copy(), pred_name)
    matcher = PointMatcher(threshold=match_threshold)

    results, _ = run_metrics(
        gt_data=gt_graph,
        pred_data=pred_graph,
        matcher=matcher,
        metrics=[TrackOverlapMetrics(), BasicMetrics()],
    )

    flat = {}
    if isinstance(results, dict):
        for value in results.values():
            if isinstance(value, dict):
                flat.update(value)
        if not flat:
            flat.update(results)
    else:
        for result in results:
            flat.update(_extract_results(result))
    return flat


def count_tracks_nodes_edges(g: nx.DiGraph) -> tuple[int, int, int]:
    return (
        len(list(nx.weakly_connected_components(g))) if g.number_of_nodes() else 0,
        g.number_of_nodes(),
        g.number_of_edges(),
    )


def get_pred_row(
    movie_id: str,
    prediction: str,
    prediction_path: Path,
    gt_nx: nx.DiGraph,
    gt_tracks: int,
    gt_nodes: int,
    gt_edges: int,
    meta: dict,
    match_threshold: float,
) -> dict:
    pred_nx = geff_to_nx(prediction_path)
    pred_tracks, pred_nodes, pred_edges = count_tracks_nodes_edges(pred_nx)

    metrics = run_accuracy(
        gt_nx,
        pred_nx,
        gt_name=f"GT_{movie_id}",
        pred_name=f"{movie_id}_{prediction}",
        match_threshold=match_threshold,
    )

    tracking = meta.get("tracking_params", {})
    posthoc = meta.get("posthoc_results", {})

    row = {
        "movie": movie_id,
        "prediction": prediction,
        "gt_csv": "",
        "prediction_geff": str(prediction_path),
        "run_id": meta.get("run_id"),
        "reference_run_id": meta.get("reference_run_id"),
        "match_threshold_px": match_threshold,
        "gt_tracks": gt_tracks,
        "gt_nodes": gt_nodes,
        "gt_edges": gt_edges,
        "pred_tracks": pred_tracks,
        "pred_nodes": pred_nodes,
        "pred_edges": pred_edges,
        "edge_selected_constant": tracking.get("edge_selected_constant"),
        "max_gap": tracking.get("max_gap"),
        "gap_penalty": tracking.get("gap_penalty"),
        "posthoc_flagged": posthoc.get("n_flagged_tracks"),
        "posthoc_short": posthoc.get("n_short_tracks"),
        "posthoc_removed": posthoc.get("n_tracks_removed"),
        "posthoc_kept": posthoc.get("n_tracks_kept"),
        "posthoc_detections_kept": posthoc.get("n_detections_kept"),
    }
    row.update(metrics)
    return row


def process_movie(
    movie_dir: Path,
    gt_root: Path,
    out_root: Path,
    match_threshold: float,
    only_clean: bool,
) -> list[dict]:
    meta = load_metadata(movie_dir)
    input_zarr = meta.get("input_zarr")
    movie_id = movie_id_from_zarr(input_zarr) if input_zarr else movie_dir.name

    gt_csv = gt_root / f"{movie_id}_valid_tracks.csv"
    if not gt_csv.exists():
        raise FileNotFoundError(f"GT not found: {gt_csv}")

    gt_df = pd.read_csv(gt_csv)
    gt_df.columns = gt_df.columns.str.strip()
    gt_nx = gt_csv_to_nx(gt_df)
    gt_tracks = int(gt_df["particle"].nunique())
    gt_nodes = int(len(gt_df))
    gt_edges = max(gt_nodes - gt_tracks, 0)

    predictions = []
    raw_path = movie_dir / "tracks.geff"
    clean_path = movie_dir / "tracks_clean.geff"

    if not only_clean and raw_path.exists():
        predictions.append(("raw", raw_path))
    if clean_path.exists():
        predictions.append(("clean", clean_path))

    if not predictions:
        raise FileNotFoundError(f"No usable GEFFs in {movie_dir}")

    movie_out = out_root / movie_id
    movie_out.mkdir(parents=True, exist_ok=True)

    rows = []
    for prediction, prediction_path in predictions:
        row = get_pred_row(
            movie_id,
            prediction,
            prediction_path,
            gt_nx,
            gt_tracks,
            gt_nodes,
            gt_edges,
            meta,
            match_threshold,
        )
        row["gt_csv"] = str(gt_csv)
        rows.append(row)

    pd.DataFrame(rows).to_csv(movie_out / "traccuracy_vs_gt.csv", index=False)
    return rows


def make_paired_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    wide = None
    base_cols = [
        "movie",
        "run_id",
        "reference_run_id",
        "gt_csv",
        "gt_tracks",
        "gt_nodes",
        "gt_edges",
        "match_threshold_px",
        "edge_selected_constant",
        "max_gap",
        "gap_penalty",
        "posthoc_flagged",
        "posthoc_short",
        "posthoc_removed",
        "posthoc_kept",
        "posthoc_detections_kept",
    ]

    # Start from one row per movie using the first available row.
    base = (
        long_df.sort_values("prediction")
        .groupby("movie", as_index=False)
        .first()[[c for c in base_cols if c in long_df.columns]]
    )
    wide = base.copy()

    value_cols = [
        "pred_tracks",
        "pred_nodes",
        "pred_edges",
        *KEY_METRICS,
    ]

    for prediction in ["raw", "clean"]:
        sub = long_df[long_df["prediction"] == prediction].copy()
        if sub.empty:
            continue
        keep_cols = ["movie"] + [c for c in value_cols if c in sub.columns]
        sub = sub[keep_cols].copy()
        sub = sub.rename(columns={c: f"{prediction}_{c}" for c in keep_cols if c != "movie"})
        wide = wide.merge(sub, on="movie", how="left")

    # Clean-minus-raw deltas for the main continuous metrics.
    for metric in ["Node Recall", "Edge Recall", "target_effectiveness", "track_fractions"]:
        raw_col = f"raw_{metric}"
        clean_col = f"clean_{metric}"
        if raw_col in wide.columns and clean_col in wide.columns:
            wide[f"delta_clean_minus_raw_{metric}"] = wide[clean_col] - wide[raw_col]

    # Explicit removal fraction where raw track counts exist.
    if "raw_pred_tracks" in wide.columns and "posthoc_removed" in wide.columns:
        wide["posthoc_track_removal_fraction"] = (
            wide["posthoc_removed"] / wide["raw_pred_tracks"]
        )
    if "raw_pred_nodes" in wide.columns and "posthoc_detections_kept" in wide.columns:
        wide["posthoc_node_retention_fraction"] = (
            wide["posthoc_detections_kept"] / wide["raw_pred_nodes"]
        )

    return wide.sort_values("movie").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run traccuracy vs GT for every movie in a tracking batch."
    )
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument("--match-threshold", type=float, default=DEFAULT_MATCH_THRESHOLD)
    parser.add_argument("--only-clean", action="store_true")
    parser.add_argument("--movie", action="append", help="Movie id; repeat for multiple movies")
    parser.add_argument("--strict", action="store_true", help="Stop on first error")
    args = parser.parse_args()

    batch_root = args.batch_root.expanduser().resolve()
    gt_root = args.gt_root.expanduser().resolve()
    out_root = batch_root / "traccuracy"
    out_root.mkdir(parents=True, exist_ok=True)

    movie_dirs = sorted(
        p for p in batch_root.iterdir()
        if p.is_dir() and p.name != "traccuracy"
    )
    if args.movie:
        requested = set(args.movie)
        movie_dirs = [p for p in movie_dirs if p.name in requested]
    if not movie_dirs:
        raise FileNotFoundError(f"No movie directories found in {batch_root}")

    print(f"Batch root : {batch_root}")
    print(f"GT root    : {gt_root}")
    print(f"Threshold  : {args.match_threshold} px")
    print(f"Movies     : {len(movie_dirs)}")
    print()

    all_rows = []
    errors = []

    for movie_dir in movie_dirs:
        print("=" * 80)
        print(f"MOVIE: {movie_dir.name}")
        print("=" * 80)
        try:
            rows = process_movie(
                movie_dir,
                gt_root,
                out_root,
                args.match_threshold,
                args.only_clean,
            )
            all_rows.extend(rows)

            for row in rows:
                print(
                    f"  {row['prediction']:>5}: "
                    f"pred tracks={row['pred_tracks']:>5} | "
                    f"nodes={row['pred_nodes']:>6} | "
                    f"Node Recall={row.get('Node Recall', float('nan')):.3f} | "
                    f"Edge Recall={row.get('Edge Recall', float('nan')):.3f} | "
                    f"Target Eff={row.get('target_effectiveness', float('nan')):.3f}"
                )
        except Exception as exc:
            error = {
                "movie": movie_dir.name,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            errors.append(error)
            print(f"  ERROR: {exc}")
            if args.strict:
                raise

    if not all_rows:
        raise RuntimeError("No successful traccuracy comparisons were produced.")

    long_df = pd.DataFrame(all_rows)
    long_path = out_root / "traccuracy_summary_long.csv"
    long_df.to_csv(long_path, index=False)

    paired_df = make_paired_summary(long_df)
    paired_path = out_root / "traccuracy_summary_paired.csv"
    paired_df.to_csv(paired_path, index=False)

    print()
    print(f"Saved long summary  : {long_path}")
    print(f"Saved paired summary: {paired_path}")

    if errors:
        error_path = out_root / "traccuracy_errors.csv"
        pd.DataFrame(errors).to_csv(error_path, index=False)
        print(f"Saved errors        : {error_path}")
        print(f"Failed movies       : {len(errors)}")

    # Print the key paired readout when both raw and clean were evaluated.
    if {"raw_Node Recall", "clean_Node Recall"}.issubset(paired_df.columns):
        cols = [
            c for c in [
                "movie",
                "raw_pred_tracks",
                "clean_pred_tracks",
                "raw_Node Recall",
                "clean_Node Recall",
                "delta_clean_minus_raw_Node Recall",
                "raw_Edge Recall",
                "clean_Edge Recall",
                "delta_clean_minus_raw_Edge Recall",
                "raw_target_effectiveness",
                "clean_target_effectiveness",
                "delta_clean_minus_raw_target_effectiveness",
                "posthoc_removed",
                "posthoc_kept",
            ]
            if c in paired_df.columns
        ]
        print()
        print("RAW vs CLEAN SUMMARY")
        print(paired_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
