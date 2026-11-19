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
blobs_0_00017.csv. The remaining detection settings are in DETECTION_PARAMS below
(the detection, flow and tracking steps live in src/TrainTracks/).

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
import numpy as np
import zarr
from TrainTracks.config import load_paths, to_relative
from TrainTracks.detection import run_detection
from TrainTracks.flow import compute_optical_flow, write_flow_zarr
from TrainTracks.tracking import (
    annotate_solution_track_ids, run_tracking, solution_to_track_df, track_metrics,
)
from TrainTracks.scoring import (
    COMPOSITE_WEIGHTS, MIN_EDGES_FOR_STATS, MIN_SPAN_FOR_SCORE, MIN_SPAN_KEEP,
    PAUSE_SPEED_PX_PER_FRAME, TURN_FLAG_RADIANS,
    make_flow_sampler, score_all, write_clean_geff,
)


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


# ============================================================================
# Optical flow
# ============================================================================


# ============================================================================
# Candidate graph + solver
# ============================================================================


# ============================================================================
# Metrics
# ============================================================================


# ============================================================================
# Post-hoc flow scoring / filtering
# ============================================================================


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
    flow_scores_csv = movie_out / "flow_scores.csv"
    tracks_clean_geff = movie_out / "tracks_clean.geff"
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
    raw_blobs, blobs_df = run_detection(img_stack, DETECTION_PARAMS)
    blobs_df.to_csv(blobs_csv, index=False)
    print(f"Blobs: {len(raw_blobs)} raw -> {len(blobs_df)} after density filter")

    # Flow
    print("Computing optical flow...")
    flow_raw = compute_optical_flow(img_stack, FLOW_PARAMS)
    write_flow_zarr(flow_raw, flow_zarr)
    print(f"Flow shape: {flow_raw.shape} (explicitly T-1 transitions)")

    # Candidate graph and solve
    solution_nx, cand_stats = run_tracking(blobs_df, flow_raw, T, TRACKING_PARAMS)
    print(
        f"Candidate graph: {cand_stats['n_candidate_nodes']} nodes, "
        f"{cand_stats['n_candidate_edges']} edges"
    )
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
        annotate_solution_track_ids(solution_nx)
        geff.write(solution_nx, str(tracks_geff), zarr_format=3, overwrite=True)
    else:
        print("Warning: empty solution; no tracks.geff written")

    # Post-hoc flow scoring + filtering. Keep the raw solver GEFF untouched.
    posthoc = {
        "n_flagged_tracks": 0,
        "n_short_tracks": 0,
        "n_tracks_removed": 0,
        "n_tracks_kept": 0,
        "n_detections_kept": 0,
    }
    if solution_nx.number_of_nodes() > 0:
        print("Running post-hoc flow scoring / track filtering...")
        track_df = solution_to_track_df(solution_nx)
        flow_vec = make_flow_sampler(flow_raw)
        feats = score_all(track_df, flow_vec)
        feats.to_csv(flow_scores_csv)
        print(f"Wrote flow scores: {flow_scores_csv}")

        clean = write_clean_geff(track_df, feats, tracks_clean_geff)
        posthoc = {
            "n_flagged_tracks": clean["n_flagged_removed"],
            "n_short_tracks": clean["n_short_removed"],
            "n_tracks_removed": clean["n_tracks_removed"],
            "n_tracks_kept": clean["n_clean_tracks"],
            "n_detections_kept": clean["n_clean_detections"],
        }
        print(
            f"Post-hoc filter: flagged={posthoc['n_flagged_tracks']} | "
            f"short={posthoc['n_short_tracks']} | "
            f"removed={posthoc['n_tracks_removed']} | "
            f"kept={posthoc['n_tracks_kept']}"
        )
    else:
        print("Skipping post-hoc filtering for empty solution")

    record = {
        "run_id": run_id,
        "reference_run_id": REFERENCE_RUN_ID,
        "reference_run_note": REFERENCE_RUN_NOTE,
        "timestamp_start": started,
        "timestamp_end": now_iso(),
        # Paths are stored relative (input_zarr: file name under raw_root; output_*: relative to
        # the results root) so moving/renaming a folder cannot leave stale absolute paths.
        "path_schema": "relative-v1",
        "input_zarr": movie_path.name,
        "zarr_array_key": ZARR_ARRAY_KEY,
        "output_dir": to_relative(movie_out, output_root),
        "output_blobs_csv": to_relative(blobs_csv, output_root),
        "output_flow_zarr": to_relative(flow_zarr, output_root),
        "output_geff": to_relative(tracks_geff, output_root),
        "output_flow_scores_csv": to_relative(flow_scores_csv, output_root),
        "output_clean_geff": to_relative(tracks_clean_geff, output_root),
        "image_shape": list(img_stack.shape),
        "n_blobs_raw": int(len(raw_blobs)),
        "n_blobs_input": int(len(blobs_df)),
        "long_track_min_frames": LONG_TRACK_MIN_FRAMES,
        **metrics,
        "detection_params": DETECTION_PARAMS,
        "flow_params": FLOW_PARAMS,
        "tracking_params": TRACKING_PARAMS,
        "posthoc_params": {
            "min_span_for_score": MIN_SPAN_FOR_SCORE,
            "min_span_keep": MIN_SPAN_KEEP,
            "min_edges_for_stats": MIN_EDGES_FOR_STATS,
            "turn_flag_radians": TURN_FLAG_RADIANS,
            "composite_weights": COMPOSITE_WEIGHTS,
            "pause_speed_px_per_frame": PAUSE_SPEED_PX_PER_FRAME,
            "turn_definition": "mean |signed turn| between consecutive moving steps; "
                                 "steps <= pause_speed_px_per_frame are neutral "
                                 "(TrainTracks.scoring)",
        },
        "posthoc_results": posthoc,
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
        "output_clean_geff",
        "n_flagged_tracks",
        "n_short_tracks",
        "n_tracks_removed",
        "n_tracks_kept",
        "n_detections_kept",
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
        nargs="?",
        default=None,
        help="Folder containing *.zarr movie directories (default: raw_root from the config)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root (default: results_root from the config, else <input_dir>/tracking_batch)",
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

    cfg = load_paths(require=False)
    if args.input_dir is not None:
        input_dir = args.input_dir.expanduser().resolve()
    elif cfg is not None:
        input_dir = cfg.raw_root
    else:
        parser.error("input_dir not given and raw_root is not configured (see TrainTracks.example.toml)")
    if not input_dir.is_dir():
        parser.error(f"Input directory not found: {input_dir}")

    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    elif cfg is not None:
        output_dir = cfg.results_root
    else:
        output_dir = input_dir / "tracking_batch"
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
