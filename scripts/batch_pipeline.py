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
import os
import shutil
import sys
import time
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
from TrainTracks.parallel import run_jobs, shard, shard_from_env
from TrainTracks.resources import resolve_resources
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


class _Stopwatch:
    """Seconds spent in each stage of one movie (recorded in metadata.json)."""

    def __init__(self) -> None:
        self.t = time.perf_counter()
        self.laps: dict[str, float] = {}

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.laps[name] = round(now - self.t, 2)
        self.t = now


def _peak_rss_gb() -> float | None:
    """Peak resident memory of this process (GB). In a pool, each movie runs in a fresh process."""
    try:
        import resource
    except ImportError:  # not available on Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / 1024**3 if sys.platform == "darwin" else peak / 1024**2, 2)  # bytes on macOS, KB on Linux


def movie_id_of(movie_path: Path) -> str:
    return movie_path.name[:-5] if movie_path.name.endswith(".zarr") else movie_path.stem


def process_movie(
    movie_path: Path, output_root: Path, overwrite: bool, skip_existing: bool = False
) -> dict:
    """Run the full pipeline on one movie.

    Everything is written to ``<movie>.partial`` and renamed to ``<movie>`` only when the movie is
    complete, so an interrupted job (crash, killed cluster task) never leaves a half-written movie
    folder, and with ``overwrite`` the old output survives until the new one is finished.
    ``skip_existing`` returns the stored metadata of an already complete movie.
    """
    movie_id = movie_id_of(movie_path)
    run_id = make_run_id()
    final_out = output_root / movie_id
    movie_out = output_root / f"{movie_id}.partial"
    sw = _Stopwatch()
    t_start = time.perf_counter()

    if final_out.exists():
        complete = (final_out / "metadata.json").exists()
        if skip_existing and complete:
            print(f"Skipping {movie_id}: complete output already exists")
            return json.loads((final_out / "metadata.json").read_text())
        if not (overwrite or (skip_existing and not complete)):
            raise FileExistsError(
                f"Output already exists: {final_out}. Use --overwrite to replace it "
                "(or --skip-existing to resume)."
            )
    if movie_out.exists():
        shutil.rmtree(movie_out)  # leftover from an interrupted run
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
    sw.lap("load_s")

    # Detection
    print(f"Detecting blobs (threshold={DETECTION_PARAMS['threshold']})...")
    raw_blobs, blobs_df = run_detection(img_stack, DETECTION_PARAMS)
    blobs_df.to_csv(blobs_csv, index=False)
    print(f"Blobs: {len(raw_blobs)} raw -> {len(blobs_df)} after density filter")
    sw.lap("detection_s")

    # Flow
    print("Computing optical flow...")
    flow_raw = compute_optical_flow(img_stack, FLOW_PARAMS)
    write_flow_zarr(flow_raw, flow_zarr)
    print(f"Flow shape: {flow_raw.shape} (explicitly T-1 transitions)")
    sw.lap("flow_s")

    # Candidate graph and solve
    solution_nx, cand_stats = run_tracking(blobs_df, flow_raw, T, TRACKING_PARAMS)
    print(
        f"Candidate graph: {cand_stats['n_candidate_nodes']} nodes, "
        f"{cand_stats['n_candidate_edges']} edges"
    )
    sw.lap("tracking_s")
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

    sw.lap("write_tracks_s")

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

    sw.lap("scoring_s")
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
        "output_dir": to_relative(final_out, output_root),
        "output_blobs_csv": to_relative(final_out / blobs_csv.name, output_root),
        "output_flow_zarr": to_relative(final_out / flow_zarr.name, output_root),
        "output_geff": to_relative(final_out / tracks_geff.name, output_root),
        "output_flow_scores_csv": to_relative(final_out / flow_scores_csv.name, output_root),
        "output_clean_geff": to_relative(final_out / tracks_clean_geff.name, output_root),
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
        # where the time and memory went (all seconds; tracking_* split from the solver stats)
        "timings_seconds": {
            **sw.laps,
            "tracking_graph_s": cand_stats["seconds_graph"],
            "tracking_edges_s": cand_stats["seconds_edges"],
            "tracking_solve_s": cand_stats["seconds_solve"],
            "total_s": round(time.perf_counter() - t_start, 2),
        },
        "peak_rss_gb": _peak_rss_gb(),
    }
    write_json(metadata_json, record)
    # Complete: swap the finished folder into place (the old one, if any, is removed only now).
    if final_out.exists():
        shutil.rmtree(final_out)
    os.replace(movie_out, final_out)
    print(f"Wrote: {final_out / 'metadata.json'}")
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
    # The post-hoc counts are stored under "posthoc_results"; lift them to the top level so they
    # reach the columns above (they were silently blank before).
    rows = [{**rec, **rec.get("posthoc_results", {})} for rec in records]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _movie_job(job: dict) -> dict:
    """Worker entry point (module level so a spawn-based process pool can import it)."""
    return process_movie(
        Path(job["movie_path"]), Path(job["output_root"]), job["overwrite"], job["skip_existing"]
    )


def _records_from_metadata(output_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(output_dir.glob("*/metadata.json"))]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the validated tracking configuration over a folder of Zarr movies.",
        epilog="Parallelism: --jobs N runs N movies at once in a local process pool; on a SLURM "
        "job array each array task takes its share of the movies automatically "
        "(or pass --task-index / --num-tasks). See docs/parallel_and_hpc.md.",
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
    parser.add_argument("--movie", action="append", help="Movie id to run; repeat for several (default: all)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing per-movie output directory",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip movies whose output is already complete (resume an interrupted batch)",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the batch at the first failed movie",
    )
    parser.add_argument("--jobs", default=None, help="Movies to run at once: an integer or 'auto' (default: auto)")
    parser.add_argument("--threads-per-job", type=int, default=None, help="Cap on threads used by each movie")
    parser.add_argument("--profile", default=None, help="Resource profile from TrainTracks.toml ([profiles.NAME])")
    parser.add_argument("--task-index", type=int, default=None, help="This task's shard index (cluster job arrays)")
    parser.add_argument("--num-tasks", type=int, default=None, help="Total number of shards (cluster job arrays)")
    parser.add_argument("--log-dir", type=Path, default=None, help="Per-movie logs when --jobs > 1 (default: <output>/logs)")
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Do not run anything; rebuild batch_summary.csv from the metadata.json files in the output folder",
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

    summary_csv = output_dir / "batch_summary.csv"
    if args.summarize_only:
        records = _records_from_metadata(output_dir)
        if not records:
            parser.error(f"no <movie>/metadata.json found in {output_dir}")
        write_summary(records, summary_csv)
        print(f"Wrote batch summary for {len(records)} movies: {summary_csv}")
        return 0

    movies = resolve_movie_dirs(input_dir)
    if args.movie:
        wanted = set(args.movie)
        unknown = wanted - {movie_id_of(m) for m in movies}
        if unknown:
            parser.error(f"unknown movie id(s): {sorted(unknown)}")
        movies = [m for m in movies if movie_id_of(m) in wanted]
    if args.num_tasks is not None:
        index, count = (args.task_index or 0), args.num_tasks
    else:
        index, count = shard_from_env()
    if count > 1:
        movies = shard(movies, index, count)
    if not movies:
        parser.error("no movies to run")

    res = resolve_resources(profile=args.profile, jobs=args.jobs, threads_per_job=args.threads_per_job)
    jobs = min(res.jobs, len(movies))
    print(f"Found {len(movies)} Zarr movies" + (f" (shard {index + 1}/{count})" if count > 1 else ""))
    print(f"Output root: {output_dir}")
    print(f"Resources: {jobs} movie(s) at once, {res.threads_per_job} thread(s) each")
    print(f"Reference tracking run: {REFERENCE_RUN_ID}")
    print(json.dumps(json_safe(TRACKING_PARAMS), indent=2))
    print(f"Detection threshold: {DETECTION_PARAMS['threshold']}")

    job_list = [
        {"id": movie_id_of(m), "movie_path": str(m), "output_root": str(output_dir),
         "overwrite": args.overwrite, "skip_existing": args.skip_existing}
        for m in movies
    ]

    def on_done(r, n_done, n_total):
        status = "ok" if r.ok else f"FAILED ({r.error})"
        print(f"[{n_done}/{n_total}] {r.item['id']}: {status} in {r.seconds:.0f}s", flush=True)

    results = run_jobs(
        _movie_job,
        job_list,
        jobs=jobs,
        threads_per_job=res.threads_per_job,
        log_dir=(args.log_dir or output_dir / "logs") if jobs > 1 else None,
        name_of=lambda j: j["id"],
        stop_on_error=args.stop_on_error,
        on_done=on_done,
    )
    records = [r.value for r in results if r.ok]
    failures = [{"input_zarr": r.item["movie_path"], "error": r.error} for r in results if not r.ok]
    for r in results:
        if not r.ok:
            print(f"\nFAILED: {r.item['id']}: {r.error}", file=sys.stderr)
            print(r.traceback, file=sys.stderr)

    suffix = f".shard{index}of{count}" if count > 1 else ""  # shards must not overwrite each other
    if records:
        shard_summary = output_dir / f"batch_summary{suffix}.csv"
        write_summary(records, shard_summary)
        print(f"\nWrote batch summary: {shard_summary}"
              + ("  (combine with --summarize-only after all shards finish)" if count > 1 else ""))

    if failures:
        failures_json = output_dir / f"batch_failures{suffix}.json"
        write_json(failures_json, failures)
        print(f"Wrote failure log: {failures_json}", file=sys.stderr)

    print(
        f"\nBatch complete: {len(records)} succeeded, "
        f"{len(failures)} failed, {len(movies)} total"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
