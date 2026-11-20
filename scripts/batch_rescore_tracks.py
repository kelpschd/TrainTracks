#!/usr/bin/env python3
"""
Batch-rescore existing tracking outputs with the current TrainTracks.scoring.

IMPORTANT:
    This does NOT rerun blob detection, optical flow, candidate graph creation,
    Motile, or any track filtering. It only recalculates flow_scores.csv for
    each existing movie using the CURRENT contents of src/TrainTracks/scoring.py.

Expected layout:
    tracking_batch/
        sg100_Well1_1001/
            metadata.json
            tracks.geff
            flow.zarr/flow_raw
            flow_scores.csv       <-- overwritten
        ...

Usage:
    python batch_rescore_tracks.py /path/to/tracking_batch
    python batch_rescore_tracks.py /path/to/tracking_batch --movie sg100_Well1_1003
    python batch_rescore_tracks.py /path/to/tracking_batch --write-clean
    python batch_rescore_tracks.py /path/to/tracking_batch --movie sg100_Well1_1003 --write-clean
    python batch_rescore_tracks.py /path/to/tracking_batch --no-backup
"""

from __future__ import annotations

import argparse
import json
import shutil
import traceback

from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from TrainTracks.scoring import MIN_SPAN_KEEP, make_flow_sampler, score_all, write_clean_geff
from TrainTracks.tracks_io import geff_to_track_df


def read_metadata(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def rescore_movie(movie_dir: Path, no_backup: bool = False, write_clean: bool = False) -> dict:
    meta_path = movie_dir / "metadata.json"
    raw_geff = movie_dir / "tracks.geff"
    flow_path = movie_dir / "flow.zarr"
    scores_path = movie_dir / "flow_scores.csv"

    if not raw_geff.exists():
        raise FileNotFoundError(f"Missing raw GEFF: {raw_geff}")
    if not flow_path.exists():
        raise FileNotFoundError(f"Missing flow.zarr: {flow_path}")

    meta = read_metadata(meta_path) if meta_path.exists() else {}
    flow_root = zarr.open_group(str(flow_path), mode="r")
    flow_raw = np.asarray(flow_root["flow_raw"])
    flow_vec = make_flow_sampler(flow_raw)

    track_df = geff_to_track_df(raw_geff)
    feats = score_all(track_df, flow_vec)

    if scores_path.exists() and not no_backup:
        backup = scores_path.with_name("flow_scores_pre_rescore.csv")
        if not backup.exists():
            shutil.copy2(scores_path, backup)

    feats.to_csv(scores_path)

    clean_result = {}
    if write_clean:
        clean_result = write_clean_geff(
            track_df, feats, movie_dir / "tracks_clean.geff", min_span_keep=MIN_SPAN_KEEP,
        )

    flagged = int(feats["flagged"].sum()) if "flagged" in feats.columns else 0
    eligible = int(feats["eligible"].sum()) if "eligible" in feats.columns else 0

    result = {
        "movie": movie_dir.name,
        "run_id": meta.get("run_id"),
        "n_tracks": int(track_df["particle"].nunique()) if len(track_df) else 0,
        "n_detections": int(len(track_df)),
        "n_eligible": eligible,
        "n_flagged": flagged,
        "score_csv": str(scores_path),
        **clean_result,
    }

    # Include useful summary statistics when the updated scorer provides them.
    for col in [
        "total_turn_radians",
        "net_turn_radians",
        "turn_per_moving_step",
        "n_turn_events_30deg",
        "n_turn_events_45deg",
        "n_turn_events_60deg",
        "pause_fraction",
        "path_length",
        "net_displacement",
        "path_straightness",
        "turning_density",
        "n_self_revisit_pairs",
        "n_self_revisit_points",
        "self_revisit_fraction",
    ]:
        if col in feats.columns:
            vals = pd.to_numeric(feats[col], errors="coerce").dropna()
            result[f"mean_{col}"] = float(vals.mean()) if len(vals) else np.nan
            result[f"median_{col}"] = float(vals.median()) if len(vals) else np.nan

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Recompute flow_scores.csv for existing batch tracking outputs."
    )
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("--movie", action="append", help="Movie id; repeat for multiple movies")
    parser.add_argument("--no-backup", action="store_true", help="Do not save the old flow_scores.csv")
    parser.add_argument("--strict", action="store_true", help="Stop on first failure")
    parser.add_argument("--write-clean", action="store_true", help="Rebuild tracks_clean.geff from the current score decisions")
    args = parser.parse_args()

    batch_root = args.batch_root.expanduser().resolve()

    if not batch_root.is_dir():
        raise FileNotFoundError(f"Batch directory not found: {batch_root}")
    movie_dirs = sorted(
        p for p in batch_root.iterdir()
        if p.is_dir() and (p / "tracks.geff").exists()
    )
    if args.movie:
        requested = set(args.movie)
        movie_dirs = [p for p in movie_dirs if p.name in requested]
    if not movie_dirs:
        raise FileNotFoundError(f"No movie directories with tracks.geff found in {batch_root}")

    print(f"Batch root: {batch_root}")
    print(f"Movies   : {len(movie_dirs)}")
    print()

    rows = []
    errors = []

    for movie_dir in movie_dirs:
        print("=" * 80)
        print(f"MOVIE: {movie_dir.name}")
        print("=" * 80)
        try:
            row = rescore_movie(movie_dir, no_backup=args.no_backup, write_clean=args.write_clean)
            rows.append(row)
            print(
                f"tracks={row['n_tracks']} | detections={row['n_detections']} | "
                f"eligible={row['n_eligible']} | flagged={row['n_flagged']}"
            )
            print(f"wrote: {row['score_csv']}")
            if args.write_clean:
                print(f"clean GEFF: {row['clean_geff']} | kept tracks={row['n_clean_tracks']} | kept detections={row['n_clean_detections']}")
        except Exception as exc:
            err = {
                "movie": movie_dir.name,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            errors.append(err)
            print(f"ERROR: {exc}")
            if args.strict:
                raise

    summary_path = batch_root / "batch_rescore_summary.csv"
    if rows:
        summary = pd.DataFrame(rows)
        # A --movie run must not wipe the other movies' rows: merge by movie.
        if summary_path.exists():
            old = pd.read_csv(summary_path)
            if "movie" in old.columns:
                old = old[~old["movie"].isin(summary["movie"])]
                summary = pd.concat([old, summary], ignore_index=True)
                summary = summary.sort_values("movie").reset_index(drop=True)
        summary.to_csv(summary_path, index=False)
        print(f"\nSaved summary: {summary_path} ({len(summary)} movies)")

    if errors:
        error_path = batch_root / "batch_rescore_errors.csv"
        pd.DataFrame(errors).to_csv(error_path, index=False)
        print(f"Saved errors : {error_path}")


if __name__ == "__main__":
    main()
