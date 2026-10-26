#!/usr/bin/env python3
"""
Batch-rescore existing tracking outputs with the current track_flow_score.py.

IMPORTANT:
    This does NOT rerun blob detection, optical flow, candidate graph creation,
    Motile, or any track filtering. It only recalculates flow_scores.csv for
    each existing movie using the CURRENT contents of track_flow_score.py.

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
import importlib.util
import json
import shutil
import traceback

import networkx as nx
import geff
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from funtracks.import_export import import_from_geff


DEFAULT_SCORER = Path(__file__).with_name("track_flow_score.py")


def load_scorer(path: Path):
    spec = importlib.util.spec_from_file_location("track_flow_score_batch", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load scorer module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_metadata(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def geff_to_track_df(scorer, geff_path: Path) -> pd.DataFrame:
    sg = import_from_geff(
        directory=str(geff_path),
        node_name_map={"time": "t", "pos": ["y", "x"]},
    )
    # Reuse the user's current scorer conversion logic where available.
    return scorer.geff_to_track_df(sg)


def write_clean_geff(movie_dir: Path, track_df: pd.DataFrame, feats: pd.DataFrame, min_span_keep: int) -> dict:
    """Rebuild tracks_clean.geff from the CURRENT score decisions.

    The raw tracks.geff is never modified. Short tracks (span < MIN_SPAN_KEEP)
    and tracks with scorer flag == True are excluded. The original numeric
    particle/tracklet_id values are preserved in the clean GEFF.
    """
    if "flagged" not in feats.columns:
        raise ValueError("Score table is missing required 'flagged' column")
    if "span" not in feats.columns:
        raise ValueError("Score table is missing required 'span' column")

    flagged_ids = set(feats.index[feats["flagged"].fillna(False).astype(bool)].astype(int))
    short_ids = set(feats.index[feats["span"].astype(float) < min_span_keep].astype(int))
    drop_ids = flagged_ids | short_ids

    keep = track_df[~track_df["particle"].astype(int).isin(drop_ids)].copy()
    keep = keep.sort_values(["particle", "frame"]).reset_index(drop=True)

    g = nx.DiGraph()
    next_id = 0
    for pid, group in keep.groupby("particle", sort=True):
        prev = None
        for _, row in group.iterrows():
            nid = next_id
            next_id += 1
            g.add_node(
                nid,
                t=int(row["frame"]),
                y=float(row["y"]),
                x=float(row["x"]),
                track_id=int(pid),
                tracklet_id=int(pid),
            )
            if prev is not None:
                g.add_edge(prev, nid)
            prev = nid

    out = movie_dir / "tracks_clean.geff"
    geff.write(g, str(out), zarr_format=3, overwrite=True)

    return {
        "clean_geff": str(out),
        "n_clean_tracks": int(keep["particle"].nunique()) if len(keep) else 0,
        "n_clean_detections": int(len(keep)),
        "n_short_removed": len(short_ids),
        "n_flagged_removed": len(flagged_ids),
        "n_tracks_removed": len(drop_ids),
    }


def rescore_movie(movie_dir: Path, scorer, no_backup: bool = False, write_clean: bool = False) -> dict:
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
    flow_vec = scorer.make_flow_sampler(flow_raw)

    track_df = geff_to_track_df(scorer, raw_geff)
    feats = scorer.score_all(track_df, flow_vec)

    if scores_path.exists() and not no_backup:
        backup = scores_path.with_name("flow_scores_pre_rescore.csv")
        if not backup.exists():
            shutil.copy2(scores_path, backup)

    feats.to_csv(scores_path)

    clean_result = {}
    if write_clean:
        clean_result = write_clean_geff(
            movie_dir,
            track_df,
            feats,
            min_span_keep=int(getattr(scorer, "MIN_SPAN_KEEP", 5)),
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
    parser.add_argument(
        "--scorer", type=Path, default=DEFAULT_SCORER,
        help=f"Updated track_flow_score.py to import (default: {DEFAULT_SCORER})",
    )
    parser.add_argument("--movie", action="append", help="Movie id; repeat for multiple movies")
    parser.add_argument("--no-backup", action="store_true", help="Do not save the old flow_scores.csv")
    parser.add_argument("--strict", action="store_true", help="Stop on first failure")
    parser.add_argument("--write-clean", action="store_true", help="Rebuild tracks_clean.geff from the current score decisions")
    args = parser.parse_args()

    batch_root = args.batch_root.expanduser().resolve()
    scorer_path = args.scorer.expanduser().resolve()

    if not batch_root.is_dir():
        raise FileNotFoundError(f"Batch directory not found: {batch_root}")
    if not scorer_path.exists():
        raise FileNotFoundError(f"Scorer not found: {scorer_path}")

    scorer = load_scorer(scorer_path)

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
    print(f"Scorer   : {scorer_path}")
    print(f"Movies   : {len(movie_dirs)}")
    print()

    rows = []
    errors = []

    for movie_dir in movie_dirs:
        print("=" * 80)
        print(f"MOVIE: {movie_dir.name}")
        print("=" * 80)
        try:
            row = rescore_movie(movie_dir, scorer, no_backup=args.no_backup, write_clean=args.write_clean)
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
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"\nSaved summary: {summary_path}")

    if errors:
        error_path = batch_root / "batch_rescore_errors.csv"
        pd.DataFrame(errors).to_csv(error_path, index=False)
        print(f"Saved errors : {error_path}")


if __name__ == "__main__":
    main()
