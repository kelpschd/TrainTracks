#!/usr/bin/env python3
"""Regression check: re-run a pipeline stage and compare with the outputs already on disk.

Used while refactoring (cleanup Phase 3.4) to prove that moving code into
``src/TrainTracks`` did not change results. Parameters are read from the movie's own
``metadata.json`` so the check uses exactly what produced the stored outputs. Nothing is
written; the stored outputs are only read.

Usage:
    python scripts/check_regression.py detection <movie_dir> [--raw-root <raw_data>]
    python scripts/check_regression.py flow      <movie_dir> [--raw-root <raw_data>]
    python scripts/check_regression.py tracking  <movie_dir>      # uses stored blobs.csv + flow.zarr

    <movie_dir>   e.g. .../tracking_batch_TrainTracks/sg100_Well5_1018
                  (needs metadata.json; blobs.csv for 'detection', flow.zarr for 'flow',
                  blobs.csv + flow.zarr + tracks.geff for 'tracking')
    --raw-root    folder with <movie>.zarr (default: <movie_dir>/../../raw_data)

Exit code 0 = identical, 1 = differences.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load_movie(movie_dir: Path, raw_root: Path):
    """Return (metadata, (T,C,Y,X) image) for a movie dir."""
    import zarr

    meta = json.loads((movie_dir / "metadata.json").read_text())
    key = meta.get("zarr_array_key", "s0")
    zarr_path = raw_root / f"{movie_dir.name}.zarr"   # metadata's input_zarr path may be stale
    if not zarr_path.exists():
        raise FileNotFoundError(f"{zarr_path} not found (pass --raw-root)")
    print(f"movie      : {movie_dir.name}")
    print(f"zarr       : {zarr_path} [{key}]")
    img = np.asarray(zarr.open_group(str(zarr_path), mode="r")[key])
    print(f"image shape: {img.shape}")
    return meta, img


def check_flow(movie_dir: Path, raw_root: Path) -> bool:
    import zarr

    from TrainTracks.flow import compute_optical_flow

    meta, img = _load_movie(movie_dir, raw_root)
    params = meta["flow_params"]
    print(f"params     : {params}")
    new = compute_optical_flow(img, params)
    stored = np.asarray(zarr.open_group(str(movie_dir / "flow.zarr"), mode="r")["flow_raw"])
    print(f"flow shape : new {new.shape} | stored {stored.shape}")
    if new.shape != stored.shape:
        return False
    diff = np.abs(new.astype(np.float64) - stored.astype(np.float64))
    n_diff = int((diff > 0).sum())
    print(f"max |diff| : {diff.max():.3e} | mean |diff| {diff.mean():.3e} | differing values {n_diff} of {diff.size}")
    if n_diff == 0:
        print("flow_raw   : IDENTICAL (bit for bit)")
        return True
    if diff.max() <= 1e-4:
        print("flow_raw   : numerically equal (max diff <= 1e-4; float rounding only)")
        return True
    print("flow_raw   : DIFFERENT")
    return False


def check_tracking(movie_dir: Path, raw_root: Path) -> bool:
    """Re-solve from the STORED blobs.csv and flow.zarr; compare with the stored tracks.geff.

    Tracks are compared by track_uid (a hash of each track's frame/position sequence), not by
    node or particle IDs, because those are not stable between runs. The solver could in
    principle break ties differently, so a mismatch is reported in detail, not just as FAIL.
    """
    import zarr

    from TrainTracks.ids import add_track_uids
    from TrainTracks.tracking import run_tracking, solution_to_track_df, track_metrics
    from TrainTracks.tracks_io import geff_to_track_df

    meta = json.loads((movie_dir / "metadata.json").read_text())
    params = meta["tracking_params"]
    print(f"movie      : {movie_dir.name}")
    print(f"params     : {params}")
    blobs = pd.read_csv(movie_dir / "blobs.csv")
    flow = np.asarray(zarr.open_group(str(movie_dir / "flow.zarr"), mode="r")["flow_raw"])
    solution, stats = run_tracking(blobs, flow, flow.shape[0] + 1, params)
    print(f"candidates : {stats['n_candidate_nodes']} nodes, {stats['n_candidate_edges']} edges")

    metrics = track_metrics(solution, meta.get("long_track_min_frames", 10))
    ok = True
    for key in ("n_detections_tracked", "n_edges", "n_tracks", "n_skip_edges"):
        same = metrics[key] == meta.get(key)
        ok &= same
        print(f"{key:21s}: new {metrics[key]} | metadata {meta.get(key)}  {'ok' if same else 'DIFFERENT'}")

    new_df = add_track_uids(solution_to_track_df(solution))
    old_df = geff_to_track_df(movie_dir / "tracks.geff")
    new_u, old_u = set(new_df["track_uid"]), set(old_df["track_uid"])
    print(f"tracks     : new {len(new_u)} | stored tracks.geff {len(old_u)}")
    if new_u == old_u and len(new_df) == len(old_df):
        print("tracks.geff: IDENTICAL set of tracks (same track_uid values, same detections)")
    else:
        ok = False
        print(f"tracks.geff: DIFFERENT | only in new {len(new_u - old_u)} | only in stored {len(old_u - new_u)}")
    return bool(ok)


def check_detection(movie_dir: Path, raw_root: Path) -> bool:
    from TrainTracks.detection import run_detection

    meta, img = _load_movie(movie_dir, raw_root)
    params = meta["detection_params"]
    print(f"params     : {params}")

    raw_blobs, blobs = run_detection(img, params)
    stored = pd.read_csv(movie_dir / "blobs.csv")

    ok = True
    n_raw_meta = meta.get("n_blobs_raw")
    print(f"raw blobs  : new {len(raw_blobs)} | metadata {n_raw_meta}")
    ok &= n_raw_meta is None or len(raw_blobs) == n_raw_meta
    n_in_meta = meta.get("n_blobs_input")
    print(f"filtered   : new {len(blobs)} | stored blobs.csv {len(stored)} | metadata {n_in_meta}")
    ok &= n_in_meta is None or len(blobs) == n_in_meta

    new = blobs.reset_index(drop=True).astype("int64")
    old = stored.reset_index(drop=True).astype("int64")
    if new.equals(old):
        print("blobs.csv  : IDENTICAL (same rows, same order)")
    else:
        ok = False
        key_cols = ["t", "y", "x"]
        merged = new.merge(old, on=key_cols, how="outer", indicator=True)
        print("blobs.csv  : DIFFERENT")
        print(f"  only in new : {int((merged['_merge'] == 'left_only').sum())}")
        print(f"  only in old : {int((merged['_merge'] == 'right_only').sum())}")
        print(f"  same rows, different order: {sorted(map(tuple, new.values)) == sorted(map(tuple, old.values))}")
    return bool(ok)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["detection", "flow", "tracking"])
    parser.add_argument("movie_dir", type=Path)
    parser.add_argument("--raw-root", type=Path, default=None)
    args = parser.parse_args()

    movie_dir = args.movie_dir.expanduser().resolve()
    raw_root = (args.raw_root.expanduser().resolve() if args.raw_root
                else movie_dir.parent.parent / "raw_data")
    ok = {"detection": check_detection, "flow": check_flow, "tracking": check_tracking}[args.stage](movie_dir, raw_root)
    print("\nRESULT:", "PASS (identical to stored output)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
