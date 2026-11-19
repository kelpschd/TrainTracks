#!/usr/bin/env python3
"""Regression check: re-run a pipeline stage and compare with the outputs already on disk.

Used while refactoring (cleanup Phase 3.4) to prove that moving code into
``src/TrainTracks`` did not change results. Parameters are read from the movie's own
``metadata.json`` so the check uses exactly what produced the stored outputs. Nothing is
written; the stored outputs are only read.

Usage:
    python scripts/check_regression.py detection <movie_dir> [--raw-root <raw_data>]

    <movie_dir>   e.g. .../tracking_batch_TrainTracks/sg100_Well5_1018
                  (needs metadata.json and blobs.csv)
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


def check_detection(movie_dir: Path, raw_root: Path) -> bool:
    import zarr

    from TrainTracks.detection import run_detection

    meta = json.loads((movie_dir / "metadata.json").read_text())
    params = meta["detection_params"]
    key = meta.get("zarr_array_key", "s0")
    zarr_path = raw_root / f"{movie_dir.name}.zarr"   # metadata's input_zarr path may be stale
    if not zarr_path.exists():
        raise FileNotFoundError(f"{zarr_path} not found (pass --raw-root)")

    print(f"movie      : {movie_dir.name}")
    print(f"zarr       : {zarr_path} [{key}]")
    print(f"params     : {params}")
    img = np.asarray(zarr.open_group(str(zarr_path), mode="r")[key])
    print(f"image shape: {img.shape}")

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
    parser.add_argument("stage", choices=["detection"])
    parser.add_argument("movie_dir", type=Path)
    parser.add_argument("--raw-root", type=Path, default=None)
    args = parser.parse_args()

    movie_dir = args.movie_dir.expanduser().resolve()
    raw_root = (args.raw_root.expanduser().resolve() if args.raw_root
                else movie_dir.parent.parent / "raw_data")
    ok = {"detection": check_detection}[args.stage](movie_dir, raw_root)
    print("\nRESULT:", "PASS (identical to stored output)" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
