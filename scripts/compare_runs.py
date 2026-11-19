#!/usr/bin/env python3
"""Compare two pipeline runs (e.g. the archived batch and a fresh run), movie by movie.

For every movie present in both folders it checks: blobs.csv, flow.zarr (bit for bit, frame by
frame), tracks.geff and tracks_clean.geff (same set of tracks by track_uid, same number of
detections), flow_scores.csv (matched by track_uid, every column) and the counts and parameters
in metadata.json. `particle` and node IDs are never compared: they are not stable between runs.
Nothing is written.

Usage:
    python scripts/compare_runs.py <old_root> <new_root> [--movie ID ...] [--skip flow]

Exit code 0 = every movie identical, 1 = at least one difference or a movie missing on one side.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from TrainTracks import compare


def compare_movie(old: Path, new: Path, skip: set[str]) -> list[compare.Check]:
    import zarr

    from TrainTracks.tracks_io import geff_to_track_df

    checks = [compare.compare_metadata(json.loads((old / "metadata.json").read_text()),
                                       json.loads((new / "metadata.json").read_text()))]
    if "blobs" not in skip:
        checks.append(compare.compare_blobs(pd.read_csv(old / "blobs.csv"), pd.read_csv(new / "blobs.csv")))
    if "flow" not in skip:
        fa = zarr.open_group(str(old / "flow.zarr"), mode="r")["flow_raw"]
        fb = zarr.open_group(str(new / "flow.zarr"), mode="r")["flow_raw"]
        checks.append(compare.compare_flow(fa, fb))
    if "tracks" not in skip:
        for name in ("tracks.geff", "tracks_clean.geff"):
            if (old / name).exists() and (new / name).exists():
                checks.append(compare.compare_tracks(name, geff_to_track_df(old / name), geff_to_track_df(new / name)))
            else:
                checks.append(compare.Check(name, (old / name).exists() == (new / name).exists(), "missing on one side"))
    if "scores" not in skip:
        checks.append(compare.compare_scores(pd.read_csv(old / "flow_scores.csv"), pd.read_csv(new / "flow_scores.csv")))
    return checks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old_root", type=Path)
    ap.add_argument("new_root", type=Path)
    ap.add_argument("--movie", action="append")
    ap.add_argument("--skip", action="append", default=[], choices=["blobs", "flow", "tracks", "scores"],
                    help="leave a check out (flow is the slowest)")
    args = ap.parse_args()

    old_root, new_root = args.old_root.expanduser().resolve(), args.new_root.expanduser().resolve()
    movies = lambda r: {p.parent.name for p in r.glob("*/metadata.json")}
    mo, mn = movies(old_root), movies(new_root)
    names = sorted((mo & mn) if not args.movie else set(args.movie))
    if args.movie:  # compare only what was asked for; it must exist on both sides
        missing = [m for m in names if m not in (mo & mn)]
        for m in missing:
            print(f"NOT ON BOTH SIDES: {m}")
        names = [m for m in names if m in (mo & mn)]
        all_ok = not missing
    else:
        for m in sorted(mo ^ mn):
            print(f"NOT ON BOTH SIDES: {m} ({'old' if m in mo else 'new'} only)")
        all_ok = not (mo ^ mn)

    for m in names:
        try:
            checks = compare_movie(old_root / m, new_root / m, set(args.skip))
        except Exception as exc:  # noqa: BLE001
            print(f"{m}: ERROR {type(exc).__name__}: {exc}")
            all_ok = False
            continue
        ok = all(c.ok for c in checks)
        all_ok &= ok
        print(f"{m}: {'PASS' if ok else 'FAIL'}")
        for c in checks:
            if not c.ok or len(names) == 1:
                print(f"    {'ok  ' if c.ok else 'DIFF'} {c.name}: {c.detail}")
            elif "differ" in c.detail:
                print(f"    note {c.name}: {c.detail}")
    print(f"\n{len(names)} movie(s) compared: {'ALL IDENTICAL' if all_ok else 'DIFFERENCES FOUND'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
