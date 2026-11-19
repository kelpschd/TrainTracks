#!/usr/bin/env python3
"""Check the cull cutoff against the ground-truth tracks of ALL movies.

For every movie that has a GT file (``<gt_root>/<movie>_valid_tracks.csv``) and a results
folder with ``flow.zarr``, score the GT tracks with the same scorer the pipeline uses and
report which known-good tracks the turn cutoff would flag. A safe cutoff flags none.

Usage:
    python scripts/calibrate_cull_on_gt.py [<results_root>] [--gt-root DIR]
        [--movie ID ...] [--cutoff 1.4] [--out gt_calibration.csv]

<results_root> defaults to the configured results_root (see TrainTracks.config). Reads only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from TrainTracks import calibration, scoring
from TrainTracks.config import load_paths

DEFAULT_GT_ROOT = Path(__file__).resolve().parents[1] / "ground_truth"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_root", type=Path, nargs="?", default=None)
    ap.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    ap.add_argument("--movie", action="append", help="movie id; repeat for several")
    ap.add_argument("--cutoff", type=float, default=scoring.TURN_FLAG_RADIANS)
    ap.add_argument("--out", type=Path, default=None, help="write the per-track table here (csv)")
    args = ap.parse_args()

    import zarr

    if args.results_root is not None:
        results_root = args.results_root.expanduser().resolve()
    else:
        cfg = load_paths(require=False)
        if cfg is None:
            ap.error("results_root not given and not configured (see TrainTracks.example.toml)")
        results_root = cfg.results_root
    gt_root = args.gt_root.expanduser().resolve()

    movies = sorted(p.name[: -len("_valid_tracks.csv")] for p in gt_root.glob("*_valid_tracks.csv"))
    if args.movie:
        movies = [m for m in movies if m in set(args.movie)]
    print(f"results root: {results_root}\nGT root     : {gt_root}\ncutoff      : {args.cutoff} rad\n")

    parts = []
    for m in movies:
        flow_path = results_root / m / "flow.zarr"
        if not flow_path.exists():
            print(f"  skip {m}: no {flow_path}")
            continue
        gt = pd.read_csv(gt_root / f"{m}_valid_tracks.csv")
        flow_raw = np.asarray(zarr.open_group(str(flow_path), mode="r")["flow_raw"])
        feats = calibration.score_gt_movie(gt, scoring.make_flow_sampler(flow_raw), m, args.cutoff)
        del flow_raw
        parts.append(feats)
        n_flag = int(feats["flagged"].sum())
        print(f"  {m}: {len(feats)} GT tracks, {int(feats['eligible'].sum())} eligible, "
              f"max turn {feats.loc[feats['eligible'], 'turn_incoherence'].max():.2f}, flagged {n_flag}")
    if not parts:
        print("no movies scored")
        return 1

    feats = pd.concat(parts, ignore_index=True)
    elig = feats[feats["eligible"]]
    print(f"\n{feats['movie'].nunique()} movies | {len(feats)} GT tracks | {len(elig)} eligible "
          f"(span >= {scoring.MIN_SPAN_FOR_SCORE})")
    print("turn_incoherence of eligible GT tracks (rad): "
          + " | ".join(f"{q:.0%}: {elig['turn_incoherence'].quantile(q):.2f}" for q in (0.5, 0.9, 0.99, 1.0)))
    print("\nCutoff sweep (eligible GT tracks that would be flagged):")
    print(calibration.cutoff_sweep(feats).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    bad = feats[feats["flagged"]]
    if bad.empty:
        print(f"\nPASS: no GT track is flagged at turn >= {args.cutoff}.")
    else:
        print(f"\nWARNING: {len(bad)} GT track(s) would be flagged at turn >= {args.cutoff}:")
        print(bad[["movie", "particle", "span", "turn_incoherence"]].sort_values("turn_incoherence", ascending=False)
              .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    if args.out:
        feats.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
