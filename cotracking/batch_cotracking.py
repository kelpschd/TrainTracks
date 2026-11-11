#!/usr/bin/env python3
"""
Batch cotracking: particle tracks vs. independently tracked lysosome and
mitochondria tracks, for every movie in a tracking_batch folder.

Per movie
---------
  particle tracks : <batch_root>/<movie>/tracks_clean.geff   (same source and loader
                    as batch_lyso_interaction.py, via geff_to_track_df)
  raw image       : <raw_root>/<movie>.zarr  (s0, (T, C, Y, X); ch1 lysosome, ch2 mito)
  organelle tracks: organelle_tracks.track_organelles()  -- your notebook's
                    segmentation / density filter / Kalman linking / net-displacement
                    filter, unchanged. Cached to disk so re-running with different
                    cotracking parameters does not redo the ~15 s/movie segmentation.

NOTE: the hand-annotated sg100_Well5_1018_valid_tracks.csv is ground truth for one
movie only. This batch uses the pipeline's tracks_clean.geff for every movie.

Outputs under <out>/<movie>/ :
    particle_tracks.csv
    tracks_lysosome.csv, tracks_mitochondria.csv        (cache)
    episodes_{lysosome,mitochondria}.csv
    steps_{lysosome,mitochondria}.csv                   (particle steps with state)
    null_{lysosome,mitochondria}.csv                    (per-permutation null stats)
and <out>/cotracking_params.json

Then run summarize_cotracking_velocity.py <out>.

Usage:
    python batch_cotracking.py /path/to/tracking_batch --raw-root /path/to/raw_data
    python batch_cotracking.py /path/to/tracking_batch --raw-root ... --movie sg100_Well5_1018
    python batch_cotracking.py ... --contact-px 15 --min-contact-frames 3 --n-perm 500
    python batch_cotracking.py ... --force-retrack      # redo organelle tracking
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                    # cotracking.py, organelle_tracks.py
sys.path.insert(0, str(HERE.parent / "scripts"))  # batch_lyso_interaction.py

import cotracking as ct  # noqa: E402


def load_particle_tracks(movie_dir: Path) -> pd.DataFrame:
    from batch_lyso_interaction import geff_to_track_df
    df = geff_to_track_df(movie_dir / "tracks_clean.geff")
    return df[["frame", "x", "y", "particle"]].copy()


def load_image(movie_dir: Path, raw_root: Path | None) -> np.ndarray:
    import zarr
    from batch_lyso_interaction import resolve_raw_zarr, ZARR_ARRAY_KEY
    z = resolve_raw_zarr(movie_dir, raw_root)
    arr = np.asarray(zarr.open_group(str(z), mode="r")[ZARR_ARRAY_KEY])
    if arr.ndim != 4:
        raise ValueError(f"{z}: expected (T,C,Y,X), got {arr.shape}")
    return arr


def get_organelle_tracks(movie_dir, raw_root, out_dir, force):
    lp, mp = out_dir / "tracks_lysosome.csv", out_dir / "tracks_mitochondria.csv"
    if lp.exists() and mp.exists() and not force:
        print("  organelle tracks: using cache")
        return pd.read_csv(lp), pd.read_csv(mp)
    from organelle_tracks import track_organelles
    img = load_image(movie_dir, raw_root)
    lys, mito, _, _ = track_organelles(img)
    lys.to_csv(lp, index=False)
    mito.to_csv(mp, index=False)
    return lys, mito


def process_movie(movie_dir, raw_root, out_root, args, seed):
    out_dir = out_root / movie_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    P = load_particle_tracks(movie_dir)
    P.to_csv(out_dir / "particle_tracks.csv", index=False)
    lys, mito = get_organelle_tracks(movie_dir, raw_root, out_dir, args.force_retrack)

    ep_kw = dict(contact_px=args.contact_px, min_contact_frames=args.min_contact_frames,
                 max_gap=args.max_gap, stationary_px=args.stationary_px,
                 cos_thresh=args.cos_thresh)
    summary = {"movie": movie_dir.name, "n_particles": int(P.particle.nunique()),
               "n_lys_tracks": int(lys.particle.nunique()) if len(lys) else 0,
               "n_mito_tracks": int(mito.particle.nunique()) if len(mito) else 0}
    for name, O in (("lysosome", lys), ("mitochondria", mito)):
        ep = ct.find_episodes(P, O, **ep_kw) if len(O) else ct.find_episodes(P, O.reindex(columns=ct.REQUIRED), **ep_kw)
        steps = ct.particle_steps(P, ep)
        ep.to_csv(out_dir / f"episodes_{name}.csv", index=False)
        steps.to_csv(out_dir / f"steps_{name}.csv", index=False)
        if args.n_perm > 0 and len(O):
            null = ct.shifted_null(P, O, n_perm=args.n_perm, mode="local",
                                   shift_range=(args.shift_min, args.shift_max),
                                   seed=seed, **ep_kw)
        else:
            null = pd.DataFrame(columns=["n_episodes", "frac_particles_with_episode",
                                         "particle_frames_in_episode",
                                         "frac_particle_frames_in_episode"])
        null["n_particles"] = summary["n_particles"]
        null["n_particle_frames"] = len(P)
        null.to_csv(out_dir / f"null_{name}.csv", index=False)
        summary[f"{name}_episodes"] = len(ep)
        print(f"  {name}: {len(ep)} episodes"
              + (f" | null mean {null.n_episodes.mean():.2f}" if len(null) else ""))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("batch_root", type=Path)
    ap.add_argument("--raw-root", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="default: <batch_root>/cotracking")
    ap.add_argument("--movie", action="append")
    ap.add_argument("--contact-px", type=float, default=10.0)
    ap.add_argument("--min-contact-frames", type=int, default=3)
    ap.add_argument("--max-gap", type=int, default=1)
    ap.add_argument("--stationary-px", type=float, default=5.0)
    ap.add_argument("--cos-thresh", type=float, default=0.5)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--shift-min", type=float, default=40.0)
    ap.add_argument("--shift-max", type=float, default=120.0)
    ap.add_argument("--force-retrack", action="store_true")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    batch_root = args.batch_root.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve() if args.raw_root else None
    out_root = (args.out or batch_root / "cotracking").expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    movie_dirs = sorted(p for p in batch_root.iterdir()
                        if p.is_dir() and (p / "tracks_clean.geff").exists())
    if args.movie:
        movie_dirs = [p for p in movie_dirs if p.name in set(args.movie)]
    if not movie_dirs:
        raise FileNotFoundError(f"No movie dirs with tracks_clean.geff in {batch_root}")
    print(f"{len(movie_dirs)} movies -> {out_root}\n")

    rows, errors = [], []
    for i, md in enumerate(movie_dirs):
        print("=" * 70, f"\n{md.name}  ({i + 1}/{len(movie_dirs)})")
        try:
            rows.append(process_movie(md, raw_root, out_root, args, seed=1000 + i))
        except Exception as exc:
            errors.append({"movie": md.name, "error": str(exc), "traceback": traceback.format_exc()})
            print(f"  ERROR: {exc}")
            if args.strict:
                raise

    pd.DataFrame(rows).to_csv(out_root / "movie_summary.csv", index=False)
    (out_root / "cotracking_params.json").write_text(json.dumps(
        {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, indent=2))
    if errors:
        pd.DataFrame(errors).to_csv(out_root / "cotracking_errors.csv", index=False)
    print(f"\nDone: {len(rows)} ok, {len(errors)} failed. Next:\n"
          f"  python {HERE / 'summarize_cotracking_velocity.py'} {out_root}")


if __name__ == "__main__":
    main()
