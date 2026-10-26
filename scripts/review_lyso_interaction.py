#!/usr/bin/env python3
"""
Napari review for ONE movie's organelle-interaction analysis.

Shows, for a single batch-tracked movie:
    - raw puncta channel
    - raw lysosome channel, raw mitochondria channel
    - lysosome labels (recomputed), mitochondria labels (recomputed)
    - clean tracks (tracks_clean.geff)
    - detected puncta positions, optionally split into "inside an organelle"
      vs "not", so you can see whether the masks line up with the spots

CRITICAL: the masks here are recomputed by importing the SAME segmentation
functions + params from batch_lyso_interaction, so what you see is exactly what
produced detections_organelle.csv / tracks_organelle.csv. Nothing is re-tuned.

This reruns segmentation for the one movie (it does not read masks from disk,
since the batch script doesn't persist the label stacks -- only the CSVs). So it
takes as long as segmenting one movie. That's the point of the single-movie QC.

Usage:
    python review_lyso_interaction.py /path/to/tracking_batch --movie sg100_Well1_1003
    python review_lyso_interaction.py /path/to/tracking_batch --movie sg100_Well1_1003 --raw-root /path/to/raw_data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import napari
import numpy as np
import pandas as pd
import zarr

# Import segmentation + params from the batch script so the review CANNOT drift
# from what produced the CSVs. Same-folder import.
import batch_lyso_interaction as B


def resolve_raw_zarr(movie_dir: Path, raw_root: Path | None) -> Path:
    return B.resolve_raw_zarr(movie_dir, raw_root)


def main():
    parser = argparse.ArgumentParser(
        description="Napari QC for one movie's organelle interaction analysis."
    )
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("--movie", required=True, help="Movie id, e.g. sg100_Well1_1003")
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--out-subdir", default="organelle",
                        help="Per-movie output subdir the batch wrote to (default: organelle)")
    args = parser.parse_args()

    batch_root = args.batch_root.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve() if args.raw_root else None
    movie_dir = batch_root / args.movie
    if not movie_dir.is_dir():
        raise FileNotFoundError(f"Movie dir not found: {movie_dir}")

    clean_geff = movie_dir / "tracks_clean.geff"
    if not clean_geff.exists():
        raise FileNotFoundError(f"Missing tracks_clean.geff in {movie_dir}")

    # Tracks (same loader as the batch).
    track_df = B.geff_to_track_df(clean_geff)
    print(f"Clean tracks: {track_df['particle'].nunique()} "
          f"({len(track_df)} detections)")

    # Raw image.
    raw_zarr = resolve_raw_zarr(movie_dir, raw_root)
    print(f"Raw zarr: {raw_zarr}")
    root = zarr.open_group(str(raw_zarr), mode="r")
    img = np.asarray(root[B.ZARR_ARRAY_KEY])
    if img.ndim != 4:
        raise ValueError(f"Expected (T,C,Y,X), got {img.shape}")
    T = img.shape[0]

    # Recompute masks with the batch's exact functions + params.
    print("Segmenting (same params as batch)...")
    lyso_labels, mito_labels, area_log = B.segment_stack(img)
    area_df = pd.DataFrame(area_log)
    print("Mask-area summary (watch for zero / frame-filling frames):")
    print(area_df.describe().loc[["min", "mean", "max"]].to_string())

    # Match detections so we can color spots by inside/outside (recompute, cheap).
    det = B.match_detections_to_masks(lyso_labels, track_df, "lysosome", T)
    det = B.match_detections_to_masks(mito_labels, det, "mitochondria", T)

    # Points split by lysosome membership (napari Points want [t, y, x]).
    inside_l = det[det["lysosome_inside"].astype(bool)]
    outside_l = det[~det["lysosome_inside"].astype(bool)]

    # Load the batch's per-track summary if present, for console context.
    tracks_csv = movie_dir / args.out_subdir / "tracks_organelle.csv"
    if tracks_csv.exists():
        ts = pd.read_csv(tracks_csv)
        print(f"\nPer-track summary ({len(ts)} tracks) — lysosome highlights:")
        cols = [c for c in ["particle", "span", "lysosome_frac_inside",
                            "lysosome_n_assoc_events", "lysosome_label_persistence"]
                if c in ts.columns]
        with pd.option_context("display.max_rows", 20):
            print(ts.sort_values("lysosome_frac_inside", ascending=False)[cols].head(20).to_string(index=False))

    # ------------------------------------------------------------------
    # Napari
    # ------------------------------------------------------------------
    viewer = napari.Viewer(title=f"Organelle review: {args.movie}")

    viewer.add_image(img[:, B.PUNCTA_CHANNEL], name="Puncta (ch0)",
                     colormap="gray")
    viewer.add_image(img[:, B.LYSOSOME_CHANNEL], name="Lyso raw (ch1)",
                     colormap="green", blending="additive", visible=False)
    viewer.add_image(img[:, B.MITOCHONDRIA_CHANNEL], name="Mito raw (ch2)",
                     colormap="magenta", blending="additive", visible=False)

    viewer.add_labels(lyso_labels, name="Lysosome labels", opacity=0.45)
    viewer.add_labels(mito_labels, name="Mito labels", opacity=0.45, visible=False)

    viewer.add_tracks(
        track_df[["particle", "frame", "y", "x"]].to_numpy(),
        name="Clean tracks",
    )

    if not outside_l.empty:
        viewer.add_points(
            outside_l[["frame", "y", "x"]].to_numpy(),
            name="Puncta: NOT in lyso", size=10,
            face_color="transparent", border_color="white",
            border_width=0.15, opacity=0.5,
        )
    if not inside_l.empty:
        viewer.add_points(
            inside_l[["frame", "y", "x"]].to_numpy(),
            name="Puncta: IN lyso", size=10,
            face_color="transparent", border_color="yellow",
            border_width=0.25, opacity=0.8,
        )

    print("\nWhat to check visually:")
    print("  - Lysosome labels sit on actual lyso structures in 'Lyso raw'")
    print("    (toggle it on; it starts hidden).")
    print("  - Mito labels aren't exploding across the frame (channel 2 was the")
    print("    saturated one -- toggle 'Mito labels' + 'Mito raw' and scrub time).")
    print("  - Yellow-outlined puncta ('IN lyso') actually land on colored")
    print("    lysosome blobs; white ones sit off them.")
    print("  - Scrub a high-frac_inside track: does its spot stay on ONE colored")
    print("    blob (real co-travel) or hop between blobs (label_persistence low)?")

    napari.run()


if __name__ == "__main__":
    main()
