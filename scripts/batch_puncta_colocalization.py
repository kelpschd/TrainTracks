#!/usr/bin/env python3
"""
Batch box-based puncta-organelle colocalization for the tracking_batch outputs.

An alternative to batch_lyso_interaction.py's mask-membership approach. Instead of
asking "is this punctum inside a segmented lysosome/mitochondria mask", this crops a
small box centered on each punctum's own (y, x) at each frame and computes Pearson's
correlation coefficient (PCC) between the puncta channel and each organelle channel
within that box. High local correlation means the two channels' intensity patterns
co-vary in that neighborhood -- a continuous, segmentation-free signal that doesn't
depend on the frangi/LoG mask-tuning parameters at all.

For every detection (particle, frame), four PCC values are computed:
    puncta_lysosome_pcc_raw        puncta_lysosome_pcc_bgsub
    puncta_mitochondria_pcc_raw    puncta_mitochondria_pcc_bgsub

"raw" uses the untouched channel; "bgsub" subtracts a per-movie, time-invariant
background estimate first (same formula track_flow_score.py uses for the puncta
channel: median-filter the time-mean projection, subtract, clip at 0).

For every movie:
  1. Load tracks_clean.geff (same track population as batch_lyso_interaction.py,
     via the shared geff_to_track_df/resolve_raw_zarr helpers).
  2. Open the matching raw .zarr s0 (T, C, Y, X).
  3. Background-subtract each of the puncta/lysosome/mitochondria channels once
     per movie.
  4. For every detection, crop a 25x25 px box (clipped at the image edge) centered
     on (y, x) from each channel (raw and bgsub), and compute the four PCC values.
  5. Aggregate to per-track summaries.

Two outputs per movie (under tracking_batch/<movie>/coloc/):
    detections_coloc.csv   one row per track detection
    tracks_coloc.csv       one row per track (mean/median PCC)

Plus, at the batch root:
    organelle_coloc_detections.csv   all detections concatenated
    organelle_coloc_tracks.csv       all per-track rows concatenated
    organelle_coloc_params.json      every param (box size, channels, bg-sub)
    organelle_coloc_errors.csv       only if failures occur

Usage:
    python batch_puncta_colocalization.py /path/to/raw_data/tracking_batch
    python batch_puncta_colocalization.py /path/to/tracking_batch --movie sg100_Well5_1018
    python batch_puncta_colocalization.py /path/to/tracking_batch --raw-root /path/to/raw_data
    python batch_puncta_colocalization.py /path/to/tracking_batch --strict
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.ndimage import median_filter

import batch_lyso_interaction as B


# ============================================================================
# Params (recorded to JSON for reproducibility)
# ============================================================================

BOX_SIZE_PX = 25
BOX_HALF_WIDTH = BOX_SIZE_PX // 2  # 12; box spans [-12, +12] = 25 px

BACKGROUND_MEDIAN_FILTER_SIZE = 10  # same as track_flow_score.py's puncta channel

ORGANELLES = {
    "lysosome": B.LYSOSOME_CHANNEL,
    "mitochondria": B.MITOCHONDRIA_CHANNEL,
}


# ============================================================================
# Background subtraction (per-channel, per-movie, time-invariant)
# ============================================================================


def background_subtract_channel(
    stack: np.ndarray, median_filter_size: int = BACKGROUND_MEDIAN_FILTER_SIZE
) -> np.ndarray:
    """stack is (T, Y, X) for one channel. Returns the background-subtracted,
    negative-clipped version. Background = median-filtered mean-over-time
    projection, subtracted from every frame unchanged (not re-estimated per frame).
    """
    mean_proj = np.mean(stack, axis=0)
    background = median_filter(mean_proj, size=median_filter_size)
    return np.clip(stack - background, 0, None).astype(np.float32)


# ============================================================================
# Box extraction + PCC
# ============================================================================


def extract_box(
    channel_frame: np.ndarray, y: float, x: float, half_width: int = BOX_HALF_WIDTH
) -> tuple[np.ndarray, bool, int]:
    """Crop a (2*half_width+1)-square box centered on (y, x), clipped to bounds.

    Returns (box, was_clipped, n_pixels). was_clipped is True when the box was
    cut short by the image edge (box smaller than the nominal box_size**2).
    """
    H, W = channel_frame.shape
    yy, xx = int(round(y)), int(round(x))
    y0, y1 = yy - half_width, yy + half_width + 1
    x0, x1 = xx - half_width, xx + half_width + 1
    y0c, y1c = max(y0, 0), min(y1, H)
    x0c, x1c = max(x0, 0), min(x1, W)
    box = channel_frame[y0c:y1c, x0c:x1c]
    nominal = (2 * half_width + 1) ** 2
    clipped = box.size != nominal
    return box, clipped, box.size


def box_pcc(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Pearson's correlation coefficient between two equal-shape boxes.

    Returns NaN (rather than raising) when either box is ~constant (zero
    variance), which happens for boxes clipped down to a single pixel or
    genuinely flat regions.
    """
    a = box_a.ravel().astype(np.float64)
    b = box_b.ravel().astype(np.float64)
    if a.size < 2:
        return np.nan
    sa, sb = a.std(), b.std()
    if sa < 1e-9 or sb < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# ============================================================================
# Per-detection colocalization
# ============================================================================


def compute_detection_coloc(
    track_df: pd.DataFrame,
    raw_channels: dict[str, np.ndarray],
    bgsub_channels: dict[str, np.ndarray],
    n_frames: int,
) -> pd.DataFrame:
    """Add box-based PCC columns to track_df (one row per particle/frame).

    raw_channels / bgsub_channels: {"puncta": (T,Y,X), "lysosome": ..., "mitochondria": ...}
    """
    out = track_df.copy().reset_index(drop=True)
    n = len(out)
    cols = {}
    for organelle in ORGANELLES:
        for variant in ("raw", "bgsub"):
            cols[f"puncta_{organelle}_pcc_{variant}"] = np.full(n, np.nan)
    box_n_pixels = np.zeros(n, dtype=int)
    box_clipped = np.zeros(n, dtype=bool)

    for t, sub in out.groupby("frame"):
        t = int(t)
        if t < 0 or t >= n_frames:
            continue
        rows = sub.index.to_numpy()

        puncta_raw_frame = raw_channels["puncta"][t]
        puncta_bgsub_frame = bgsub_channels["puncta"][t]
        organelle_frames = {
            organelle: (raw_channels[organelle][t], bgsub_channels[organelle][t])
            for organelle in ORGANELLES
        }

        for ridx in rows:
            y = float(out.at[ridx, "y"])
            x = float(out.at[ridx, "x"])

            pr_box, pr_clipped, pr_n = extract_box(puncta_raw_frame, y, x)
            pb_box, _, _ = extract_box(puncta_bgsub_frame, y, x)
            box_n_pixels[ridx] = pr_n
            box_clipped[ridx] = pr_clipped

            for organelle, (org_raw_frame, org_bgsub_frame) in organelle_frames.items():
                org_raw_box, _, _ = extract_box(org_raw_frame, y, x)
                org_bgsub_box, _, _ = extract_box(org_bgsub_frame, y, x)
                cols[f"puncta_{organelle}_pcc_raw"][ridx] = box_pcc(pr_box, org_raw_box)
                cols[f"puncta_{organelle}_pcc_bgsub"][ridx] = box_pcc(pb_box, org_bgsub_box)

    out["box_n_pixels"] = box_n_pixels
    out["box_clipped"] = box_clipped
    for name, values in cols.items():
        out[name] = values
    return out


# ============================================================================
# Per-track aggregation
# ============================================================================


def summarize_track_coloc(group: pd.DataFrame) -> dict:
    row = {
        "n_detections": int(len(group)),
        "span": int(group["frame"].max() - group["frame"].min() + 1),
    }
    for organelle in ORGANELLES:
        for variant in ("raw", "bgsub"):
            col = f"puncta_{organelle}_pcc_{variant}"
            vals = group[col].dropna()
            row[f"mean_{col}"] = float(vals.mean()) if len(vals) else np.nan
            row[f"median_{col}"] = float(vals.median()) if len(vals) else np.nan
    return row


# ============================================================================
# Per-movie driver
# ============================================================================


def process_movie(movie_dir: Path, raw_root: Path | None, out_subdir: str) -> dict:
    clean_geff = movie_dir / "tracks_clean.geff"
    if not clean_geff.exists():
        raise FileNotFoundError(f"Missing tracks_clean.geff in {movie_dir}")

    track_df = B.geff_to_track_df(clean_geff)
    if track_df.empty:
        raise ValueError(f"{movie_dir.name}: tracks_clean.geff has no tracks")

    raw_zarr = B.resolve_raw_zarr(movie_dir, raw_root)
    root = zarr.open_group(str(raw_zarr), mode="r")
    img_stack = np.asarray(root[B.ZARR_ARRAY_KEY])
    if img_stack.ndim != 4:
        raise ValueError(f"{raw_zarr}: expected (T,C,Y,X), got {img_stack.shape}")
    T = img_stack.shape[0]

    max_track_frame = int(track_df["frame"].max())
    if max_track_frame >= T:
        raise ValueError(
            f"{movie_dir.name}: track frame {max_track_frame} exceeds stack "
            f"length {T}. Frame convention mismatch between GEFF and zarr."
        )

    raw_channels = {"puncta": img_stack[:, B.PUNCTA_CHANNEL].astype(np.float32)}
    for organelle, ch in ORGANELLES.items():
        raw_channels[organelle] = img_stack[:, ch].astype(np.float32)

    bgsub_channels = {
        name: background_subtract_channel(stack) for name, stack in raw_channels.items()
    }

    det = compute_detection_coloc(track_df, raw_channels, bgsub_channels, T)

    track_rows = []
    for (pid, uid), group in det.groupby(["particle", "track_uid"]):
        row = {"movie": movie_dir.name, "particle": int(pid), "track_uid": uid}
        row.update(summarize_track_coloc(group))
        track_rows.append(row)
    track_summary = pd.DataFrame(track_rows)

    out_dir = movie_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    det["movie"] = movie_dir.name
    det.to_csv(out_dir / "detections_coloc.csv", index=False)
    track_summary.to_csv(out_dir / "tracks_coloc.csv", index=False)

    return {
        "movie": movie_dir.name,
        "raw_zarr": str(raw_zarr),
        "n_tracks": int(track_df["particle"].nunique()),
        "n_detections": int(len(track_df)),
        "n_frames": int(T),
        "detections_csv": str(out_dir / "detections_coloc.csv"),
        "tracks_csv": str(out_dir / "tracks_coloc.csv"),
        "_det_df": det,
        "_track_df": track_summary,
    }


# ============================================================================
# Batch driver
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Batch box-based puncta-organelle colocalization (Pearson's r) for tracks_clean.geff."
    )
    parser.add_argument("batch_root", type=Path)
    parser.add_argument(
        "--raw-root", type=Path, default=None,
        help="Folder of raw <movie>.zarr (fallback if metadata lacks input_zarr)",
    )
    parser.add_argument("--movie", action="append", help="Movie id; repeat for several")
    parser.add_argument(
        "--out-subdir", default="coloc", help="Per-movie output subdir name (default: coloc)"
    )
    parser.add_argument("--strict", action="store_true", help="Stop on first error")
    args = parser.parse_args()

    batch_root = args.batch_root.expanduser().resolve()
    raw_root = args.raw_root.expanduser().resolve() if args.raw_root else None

    if not batch_root.is_dir():
        raise FileNotFoundError(f"Batch root not found: {batch_root}")

    movie_dirs = sorted(
        p for p in batch_root.iterdir()
        if p.is_dir() and (p / "tracks_clean.geff").exists()
    )
    if args.movie:
        requested = set(args.movie)
        movie_dirs = [p for p in movie_dirs if p.name in requested]
    if not movie_dirs:
        raise FileNotFoundError(f"No movie dirs with tracks_clean.geff in {batch_root}")

    print(f"Batch root : {batch_root}")
    print(f"Movies     : {len(movie_dirs)}")
    print(f"Box size   : {BOX_SIZE_PX}x{BOX_SIZE_PX} px, centered on each detection")
    print(f"Channels   : puncta={B.PUNCTA_CHANNEL} " + ", ".join(
        f"{name}={ch}" for name, ch in ORGANELLES.items()
    ))
    print("Metric     : Pearson's correlation coefficient (raw + background-subtracted)\n")

    all_det = []
    all_tracks = []
    errors = []

    for movie_dir in movie_dirs:
        print("=" * 72)
        print(f"MOVIE: {movie_dir.name}")
        print("=" * 72)
        try:
            res = process_movie(movie_dir, raw_root, args.out_subdir)
            all_det.append(res.pop("_det_df"))
            all_tracks.append(res.pop("_track_df"))
            print(f"  tracks={res['n_tracks']} detections={res['n_detections']} "
                  f"frames={res['n_frames']}")
            print(f"  wrote {res['tracks_csv']}")
        except Exception as exc:
            errors.append({
                "movie": movie_dir.name,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            print(f"  ERROR: {exc}")
            if args.strict:
                raise

    if all_det:
        pd.concat(all_det, ignore_index=True).to_csv(
            batch_root / "organelle_coloc_detections.csv", index=False)
    if all_tracks:
        pd.concat(all_tracks, ignore_index=True).to_csv(
            batch_root / "organelle_coloc_tracks.csv", index=False)

    params = {
        "box_size_px": BOX_SIZE_PX,
        "box_half_width_px": BOX_HALF_WIDTH,
        "background_median_filter_size": BACKGROUND_MEDIAN_FILTER_SIZE,
        "channels": {
            "puncta": B.PUNCTA_CHANNEL,
            **{name: ch for name, ch in ORGANELLES.items()},
        },
        "zarr_array_key": B.ZARR_ARRAY_KEY,
        "metric": "pearson_correlation_coefficient",
        "variants": ["raw", "bgsub"],
        "track_source": "tracks_clean.geff",
        "n_movies_processed": len(all_tracks),
        "n_movies_failed": len(errors),
    }
    (batch_root / "organelle_coloc_params.json").write_text(
        json.dumps(params, indent=2, default=str))

    if errors:
        pd.DataFrame(errors).to_csv(
            batch_root / "organelle_coloc_errors.csv", index=False)
        print(f"\n{len(errors)} movie(s) failed; see organelle_coloc_errors.csv")

    print(f"\nDone: {len(all_tracks)} succeeded, {len(errors)} failed, "
          f"{len(movie_dirs)} total")
    print(f"Batch tables:")
    print(f"  {batch_root / 'organelle_coloc_detections.csv'}")
    print(f"  {batch_root / 'organelle_coloc_tracks.csv'}")


if __name__ == "__main__":
    main()
