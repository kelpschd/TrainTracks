#!/usr/bin/env python3
"""
Batch organelle-interaction analysis for the tracking_batch outputs.

For every movie:
  1. Load tracks_clean.geff  (post-cull tracks; particle, frame, y, x, track_uid)
  2. Open the matching raw .zarr s0  (T, C, Y, X), channels:
         0 = puncta (tracked), 1 = lysosome, 2 = mitochondria, 3 = BFP (ignored)
  3. Segment lyso + mito per frame using the SAME functions you tuned
     (generate_lysosome_mask / generate_mitochondria_mask, lifted unchanged).
  4. Match every track detection to the nearest organelle of each type:
         <org>_label, <org>_centroid_dist, <org>_inside, <org>_edge_dist
  5. Aggregate to per-track interaction features (dwell, events, co-movement).

Two outputs per movie (under tracking_batch/<movie>/organelle/):
    detections_organelle.csv   one row per track detection
    tracks_organelle.csv       one row per track (interaction summary)

Plus, at the batch root:
    organelle_interaction_detections.csv   all detections concatenated
    organelle_interaction_tracks.csv       all per-track rows concatenated
    organelle_interaction_params.json      every segmentation/analysis param
    organelle_interaction_errors.csv       only if failures occur

ALL DISTANCES ARE IN PIXELS. The zarr s0 is read as a plain array (no scale
transform applied), exactly as the tracking pipeline reads it, so track coords
and mask coords share one pixel frame. Convert to microns downstream with the
ND2 axesCalibration factor if you need physical units.

Usage:
    python batch_lyso_interaction.py /path/to/raw_data/tracking_batch
    python batch_lyso_interaction.py /path/to/tracking_batch --movie sg100_Well1_1003
    python batch_lyso_interaction.py /path/to/tracking_batch --raw-root /path/to/raw_data
    python batch_lyso_interaction.py /path/to/tracking_batch --strict
"""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt, gaussian_laplace
from scipy.spatial import cKDTree
from scipy.stats import norm
from skimage import filters, morphology
from skimage.measure import label, regionprops

from funtracks.import_export import import_from_geff


# ============================================================================
# Channel assignment + segmentation params  (recorded to JSON for reproducibility)
# ============================================================================

PUNCTA_CHANNEL = 0
LYSOSOME_CHANNEL = 1
MITOCHONDRIA_CHANNEL = 2
# channel 3 (BFP) is intentionally ignored.

ZARR_ARRAY_KEY = "s0"

LYSO_PARAMS = dict(
    intensity_scaling_param=(3, 19),
    min_lysosome_area=15,
    blur_sigma=1,
    log_params=((5, 0.09), (2.5, 0.07), (1, 0.01)),
    vesselness_sigma=(1,),
    vesselness_cutoff=0.15,
)

MITO_PARAMS = dict(
    intensity_scaling_param=(3.5, 15),
    min_mitochondria_area=10,
    blur_sigma=1,
    log_params=((5, 0.09), (2.5, 0.07), (1, 0.01)),
    vesselness_sigma=(1.5,),
    vesselness_cutoff=0.16,
)

# A detection is "associated" with an organelle when it sits inside the mask.
# edge_dist is signed-by-context (inside -> distance to boundary from within;
# outside -> distance to nearest mask pixel). "near" uses this outside distance.
NEAR_EDGE_DIST_PX = 9.0   # <= this (while outside) counts as "near but not in"


# ============================================================================
# Segmentation  (your functions, unchanged except param plumbing)
# ============================================================================


def generate_lysosome_mask(frame, intensity_scaling_param, min_lysosome_area,
                           blur_sigma, log_params, vesselness_sigma,
                           vesselness_cutoff):
    m, s = norm.fit(frame.flatten())
    stretch_min = max(m - intensity_scaling_param[0] * s, frame.min())
    stretch_max = min(m + intensity_scaling_param[1] * s, frame.max())
    frame = np.clip(frame, stretch_min, stretch_max)
    denom = (stretch_max - stretch_min) or 1e-8
    image_norm = (frame - stretch_min) / denom

    blurred = filters.gaussian(image_norm, sigma=blur_sigma)

    log_masks = []
    for sigma, cutoff in log_params:
        log_img = -1 * (sigma ** 2) * gaussian_laplace(blurred, sigma=sigma)
        log_masks.append(log_img > cutoff)
    log_mask = np.logical_or.reduce(log_masks)

    vesselness = filters.frangi(blurred, sigmas=vesselness_sigma,
                                black_ridges=False) > vesselness_cutoff

    combined = np.logical_or(log_mask, vesselness)
    filled = ndi.binary_fill_holes(combined)
    cleaned = morphology.remove_small_objects(filled, min_size=min_lysosome_area)
    return cleaned.astype(np.uint8)


def generate_mitochondria_mask(frame, intensity_scaling_param,
                               min_mitochondria_area, blur_sigma, log_params,
                               vesselness_sigma, vesselness_cutoff):
    m, s = norm.fit(frame.flatten())
    stretch_min = max(m - intensity_scaling_param[0] * s, frame.min())
    stretch_max = min(m + intensity_scaling_param[1] * s, frame.max())
    clipped = np.clip(frame, stretch_min, stretch_max)
    image_norm = (clipped - stretch_min) / (stretch_max - stretch_min + 1e-8)

    blurred = filters.gaussian(image_norm, sigma=blur_sigma)

    log_masks = []
    for sigma, cutoff in log_params:
        log_response = -1 * (sigma ** 2) * gaussian_laplace(blurred, sigma=sigma)
        log_masks.append(log_response > cutoff)
    log_mask = np.logical_or.reduce(log_masks)

    vesselness = filters.frangi(blurred, sigmas=vesselness_sigma,
                                black_ridges=False)
    vessel_mask = vesselness > vesselness_cutoff

    combined = np.logical_or(log_mask, vessel_mask)
    mito = morphology.remove_small_objects(combined, min_size=min_mitochondria_area)
    return mito.astype(np.uint8)


# ============================================================================
# Track loading  (same pattern as track_flow_score.geff_to_track_df, with UID)
# ============================================================================


def track_uid_for_group(group: pd.DataFrame) -> str:
    g = group.sort_values("frame")[["frame", "y", "x"]]
    payload = [
        (int(r.frame),
         int(round(float(r.y) * 1_000_000)),
         int(round(float(r.x) * 1_000_000)))
        for r in g.itertuples(index=False)
    ]
    return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()


def geff_to_track_df(geff_path: Path) -> pd.DataFrame:
    sg = import_from_geff(
        directory=str(geff_path),
        node_name_map={"time": "t", "pos": ["y", "x"]},
    )
    attrs = sg.graph.node_attrs(unpack=True).to_pandas()
    required = {"tracklet_id", "t", "pos_0", "pos_1"}
    missing = required - set(attrs.columns)
    if missing:
        raise ValueError(f"{geff_path}: missing imported attributes {sorted(missing)}")
    df = pd.DataFrame({
        "particle": attrs["tracklet_id"].astype(int),
        "frame": attrs["t"].astype(int),
        "y": attrs["pos_0"].astype(float),
        "x": attrs["pos_1"].astype(float),
    }).sort_values(["particle", "frame"]).reset_index(drop=True)
    uid_map = {int(pid): track_uid_for_group(g)
               for pid, g in df.groupby("particle", sort=False)}
    df["track_uid"] = df["particle"].astype(int).map(uid_map)
    return df


# ============================================================================
# Per-frame matching  (your match_puncta_to_organelles_fast, frame-safe)
# ============================================================================


def match_detections_to_masks(label_stack: np.ndarray, det_df: pd.DataFrame,
                              organelle: str, n_frames: int) -> pd.DataFrame:
    """Add <organelle>_{label,centroid_dist,inside,edge_dist} to det_df.

    label_stack is indexed by ABSOLUTE frame (0..n_frames-1). We look each
    detection's frame up directly instead of enumerate-aligning, so tracks that
    don't start at frame 0 (or skip frames) can't silently misalign.
    """
    out = det_df.copy().reset_index(drop=True)
    n = len(out)
    lab = np.full(n, np.nan)
    cdist = np.full(n, np.nan)
    inside = np.zeros(n, dtype=bool)
    edist = np.full(n, np.nan)

    H, W = label_stack.shape[1], label_stack.shape[2]

    for t, sub in out.groupby("frame"):
        t = int(t)
        if t < 0 or t >= n_frames:
            # Detection frame outside the segmented stack; leave as NaN/False.
            continue
        frame_labels = label_stack[t]
        props = regionprops(frame_labels)
        rows = sub.index.to_numpy()

        if not props:
            continue  # no organelles this frame -> NaN label/dist, inside False

        centroids = [(p.label, p.centroid[0], p.centroid[1]) for p in props]
        coords = np.array([(cy, cx) for _, cy, cx in centroids])
        tree = cKDTree(coords)

        binary = frame_labels > 0
        dist_inside = distance_transform_edt(binary)
        dist_outside = distance_transform_edt(~binary)

        for ridx in rows:
            y = float(out.at[ridx, "y"])
            x = float(out.at[ridx, "x"])
            d, idx = tree.query((y, x))
            label_id = centroids[idx][0]

            yy = min(max(int(round(y)), 0), H - 1)
            xx = min(max(int(round(x)), 0), W - 1)
            is_in = bool(frame_labels[yy, xx] > 0)
            ed = float(dist_inside[yy, xx]) if is_in else float(dist_outside[yy, xx])

            lab[ridx] = label_id
            cdist[ridx] = float(d)
            inside[ridx] = is_in
            edist[ridx] = ed

    out[f"{organelle}_label"] = lab
    out[f"{organelle}_centroid_dist"] = cdist
    out[f"{organelle}_inside"] = inside
    out[f"{organelle}_edge_dist"] = edist
    return out


# ============================================================================
# Per-track aggregation
# ============================================================================


def _count_events(inside_seq: np.ndarray) -> tuple[int, int, list[int]]:
    """From a boolean inside-sequence (frame-ordered): return
    (n_association_events, n_dissociation_events, list_of_dwell_lengths).

    An association event = a False/absent -> True transition (entering).
    A dissociation event = True -> False transition (leaving).
    Dwell lengths are the lengths (in detections) of each contiguous True run.
    """
    seq = inside_seq.astype(bool)
    assoc = 0
    dissoc = 0
    dwell = []
    run = 0
    prev = False
    for v in seq:
        if v and not prev:
            assoc += 1
            run = 1
        elif v and prev:
            run += 1
        elif (not v) and prev:
            dissoc += 1
            dwell.append(run)
            run = 0
        prev = v
    if prev:
        dwell.append(run)
    return assoc, dissoc, dwell


def summarize_track(group: pd.DataFrame, organelle: str) -> dict:
    """Interaction features for one track against one organelle type.

    group is frame-sorted rows for a single particle, with the organelle match
    columns already attached.
    """
    g = group.sort_values("frame")
    inside = g[f"{organelle}_inside"].to_numpy()
    labels = g[f"{organelle}_label"].to_numpy()
    edge = g[f"{organelle}_edge_dist"].to_numpy()
    cdist = g[f"{organelle}_centroid_dist"].to_numpy()
    n = len(g)

    assoc, dissoc, dwell = _count_events(inside)
    frac_inside = float(np.mean(inside)) if n else np.nan

    # "near but outside": outside AND edge_dist <= NEAR_EDGE_DIST_PX
    outside = ~inside.astype(bool)
    near_mask = outside & (np.nan_to_num(edge, nan=np.inf) <= NEAR_EDGE_DIST_PX)
    frac_near = float(np.mean(near_mask)) if n else np.nan

    # Co-movement: among consecutive frames where the detection is inside an
    # organelle in BOTH frames, how often is it the SAME organelle label? High
    # => sustained association with one organelle (real co-travel). Low => the
    # punctum is skimming across different organelles (often a spurious "match").
    same = 0
    comp = 0
    for i in range(n - 1):
        if bool(inside[i]) and bool(inside[i + 1]):
            comp += 1
            if not np.isnan(labels[i]) and labels[i] == labels[i + 1]:
                same += 1
    label_persistence = float(same / comp) if comp else np.nan

    # Distances while inside vs outside (edge_dist) for a simple contrast.
    mean_edge_inside = float(np.nanmean(edge[inside.astype(bool)])) if inside.any() else np.nan
    mean_edge_outside = float(np.nanmean(edge[outside])) if outside.any() else np.nan

    return {
        f"{organelle}_frac_inside": frac_inside,
        f"{organelle}_frac_near": frac_near,
        f"{organelle}_n_assoc_events": assoc,
        f"{organelle}_n_dissoc_events": dissoc,
        f"{organelle}_max_dwell": int(max(dwell)) if dwell else 0,
        f"{organelle}_mean_dwell": float(np.mean(dwell)) if dwell else 0.0,
        f"{organelle}_n_dwell_runs": len(dwell),
        f"{organelle}_label_persistence": label_persistence,
        f"{organelle}_mean_centroid_dist": float(np.nanmean(cdist)) if n else np.nan,
        f"{organelle}_mean_edge_dist_inside": mean_edge_inside,
        f"{organelle}_mean_edge_dist_outside": mean_edge_outside,
    }


# ============================================================================
# Per-movie driver
# ============================================================================


def resolve_raw_zarr(movie_dir: Path, raw_root: Path | None) -> Path:
    """Find the raw <movie>.zarr. Prefer metadata.json's input_zarr; else guess."""
    meta_path = movie_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        iz = meta.get("input_zarr")
        if iz and Path(iz).exists():
            return Path(iz)
    # Fall back to raw_root / <movie>.zarr
    candidate = (raw_root or movie_dir.parent.parent) / f"{movie_dir.name}.zarr"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not locate raw zarr for {movie_dir.name} "
        f"(checked metadata.json input_zarr and {candidate})"
    )


def segment_stack(img_stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Segment lyso + mito for all frames; return (lyso_labels, mito_labels, area_log)."""
    T = img_stack.shape[0]
    lyso_labels = np.zeros((T, img_stack.shape[2], img_stack.shape[3]), dtype=np.int32)
    mito_labels = np.zeros_like(lyso_labels)
    area_log = []
    for t in range(T):
        lyso_m = generate_lysosome_mask(
            img_stack[t, LYSOSOME_CHANNEL].astype(float), **LYSO_PARAMS)
        mito_m = generate_mitochondria_mask(
            img_stack[t, MITOCHONDRIA_CHANNEL].astype(float), **MITO_PARAMS)
        lyso_labels[t] = label(lyso_m)
        mito_labels[t] = label(mito_m)
        area_log.append({
            "frame": t,
            "lyso_area_px": int(lyso_m.sum()),
            "lyso_n_objects": int(lyso_labels[t].max()),
            "mito_area_px": int(mito_m.sum()),
            "mito_n_objects": int(mito_labels[t].max()),
        })
    return lyso_labels, mito_labels, area_log


def process_movie(movie_dir: Path, raw_root: Path | None, out_subdir: str) -> dict:
    clean_geff = movie_dir / "tracks_clean.geff"
    if not clean_geff.exists():
        raise FileNotFoundError(f"Missing tracks_clean.geff in {movie_dir}")

    track_df = geff_to_track_df(clean_geff)
    if track_df.empty:
        raise ValueError(f"{movie_dir.name}: tracks_clean.geff has no tracks")

    raw_zarr = resolve_raw_zarr(movie_dir, raw_root)
    root = zarr.open_group(str(raw_zarr), mode="r")
    img_stack = np.asarray(root[ZARR_ARRAY_KEY])
    if img_stack.ndim != 4:
        raise ValueError(f"{raw_zarr}: expected (T,C,Y,X), got {img_stack.shape}")
    T = img_stack.shape[0]

    # Guard the frame/stack alignment explicitly.
    max_track_frame = int(track_df["frame"].max())
    if max_track_frame >= T:
        raise ValueError(
            f"{movie_dir.name}: track frame {max_track_frame} exceeds stack "
            f"length {T}. Frame convention mismatch between GEFF and zarr."
        )

    lyso_labels, mito_labels, area_log = segment_stack(img_stack)

    det = match_detections_to_masks(lyso_labels, track_df, "lysosome", T)
    det = match_detections_to_masks(mito_labels, det, "mitochondria", T)

    # Per-track aggregation.
    track_rows = []
    for (pid, uid), group in det.groupby(["particle", "track_uid"]):
        row = {
            "movie": movie_dir.name,
            "particle": int(pid),
            "track_uid": uid,
            "n_detections": int(len(group)),
            "span": int(group["frame"].max() - group["frame"].min() + 1),
        }
        row.update(summarize_track(group, "lysosome"))
        row.update(summarize_track(group, "mitochondria"))
        track_rows.append(row)
    track_summary = pd.DataFrame(track_rows)

    # Write per-movie outputs.
    out_dir = movie_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    det["movie"] = movie_dir.name
    det.to_csv(out_dir / "detections_organelle.csv", index=False)
    track_summary.to_csv(out_dir / "tracks_organelle.csv", index=False)
    pd.DataFrame(area_log).to_csv(out_dir / "mask_area_log.csv", index=False)

    return {
        "movie": movie_dir.name,
        "raw_zarr": str(raw_zarr),
        "n_tracks": int(track_df["particle"].nunique()),
        "n_detections": int(len(track_df)),
        "n_frames": int(T),
        "detections_csv": str(out_dir / "detections_organelle.csv"),
        "tracks_csv": str(out_dir / "tracks_organelle.csv"),
        "_det_df": det,
        "_track_df": track_summary,
    }


# ============================================================================
# Batch driver
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Batch lysosome/mitochondria interaction analysis for tracks_clean.geff."
    )
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("--raw-root", type=Path, default=None,
                        help="Folder of raw <movie>.zarr (fallback if metadata lacks input_zarr)")
    parser.add_argument("--movie", action="append", help="Movie id; repeat for several")
    parser.add_argument("--out-subdir", default="organelle",
                        help="Per-movie output subdir name (default: organelle)")
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
    print(f"Channels   : puncta={PUNCTA_CHANNEL} lyso={LYSOSOME_CHANNEL} "
          f"mito={MITOCHONDRIA_CHANNEL} (BFP ch3 ignored)")
    print("Distances in PIXELS.\n")

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

    # Concatenated batch outputs.
    if all_det:
        pd.concat(all_det, ignore_index=True).to_csv(
            batch_root / "organelle_interaction_detections.csv", index=False)
    if all_tracks:
        pd.concat(all_tracks, ignore_index=True).to_csv(
            batch_root / "organelle_interaction_tracks.csv", index=False)

    # Record every parameter that shaped the output.
    params = {
        "channels": {
            "puncta": PUNCTA_CHANNEL, "lysosome": LYSOSOME_CHANNEL,
            "mitochondria": MITOCHONDRIA_CHANNEL, "bfp_ignored": 3,
        },
        "zarr_array_key": ZARR_ARRAY_KEY,
        "lyso_params": LYSO_PARAMS,
        "mito_params": MITO_PARAMS,
        "near_edge_dist_px": NEAR_EDGE_DIST_PX,
        "distance_units": "pixels",
        "track_source": "tracks_clean.geff",
        "n_movies_processed": len(all_tracks),
        "n_movies_failed": len(errors),
    }
    (batch_root / "organelle_interaction_params.json").write_text(
        json.dumps(params, indent=2, default=str))

    if errors:
        pd.DataFrame(errors).to_csv(
            batch_root / "organelle_interaction_errors.csv", index=False)
        print(f"\n{len(errors)} movie(s) failed; see organelle_interaction_errors.csv")

    print(f"\nDone: {len(all_tracks)} succeeded, {len(errors)} failed, "
          f"{len(movie_dirs)} total")
    print(f"Batch tables:")
    print(f"  {batch_root / 'organelle_interaction_detections.csv'}")
    print(f"  {batch_root / 'organelle_interaction_tracks.csv'}")


if __name__ == "__main__":
    main()
