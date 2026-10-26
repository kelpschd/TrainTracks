#!/usr/bin/env python3
"""
Interactive per-movie diagnostic viewer for the batch tracking pipeline.

For one movie, load and compare:
    - raw image
    - flow magnitude (derived from flow.zarr/flow_raw)
    - detected blobs
    - GT tracks
    - raw Motile tracks
    - short tracks removed by span < MIN_SPAN_KEEP
    - tracks removed by the turn filter (turn >= TURN_FLAG_RADIANS)
    - final clean tracks

The script uses the existing batch output + metadata, so no tracking is rerun.

Usage:
    python batch_track_review.py /path/to/tracking_batch --movie sg100_Well5_1018
    python batch_track_review.py /path/to/tracking_batch --movie sg100_Well5_1018 --no-flow

The GT root defaults to the tracking_results directory used by the batch
traccuracy script.
"""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import napari
import numpy as np
import pandas as pd
import zarr
from funtracks.import_export import import_from_geff

DEFAULT_GT_ROOT = Path(
    "/Users/kelpschdj/Documents/DataTecnica/TTU/Data/Sphere/"
    "220725_i11w-hT-M33-I76_sg1035_d10sphere/tracking_results"
)


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def movie_id_from_metadata(movie_dir: Path, meta: dict) -> str:
    input_zarr = meta.get("input_zarr")
    if input_zarr:
        name = Path(input_zarr).name
        return name[:-5] if name.endswith(".zarr") else Path(name).stem
    return movie_dir.name




def canonical_track_id(group: pd.DataFrame) -> int:
    """Return a deterministic ID for a track from its frame/position sequence.

    The ID is independent of funtracks' imported ``tracklet_id`` ordering, so
    the same unchanged trajectory gets the same particle ID in tracks.geff,
    flow_scores.csv, and tracks_clean.geff.
    """
    import hashlib

    g = group.sort_values("frame")[["frame", "y", "x"]].copy()
    # Stable byte representation; 1e6 precision is far below localization
    # precision here while avoiding incidental float serialization differences.
    payload = [
        (int(r.frame), int(round(float(r.y) * 1_000_000)), int(round(float(r.x) * 1_000_000)))
        for r in g.itertuples(index=False)
    ]
    digest = hashlib.sha1(repr(payload).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def assign_canonical_track_ids(track_df: pd.DataFrame) -> pd.DataFrame:
    """Replace particle IDs with deterministic geometry-based IDs."""
    df = track_df.copy().sort_values(["particle", "frame"]).reset_index(drop=True)
    id_map = {
        int(old): canonical_track_id(group)
        for old, group in df.groupby("particle", sort=False)
    }
    df["particle"] = df["particle"].astype(int).map(id_map).astype("int64")
    return df.sort_values(["particle", "frame"]).reset_index(drop=True)


def track_uid_for_group(group: pd.DataFrame) -> str:
    """Same deterministic geometry UID used by track_flow_score.py."""
    g = group.sort_values("frame")[["frame", "y", "x"]]
    payload = [
        (
            int(r.frame),
            int(round(float(r.y) * 1_000_000)),
            int(round(float(r.x) * 1_000_000)),
        )
        for r in g.itertuples(index=False)
    ]
    return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()


def add_track_uids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    uid_map = {
        int(pid): track_uid_for_group(group)
        for pid, group in out.groupby("particle", sort=False)
    }
    out["track_uid"] = out["particle"].astype(int).map(uid_map)
    return out


def geff_to_track_df(geff_path: Path) -> pd.DataFrame:
    sg = import_from_geff(
        directory=str(geff_path),
        node_name_map={"time": "t", "pos": ["y", "x"]},
    )
    attrs = sg.graph.node_attrs(unpack=True).to_pandas()
    required = {"tracklet_id", "t", "pos_0", "pos_1"}
    missing = required - set(attrs.columns)
    if missing:
        raise ValueError(
            f"{geff_path}: missing imported attributes {sorted(missing)}"
        )
    df = pd.DataFrame(
        {
            "particle": attrs["tracklet_id"].astype(int),
            "frame": attrs["t"].astype(int),
            "y": attrs["pos_0"].astype(float),
            "x": attrs["pos_1"].astype(float),
        }
    )
    return add_track_uids(df).sort_values(["particle", "frame"]).reset_index(drop=True)


def track_spans(df: pd.DataFrame) -> pd.Series:
    g = df.groupby("particle")["frame"]
    return g.max() - g.min() + 1


def tracks_array(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.empty((0, 4), dtype=float)
    return df[["particle", "frame", "y", "x"]].to_numpy()


def build_flow_magnitude(flow_raw: np.ndarray) -> np.ndarray:
    """Pad T-1 flow magnitudes to T image frames with a zero last frame."""
    mag = np.linalg.norm(flow_raw, axis=-1).astype(np.float32)
    return np.concatenate(
        [mag, np.zeros((1, mag.shape[1], mag.shape[2]), dtype=np.float32)],
        axis=0,
    )


def add_track_layer(viewer, df, name, visible=True, color_by=None, colormap="viridis"):
    if not df.empty:
        clean = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["particle", "frame", "y", "x"]).copy()
        clean["particle"] = pd.to_numeric(clean["particle"], errors="coerce").astype("int64")
        clean["frame"] = pd.to_numeric(clean["frame"], errors="coerce").astype("int64")
        df = clean
    layer = viewer.add_tracks(
        tracks_array(df),
        name=name,
        visible=visible,
    )
    if color_by is not None and not df.empty:
        if color_by not in df.columns:
            raise ValueError(
                f"Cannot color layer {name!r} by {color_by!r}: "
                f"column is missing. Available columns: {list(df.columns)}"
            )
        # Broadcast track-level property to every vertex.
        props = df[["particle", color_by]].copy()
        props = props.drop_duplicates("particle").set_index("particle")
        values = df["particle"].map(props[color_by]).to_numpy()
        layer.properties = {color_by: values}
        layer.color_by = color_by
        layer.colormap = colormap
    return layer


def main():
    parser = argparse.ArgumentParser(
        description="Open a diagnostic Napari view for one batch-tracked movie."
    )
    parser.add_argument(
        "batch_root", type=Path,
        help="Directory produced by batch_pipeline_with_posthoc.py",
    )
    parser.add_argument(
        "--movie", required=True,
        help="Movie id, e.g. sg100_Well5_1018",
    )
    parser.add_argument(
        "--gt-root", type=Path, default=DEFAULT_GT_ROOT,
        help=f"Directory containing <movie>_valid_tracks.csv (default: {DEFAULT_GT_ROOT})",
    )
    parser.add_argument(
        "--no-flow", action="store_true",
        help="Do not load/show flow magnitude",
    )
    parser.add_argument(
        "--no-blobs", action="store_true",
        help="Do not show detected blobs",
    )
    parser.add_argument(
        "--show-all-raw", action="store_true",
        help="Show all raw Motile tracks initially (otherwise hidden)",
    )
    parser.add_argument(
        "--show-removed", action="store_true",
        help="Show both removed categories initially (otherwise hidden)",
    )
    args = parser.parse_args()

    batch_root = args.batch_root.expanduser().resolve()
    gt_root = args.gt_root.expanduser().resolve()
    movie_dir = batch_root / args.movie

    if not movie_dir.is_dir():
        raise FileNotFoundError(f"Movie directory not found: {movie_dir}")

    metadata_path = movie_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json: {metadata_path}")
    meta = read_json(metadata_path)
    movie_id = movie_id_from_metadata(movie_dir, meta)

    zarr_path = Path(meta["input_zarr"])
    if not zarr_path.exists():
        raise FileNotFoundError(f"Input Zarr not found: {zarr_path}")

    raw_geff = movie_dir / "tracks.geff"
    clean_geff = movie_dir / "tracks_clean.geff"
    scores_csv = movie_dir / "flow_scores.csv"
    blobs_csv = movie_dir / "blobs.csv"
    flow_path = movie_dir / "flow.zarr"
    gt_csv = gt_root / f"{movie_id}_valid_tracks.csv"

    for path in [raw_geff, clean_geff, scores_csv, blobs_csv]:
        if not path.exists():
            raise FileNotFoundError(f"Required batch output missing: {path}")
    if not gt_csv.exists():
        raise FileNotFoundError(f"GT not found: {gt_csv}")
    if not args.no_flow and not flow_path.exists():
        raise FileNotFoundError(f"Flow output missing: {flow_path}")

    print(f"Movie: {movie_id}")
    print(f"Zarr : {zarr_path}")
    print(f"GT   : {gt_csv}")
    print()

    # ------------------------------------------------------------------
    # Raw image
    # ------------------------------------------------------------------
    zroot = zarr.open_group(str(zarr_path), mode="r")
    img = np.asarray(zroot["s0"])
    if img.ndim != 4:
        raise ValueError(f"Expected s0 shape (T,C,Y,X), got {img.shape}")
    T, C, Y, X = img.shape
    print(f"Image shape: {img.shape}")

    # ------------------------------------------------------------------
    # Tracks + post-hoc classification
    # ------------------------------------------------------------------
    raw_df = geff_to_track_df(raw_geff)
    clean_df = geff_to_track_df(clean_geff)
    feats = pd.read_csv(scores_csv)
    required_score_cols = {"particle", "track_uid", "flagged"}
    missing_score_cols = required_score_cols - set(feats.columns)
    if missing_score_cols:
        raise ValueError(
            f"{scores_csv}: missing required score columns {sorted(missing_score_cols)}; "
            "rerun batch_rescore_tracks.py with the current track_flow_score.py"
        )
    feats["particle"] = pd.to_numeric(feats["particle"], errors="raise").astype(np.int64)
    feats["track_uid"] = feats["track_uid"].astype(str)
    feats = feats.set_index("track_uid")

    score_uid_set = set(feats.index)
    raw_uid_set = set(raw_df["track_uid"].dropna())
    missing_uids = raw_uid_set - score_uid_set
    extra_uids = score_uid_set - raw_uid_set
    if missing_uids or extra_uids:
        raise ValueError(
            f"Track-score identity mismatch in {movie_id}: "
            f"{len(missing_uids)} raw tracks have no score row; "
            f"{len(extra_uids)} score rows have no raw track. "
            "This usually means the score file was generated from different track geometry."
        )

    spans = track_spans(raw_df).rename_axis("particle")
    min_span_keep = int(meta.get("posthoc_params", {}).get("min_span_keep", 5))
    turn_cutoff = float(meta.get("posthoc_params", {}).get("turn_flag_radians", 1.4))

    uid_by_particle = (
        raw_df[["particle", "track_uid"]]
        .drop_duplicates("particle")
        .set_index("particle")["track_uid"]
    )
    short_ids = set(spans[spans < min_span_keep].index.astype(int))
    flagged_uids = set(feats.index[feats["flagged"].astype(bool)])

    # Defensive consistency checks against metadata.
    print(f"Raw tracks  : {raw_df['particle'].nunique()}")
    print(f"Clean tracks: {clean_df['particle'].nunique()}")
    print(f"Short (<{min_span_keep}) : {len(short_ids)}")
    print(f"Turn >= {turn_cutoff:.2f} rad: {len(flagged_uids)}")

    short_df = raw_df[raw_df["particle"].isin(short_ids)]
    flagged_df = raw_df[raw_df["track_uid"].isin(flagged_uids)].copy()

    # Only eligible/non-short tracks can be in flagged_uids in the existing pipeline,
    # but keep the categories explicit for diagnostics. Attach the track-level
    # turn score to every vertex so Napari can color the track by that property.
    flagged_only_df = flagged_df[~flagged_df["particle"].isin(short_ids)].copy()
    if not flagged_only_df.empty:
        turn_props = (
            feats[["turn_incoherence"]]
            .reset_index()
            .rename(columns={"index": "track_uid"})
        )
        flagged_only_df = flagged_only_df.merge(
            turn_props,
            on="track_uid",
            how="left",
            validate="many_to_one",
        )
        missing_turn = flagged_only_df["turn_incoherence"].isna().sum()
        if missing_turn:
            raise ValueError(
                f"{missing_turn} flagged track vertices are missing turn_incoherence "
                f"after track_uid join in {movie_id}."
            )

    gt_df = pd.read_csv(gt_csv)
    gt_df.columns = gt_df.columns.str.strip()
    gt_tracks = gt_df[["particle", "frame", "y", "x"]].copy()

    # ------------------------------------------------------------------
    # Napari
    # ------------------------------------------------------------------
    viewer = napari.Viewer(title=f"Track review: {movie_id}")
    viewer.add_image(img[:, 0], name="Raw image")

    if not args.no_flow:
        froot = zarr.open_group(str(flow_path), mode="r")
        flow_raw = np.asarray(froot["flow_raw"])
        if flow_raw.shape[1:3] != (Y, X) or flow_raw.shape[-1] != 2:
            raise ValueError(
                f"Unexpected flow shape {flow_raw.shape} for image {(T,Y,X)}"
            )
        viewer.add_image(
            build_flow_magnitude(flow_raw),
            name="Flow magnitude",
            visible=False,
            colormap="turbo",
            blending="additive",
            opacity=0.45,
        )

    if not args.no_blobs:
        blobs = pd.read_csv(blobs_csv)
        if not blobs.empty:
            # napari Points expects [t, y, x]. Batch blobs.csv is exactly t,y,x.
            viewer.add_points(
                blobs[["t", "y", "x"]].to_numpy(),
                name="Detected blobs",
                size=7,
                face_color="transparent",
                border_color="white",
                border_width=0.2,
                opacity=0.35,
                visible=False,
            )

    add_track_layer(
        viewer,
        raw_df,
        "RAW Motile tracks",
        visible=args.show_all_raw,
    )

    add_track_layer(
        viewer,
        short_df,
        f"REMOVED: short < {min_span_keep}",
        visible=args.show_removed,
    )

    flagged_layer = add_track_layer(
        viewer,
        flagged_only_df,
        f"REMOVED: turn >= {turn_cutoff:.2f} rad",
        visible=args.show_removed,
        color_by="turn_incoherence" if not flagged_only_df.empty else None,
        colormap="turbo",
    )

    add_track_layer(
        viewer,
        clean_df,
        "FINAL clean tracks",
        visible=True,
    )

    add_track_layer(
        viewer,
        gt_tracks,
        "VALIDATED GT",
        visible=True,
    )

    # Add a compact textual summary to the console so the viewer always has context.
    print()
    print("Layer interpretation:")
    print("  RAW Motile tracks              = solver output before post-hoc cuts")
    print(f"  REMOVED: short < {min_span_keep}       = span-based cut")
    print(f"  REMOVED: turn >= {turn_cutoff:.2f} rad      = flow-consistency turn cut")
    print("  FINAL clean tracks             = tracks_clean.geff")
    print("  VALIDATED GT                   = manually validated trajectories")
    print()
    print("Tip: turn on one removed layer at a time, with GT visible, to inspect false removals.")

    # Keep a useful property name visible even if there are no flagged tracks.
    if flagged_layer is not None and not flagged_df.empty:
        flagged_layer.metadata["cutoff_radians"] = turn_cutoff

    napari.run()


if __name__ == "__main__":
    main()
