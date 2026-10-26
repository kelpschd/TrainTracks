#!/usr/bin/env python3
"""
Visualize tracks colored by path_straightness to pick a transport filter threshold.

Works with both the single-run layout (candidate_graph.py) and the batch layout
(batch_pipeline_with_posthoc.py).

Single-run usage:
    python batch_straightness_review.py \\
        /path/to/tracking_runs/runs/20260916_105301_9452 \\
        --flow /path/to/flow.zarr \\
        --gt /path/to/sg100_Well5_1018_valid_tracks.csv \\
        --scores /path/to/flow_scores.csv

Batch usage:
    python batch_straightness_review.py \\
        /path/to/tracking_batch/sg100_Well5_1018 \\
        --cutoffs 0.4 0.5 0.6 0.7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import napari
import numpy as np
import pandas as pd
import zarr
from funtracks.import_export import import_from_geff

DEFAULT_CUTOFFS = [0.4, 0.5, 0.6, 0.7]


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def geff_to_track_df(geff_path: Path) -> pd.DataFrame:
    sg = import_from_geff(
        directory=str(geff_path),
        node_name_map={"time": "t", "pos": ["y", "x"]},
    )
    attrs = sg.graph.node_attrs(unpack=True).to_pandas()
    return pd.DataFrame({
        "particle": attrs["tracklet_id"].astype(int),
        "frame":    attrs["t"].astype(int),
        "y":        attrs["pos_0"].astype(float),
        "x":        attrs["pos_1"].astype(float),
    }).sort_values(["particle", "frame"]).reset_index(drop=True)


def tracks_array(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.empty((0, 4), dtype=float)
    return df[["particle", "frame", "y", "x"]].to_numpy()


def resolve_paths(args) -> tuple[Path, Path, Path, Path]:
    """Return (geff, flow_zarr, scores_csv, gt_csv), resolving from run dir + overrides."""
    run_dir = args.run_dir.expanduser().resolve()

    # GEFF
    geff_path = run_dir / "tracks.geff"
    if not geff_path.exists():
        raise FileNotFoundError(f"tracks.geff not found in {run_dir}")

    # flow.zarr: explicit override > batch layout (run_dir/flow.zarr) > metadata
    if args.flow:
        flow_path = args.flow.expanduser().resolve()
    elif (run_dir / "flow.zarr").exists():
        flow_path = run_dir / "flow.zarr"
    else:
        meta_path = run_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                "flow.zarr not found and no metadata.json to read it from. "
                "Pass --flow explicitly."
            )
        meta = read_json(meta_path)
        flow_path = Path(meta.get("flow_path") or meta.get("output_flow_zarr", ""))
        if not flow_path.exists():
            raise FileNotFoundError(
                f"flow.zarr not found at {flow_path}. Pass --flow explicitly."
            )

    # flow_scores.csv: explicit override > run_dir
    if args.scores:
        scores_path = args.scores.expanduser().resolve()
    else:
        scores_path = run_dir / "flow_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(
            f"flow_scores.csv not found at {scores_path}. "
            "Run track_flow_score.py first, or pass --scores explicitly."
        )

    # GT CSV: explicit override > derive from metadata input_zarr name
    if args.gt:
        gt_path = args.gt.expanduser().resolve()
    else:
        meta_path = run_dir / "metadata.json"
        if meta_path.exists():
            meta = read_json(meta_path)
            input_zarr = meta.get("input_zarr", "")
            movie_id = Path(input_zarr).stem.replace(".zarr", "")
            # try sibling of the run dir's grandparent (tracking_runs layout)
            gt_path = run_dir.parent.parent / f"{movie_id}_valid_tracks.csv"
        else:
            gt_path = Path("__missing__")

    return geff_path, flow_path, scores_path, gt_path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize tracks colored by path_straightness."
    )
    parser.add_argument(
        "run_dir", type=Path,
        help="Run directory containing tracks.geff (single-run or batch movie dir)",
    )
    parser.add_argument(
        "--flow", type=Path, default=None,
        help="Path to flow.zarr (required for single-run layout if not in run_dir)",
    )
    parser.add_argument(
        "--scores", type=Path, default=None,
        help="Path to flow_scores.csv (default: <run_dir>/flow_scores.csv)",
    )
    parser.add_argument(
        "--gt", type=Path, default=None,
        help="Path to GT CSV (optional but recommended)",
    )
    parser.add_argument(
        "--image", type=Path, default=None,
        help="Path to image zarr (reads from metadata.json if not given)",
    )
    parser.add_argument(
        "--cutoffs", type=float, nargs="+", default=DEFAULT_CUTOFFS,
        help="Straightness thresholds to show as separate layers (default: 0.4 0.5 0.6 0.7)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    geff_path, flow_path, scores_path, gt_path = resolve_paths(args)

    print(f"Run dir   : {run_dir}")
    print(f"GEFF      : {geff_path}")
    print(f"Flow      : {flow_path}")
    print(f"Scores    : {scores_path}")
    print(f"GT        : {gt_path}")

    # --- Image ---
    img_path = args.image
    if img_path is None:
        meta_path = run_dir / "metadata.json"
        if meta_path.exists():
            meta = read_json(meta_path)
            img_path = Path(meta.get("input_zarr", ""))
    if img_path is None or not img_path.exists():
        raise FileNotFoundError(
            f"Image zarr not found at {img_path}. Pass --image explicitly."
        )
    img = np.asarray(zarr.open_group(str(img_path), mode="r")["s0"])
    if img.ndim != 4:
        raise ValueError(f"Expected (T,C,Y,X), got {img.shape}")
    print(f"Image     : {img.shape}")

    # --- Tracks ---
    track_df = geff_to_track_df(geff_path)
    print(f"Tracks    : {track_df['particle'].nunique()} total")

    # --- Scores ---
    feats = pd.read_csv(scores_path)
    feats["particle"] = feats["particle"].astype(int)

    required = {"particle", "eligible", "path_straightness", "net_displacement", "turn_incoherence"}
    missing = required - set(feats.columns)
    if missing:
        raise ValueError(
            f"flow_scores.csv is missing columns: {sorted(missing)}. "
            "Re-run track_flow_score.py to regenerate."
        )

    eligible_ids = set(feats.loc[feats["eligible"].astype(str) == "True", "particle"])
    eligible_df = track_df[track_df["particle"].isin(eligible_ids)].copy()

    feat_lookup = feats.set_index("particle")[
        ["path_straightness", "net_displacement", "turn_incoherence"]
    ]
    for col in ["path_straightness", "net_displacement", "turn_incoherence"]:
        eligible_df[col] = eligible_df["particle"].map(feat_lookup[col])

    n_eligible = eligible_df["particle"].nunique()
    n_ineligible = track_df["particle"].nunique() - n_eligible
    print(f"Eligible  : {n_eligible} | Ineligible/short: {n_ineligible}")

    # Cutoff preview
    print("\nCutoff preview (eligible tracks kept):")
    for c in sorted(args.cutoffs):
        kept = feats[
            (feats["eligible"].astype(str) == "True") &
            (feats["path_straightness"] >= c)
        ]["particle"].nunique()
        print(f"  path_straightness >= {c:.2f}: {kept:4d} / {n_eligible} ({kept/n_eligible*100:.1f}%)")

    # --- GT ---
    gt_tracks = None
    if gt_path.exists():
        gt_df = pd.read_csv(gt_path)
        gt_df.columns = gt_df.columns.str.strip()
        gt_tracks = gt_df[["particle", "frame", "y", "x"]].copy()
        print(f"\nGT        : {gt_tracks['particle'].nunique()} validated tracks")

        # Report straightness of GT tracks if they can be matched by particle id
        gt_pids = set(gt_tracks["particle"].astype(int))
        matched = feats[feats["particle"].isin(gt_pids)]
        if not matched.empty:
            print("GT track straightness (if particle IDs match scored tracks):")
            for _, row in matched.iterrows():
                print(f"  particle {int(row['particle'])}: straightness={row['path_straightness']:.3f} "
                      f"net_disp={row['net_displacement']:.1f} turn={row['turn_incoherence']:.3f}")
        else:
            print("(GT particle IDs don't match scored track IDs — spatial matching needed)")
    else:
        print(f"\nGT CSV not found at {gt_path} — skipping (pass --gt to include)")

    # --- Napari ---
    viewer = napari.Viewer(title=f"Straightness review: {run_dir.name}")
    viewer.add_image(img[:, 0], name="Raw image")

    # Continuous coloring by path_straightness (RdYlGn: red=wandering, green=straight)
    arr = tracks_array(eligible_df)
    props = {
        "path_straightness": eligible_df["path_straightness"].to_numpy(),
        "net_displacement":   eligible_df["net_displacement"].to_numpy(),
        "turn_incoherence":   eligible_df["turn_incoherence"].to_numpy(),
    }
    viewer.add_tracks(
        arr,
        properties=props,
        color_by="path_straightness",
        colormap="twilight",  # napari built-in: dark=low, bright=high
        name=f"ALL eligible — colored by straightness ({n_eligible} tracks)",
        visible=True,
    )

    # One threshold layer per cutoff
    for cutoff in sorted(args.cutoffs):
        kept_ids = feats.loc[
            (feats["eligible"].astype(str) == "True") &
            (feats["path_straightness"] >= cutoff),
            "particle"
        ].astype(int)
        kept_df = eligible_df[eligible_df["particle"].isin(kept_ids)]
        viewer.add_tracks(
            tracks_array(kept_df),
            name=f"straightness >= {cutoff:.2f}  ({kept_df['particle'].nunique()} tracks)",
            visible=False,
        )

    # Ineligible/short tracks for reference
    ineligible_df = track_df[~track_df["particle"].isin(eligible_ids)]
    if not ineligible_df.empty:
        viewer.add_tracks(
            tracks_array(ineligible_df),
            name=f"ineligible / short ({ineligible_df['particle'].nunique()} tracks)",
            visible=False,
        )

    # GT on top, always visible
    if gt_tracks is not None:
        viewer.add_tracks(
            gt_tracks[["particle", "frame", "y", "x"]].to_numpy(),
            name=f"GT validated ({gt_tracks['particle'].nunique()} tracks)",
            visible=True,
        )

    print("\nLayer guide:")
    print("  ALL eligible — RdYlGn: green=straight (transported), red=wandering (stationary/diffusing)")
    print("  Toggle threshold layers one at a time with GT visible")
    print("  Pick the lowest cutoff where all GT tracks survive")
    print("\nTip: color_by property can be changed in the Napari tracks layer controls")
    print("     Try switching to 'net_displacement' or 'turn_incoherence' for comparison")

    napari.run()


if __name__ == "__main__":
    main()