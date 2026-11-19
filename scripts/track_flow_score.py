"""
Score tracks by MOTION CONSISTENCY WITH FLOW -- a post-hoc, per-track quality
signal that is deliberately ORTHOGONAL to what the motile solver optimized.

Flag logic (current): TURN-ONLY, span>=5. Drift was dropped (GT tracks drift
heavily while straight). Intensity-continuity features are REPORT-ONLY, added
for the GT-discrimination experiment -- they do not touch `flagged`.

Usage:
    python track_flow_score.py                 # uses RUN_ID below
    python track_flow_score.py <run_id>
    python track_flow_score.py <run_id> --no-napari
    python track_flow_score.py --gt            # GT calibration + intensity table
    python track_flow_score.py <run_id> --write-clean
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from funtracks.import_export import import_from_geff
from TrainTracks import scoring
from TrainTracks.scoring import (
    MIN_SPAN_FOR_SCORE, MIN_SPAN_KEEP, TURN_FLAG_RADIANS,
    make_flow_sampler, score_all, score_track,
)
from TrainTracks.tracks_io import solution_tracks_to_track_df


# ============================================================================
# CONFIG
# ============================================================================

BASE = Path("/Users/kelpschdj/Documents/DataTecnica/TTU/TrainTracks")
RUNS_DIR = BASE / "tracking_runs" / "runs"
IMAGES_DIR = Path(
    "/Users/kelpschdj/Documents/DataTecnica/TTU/Data/Sphere/"
    "220725_i11w-hT-M33-I76_sg1035_d10sphere/raw_data"
)
# Flow field used for scoring (--gt uses it for the GT movie). Default: the batch
# pipeline's flow for sg100_Well5_1018 (T-1 frames). Override with --flow.
# (The old repo-root flow.zarr was archived on 2026-11-19.)
FLOW_PATH = (IMAGES_DIR.parent / "tracking_batch_TrainTracks"
             / "sg100_Well5_1018" / "flow.zarr")
IMG_NUM = 9

# Default run to score if none passed on the CLI.
RUN_ID = "20260916_105301_9452"   # -72 / pen10

# Manually validated ground-truth tracks (cols: frame, x, y, particle).
GT_CSV = BASE / "sg100_Well5_1018_valid_tracks.csv"

# Source movie for GT intensity sampling (the movie GT_CSV was validated on).
GT_MOVIE_ZARR = IMAGES_DIR / "sg100_Well5_1018.zarr"
ZARR_ARRAY_KEY = "s0"


# --- Intensity sampling (REPORT-ONLY) --------------------------------------
# Per-run outputs don't store pixel data, so intensity is sampled from the source
# .zarr. Two sources: raw detection channel, and `pre` = background-subtracted
# reconstruction matching detection (median-filtered mean projection subtracted).
# These MUST match the detection preprocessing (TrainTracks.detection.preprocess_stack)
# for `pre` to mean anything.
INTENSITY_BLOB_CHANNEL = 0
INTENSITY_MEDIAN_FILTER_SIZE = 10
INTENSITY_DISK_RADIUS_PX = 2        # mean over a fixed r=2 disk (13 px)


# ============================================================================
# Load run
# ============================================================================

def load_run(run_id: str):
    run_dir = RUNS_DIR / run_id
    with open(run_dir / "metadata.json") as f:
        meta = json.load(f)
    sg = import_from_geff(
        directory=str(run_dir / "tracks.geff"),
        node_name_map={"time": "t", "pos": ["y", "x"]},
    )
    return sg, meta


# ============================================================================
# Flow sampling
# ============================================================================



# ============================================================================
# Intensity sampling (REPORT-ONLY)
# ============================================================================

def _disk_offsets(radius: int):
    r = int(radius)
    offs = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy * dy + dx * dx <= r * r:
                offs.append((dy, dx))
    return np.array(offs, dtype=int)


def make_intensity_samplers(movie_zarr: Path):
    """Return (intensity_raw, intensity_pre), each f(t,y,x)->float.

    raw : detection channel from s0.
    pre : background-subtracted (median-filtered mean projection subtracted,
          negatives clipped) -- matches what blob_log saw at detection.
    Each averages a fixed disk around the rounded, bounds-clamped centroid.
    """
    from scipy.ndimage import median_filter

    root = zarr.open_group(str(movie_zarr), mode="r")
    img = np.asarray(root[ZARR_ARRAY_KEY])
    if img.ndim != 4:
        raise ValueError(f"{movie_zarr}: expected (T,C,Y,X), got {img.shape}")

    raw = img[:, INTENSITY_BLOB_CHANNEL].astype(np.float32)
    T, Y, X = raw.shape

    mean_proj = np.mean(raw, axis=0)
    background = median_filter(mean_proj, size=INTENSITY_MEDIAN_FILTER_SIZE)
    pre = np.clip(raw - background, 0, None).astype(np.float32)

    offs = _disk_offsets(INTENSITY_DISK_RADIUS_PX)

    def _sample(stack, t, y, x):
        tt = min(max(int(t), 0), T - 1)
        yy = int(round(y)); xx = int(round(x))
        ys = np.clip(yy + offs[:, 0], 0, Y - 1)
        xs = np.clip(xx + offs[:, 1], 0, X - 1)
        return float(np.mean(stack[tt, ys, xs]))

    def intensity_raw(t, y, x):
        return _sample(raw, t, y, x)

    def intensity_pre(t, y, x):
        return _sample(pre, t, y, x)

    return intensity_raw, intensity_pre




# ============================================================================
# Per-track features
# ============================================================================











# ============================================================================
# Clean GEFF writer (rule and writer live in TrainTracks.scoring)
# ============================================================================

def write_clean_geff(track_df: pd.DataFrame, feats: pd.DataFrame, run_id: str) -> Path:
    out = RUNS_DIR / run_id / "tracks_clean.geff"
    res = scoring.write_clean_geff(track_df, feats, out)
    print(f"\nWrote clean GEFF: {out}")
    print(f"  removed {res['n_flagged_removed']} flagged + "
          f"{res['n_short_removed']} short (span<{MIN_SPAN_KEEP}) "
          f"= {res['n_tracks_removed']} tracklets dropped")
    print(f"  kept {res['n_clean_tracks']} tracks ({res['n_clean_detections']} nodes)")
    return out



# ============================================================================
# napari
# ============================================================================

def show_napari(track_df: pd.DataFrame, feats: pd.DataFrame, run_id: str):
    import napari
    import zarr as _zarr

    images = sorted(IMAGES_DIR.glob("*.zarr"))
    np_arr = np.array(_zarr.open(images[IMG_NUM], mode="r")["s0"])
    flow_frames = np.array(_zarr.open_group(FLOW_PATH, mode="r")["flow_frames_XY"])

    df = track_df.sort_values(["particle", "frame"]).reset_index(drop=True)
    df = df.merge(feats[["suspect_score", "flagged"]],
                  left_on="particle", right_index=True, how="left")
    fill = df["suspect_score"].min()
    df["suspect_score"] = df["suspect_score"].fillna(fill)
    df["flagged"] = df["flagged"].fillna(False).astype(float)

    data = df[["particle", "frame", "y", "x"]].to_numpy()
    properties = {"suspect_score": df["suspect_score"].to_numpy(),
                  "flagged": df["flagged"].to_numpy()}

    drop_ids = set(feats.index[feats["flagged"]].astype(int).tolist())
    drop_ids |= set(feats.index[feats["span"] < MIN_SPAN_KEEP].astype(int).tolist())
    final = df[~df["particle"].astype(int).isin(drop_ids)]

    viewer = napari.Viewer()
    viewer.add_image(np_arr[:, 0], name="Raw image")
    viewer.add_image(flow_frames, name="Flow")
    viewer.add_tracks(data, properties=properties, color_by="suspect_score",
                      colormap="inferno", name="all tracks (suspect_score)",
                      visible=False)
    viewer.add_tracks(data, properties=properties, color_by="flagged",
                      colormap="inferno", name="all tracks (flagged)",
                      visible=False)
    if len(final):
        viewer.add_tracks(final[["particle", "frame", "y", "x"]].to_numpy(),
                          name=f"FINAL kept ({final['particle'].nunique()} tracks)")
    try:
        gt = pd.read_csv(GT_CSV)
        gt.columns = gt.columns.str.strip()
        viewer.add_tracks(gt[["particle", "frame", "y", "x"]].to_numpy(),
                          name=f"GT ({gt['particle'].nunique()} validated)")
    except FileNotFoundError:
        print(f"(GT CSV not found at {GT_CSV}; skipping GT layer)")
    napari.run()


# ============================================================================
# GT calibration
# ============================================================================

def load_gt_track_df() -> pd.DataFrame:
    gt = pd.read_csv(GT_CSV)
    gt.columns = gt.columns.str.strip()
    return pd.DataFrame({
        "particle": gt["particle"].astype(int),
        "frame": gt["frame"].astype(int),
        "y": gt["y"].astype(float),
        "x": gt["x"].astype(float),
    }).sort_values(["particle", "frame"]).reset_index(drop=True)


def print_gt_intensity(gt_feats: pd.DataFrame) -> None:
    cols = ["intensity_raw_cv", "intensity_raw_max_jump",
            "intensity_pre_cv", "intensity_pre_max_jump"]
    if not any(c in gt_feats.columns for c in cols):
        return
    print("\nGT intensity continuity (REPORT-ONLY -- not a flag):")
    print(f"{'pid':>6} | {'raw_cv':>7} | {'raw_jump':>8} | "
          f"{'pre_cv':>7} | {'pre_jump':>8} | {'span':>4}")
    print("-" * 52)
    for pid, r in gt_feats.sort_values("intensity_pre_max_jump",
                                       ascending=False).iterrows():
        def f(c):
            return f"{r[c]:.3f}" if c in r and pd.notna(r[c]) else "   n/a"
        print(f"{pid:>6} | {f('intensity_raw_cv'):>7} | "
              f"{f('intensity_raw_max_jump'):>8} | {f('intensity_pre_cv'):>7} | "
              f"{f('intensity_pre_max_jump'):>8} | {int(r['span']):>4}")
    for src in ("raw", "pre"):
        cv = gt_feats[f"intensity_{src}_cv"].dropna()
        mj = gt_feats[f"intensity_{src}_max_jump"].dropna()
        if len(cv):
            print(f"\nGT {src}: cv max={cv.max():.3f} median={cv.median():.3f} | "
                  f"max_jump max={mj.max():.3f} median={mj.median():.3f}")
    print("\nInterpretation: LOW cv + LOW max_jump across GT -> intensity continuity")
    print("is a clean candidate axis. HIGH max_jump in GT -> likely translation")
    print("bursts; a naive jump cutoff would flag real biology (need burst-vs-swap).")
    print("Next: compare these to the raw-track / flagged-track distributions.")


def calibrate_on_gt(flow_vec) -> None:
    gt_df = load_gt_track_df()
    n_tracks = gt_df["particle"].nunique()
    print(f"GT: {n_tracks} validated tracks, {len(gt_df)} detections")
    print(f"Cutoff under test: turn>={TURN_FLAG_RADIANS} (turn-only flag; "
          f"drift reported but not used)\n")

    # Intensity samplers from the GT movie (report-only columns).
    try:
        iraw, ipre = make_intensity_samplers(GT_MOVIE_ZARR)
    except Exception as e:
        print(f"(intensity sampling unavailable: {e})")
        iraw = ipre = None

    rows = []
    for pid, group in gt_df.groupby("particle"):
        feat = score_track(group, flow_vec, iraw, ipre)
        feat["particle"] = int(pid)
        rows.append(feat)
    gt_feats = pd.DataFrame(rows).set_index("particle")

    would_flag = (
        (gt_feats["span"] >= MIN_SPAN_FOR_SCORE)
        & (gt_feats["turn_incoherence"] >= TURN_FLAG_RADIANS)
    )
    gt_feats["would_flag"] = would_flag

    print(f"{'pid':>6} | {'span':>4} | {'turn':>6} | {'drift':>6} | "
          f"{'res_var':>8} | {'eligible':>8} | {'WOULD_FLAG':>10}")
    print("-" * 66)
    for pid, r in gt_feats.sort_values("turn_incoherence", ascending=False).iterrows():
        elig = (r["span"] >= MIN_SPAN_FOR_SCORE) and pd.notna(r["turn_incoherence"])
        drift = f"{r['cumulative_drift']:.2f}" if pd.notna(r["cumulative_drift"]) else "  n/a"
        turn = f"{r['turn_incoherence']:.2f}" if pd.notna(r["turn_incoherence"]) else "  n/a"
        rv = f"{r['residual_var']:.2f}" if pd.notna(r["residual_var"]) else "     n/a"
        print(f"{pid:>6} | {int(r['span']):>4} | {turn:>6} | {drift:>6} | "
              f"{rv:>8} | {str(elig):>8} | {str(bool(r['would_flag'])):>10}")

    n_would = int(would_flag.sum())
    print()
    if n_would == 0:
        print("PASS: no GT track would be flagged. Turn cutoff is safe against "
              "known-good tracks.")
    else:
        print(f"WARNING: {n_would} GT track(s) would be flagged -- turn cutoff "
              "too tight. Raise TURN_FLAG_RADIANS above the offending value.")

    elig_gt = gt_feats[(gt_feats["span"] >= MIN_SPAN_FOR_SCORE)
                       & gt_feats["turn_incoherence"].notna()]
    if len(elig_gt):
        print(f"\nWorst GT (eligible) turn -- keep cutoff ABOVE this:")
        print(f"  max GT turn : {elig_gt['turn_incoherence'].max():.2f} rad "
              f"(current cutoff {TURN_FLAG_RADIANS})")
        print(f"  (drift, unused: GT ranges {elig_gt['cumulative_drift'].min():.2f}"
              f"-{elig_gt['cumulative_drift'].max():.2f} px/frame)")

    print_gt_intensity(gt_feats)


# ============================================================================
# Main
# ============================================================================

def main():
    global FLOW_PATH
    parser = argparse.ArgumentParser(
        description="Score tracks by motion consistency with the flow field."
    )
    parser.add_argument("run_id", nargs="?", default=RUN_ID)
    parser.add_argument("--no-napari", action="store_true")
    parser.add_argument("--gt", action="store_true",
                        help="GT calibration + intensity table")
    parser.add_argument("--write-clean", action="store_true")
    parser.add_argument("--flow", type=Path, default=None,
                        help=f"flow.zarr to use (default: {FLOW_PATH})")
    args = parser.parse_args()

    if args.flow is not None:
        FLOW_PATH = args.flow.expanduser().resolve()

    flow_raw = np.array(zarr.open_group(FLOW_PATH, mode="r")["flow_raw"])
    flow_vec = make_flow_sampler(flow_raw)

    if args.gt:
        print(f"flow_raw shape: {flow_raw.shape}")
        calibrate_on_gt(flow_vec)
        return

    sg, meta = load_run(args.run_id)
    track_df = solution_tracks_to_track_df(sg)
    print(f"Run {args.run_id}: {track_df['particle'].nunique()} tracks, "
          f"{len(track_df)} detections | "
          f"max_gap={meta.get('max_gap')} gap_penalty={meta.get('gap_penalty')} "
          f"edge_const={meta.get('edge_selected_constant')}")
    print(f"flow_raw shape: {flow_raw.shape}")

    # Intensity samplers for the run. Resolve the source movie from metadata;
    # standalone runs may store the blob CSV path (input_csv) rather than the
    # movie, so fall back to None if we can't find a .zarr.
    movie = None
    for key in ("input_zarr", "input_movie"):
        v = meta.get(key)
        if v and str(v).endswith(".zarr") and Path(v).exists():
            movie = Path(v)
            break
        if movie is None:
            movie = GT_MOVIE_ZARR   # this run == Well5_1018; remove for other movies
    iraw = ipre = None
    if movie is not None:
        try:
            iraw, ipre = make_intensity_samplers(movie)
        except Exception as e:
            print(f"(intensity sampling unavailable: {e})")
    else:
        print("(no source .zarr in run metadata; intensity columns will be NaN. "
              "Set the movie path in metadata or use --gt for the calibration.)")

    feats = score_all(track_df, flow_vec, iraw, ipre)

    out_csv = RUNS_DIR / args.run_id / "flow_scores.csv"
    feats.to_csv(out_csv)
    print(f"Wrote {out_csv}")

    eligible = feats[feats["eligible"]]
    too_short = feats[~feats["eligible"]]
    flagged = feats[feats["flagged"]]
    print(f"\nPopulations:")
    print(f"  eligible (span>={MIN_SPAN_FOR_SCORE}, scorable): {len(eligible)}")
    print(f"  too short / unscorable (passed through):    {len(too_short)}")
    print(f"  FLAGGED (turn>={TURN_FLAG_RADIANS}):                    {len(flagged)}")

    if len(flagged):
        print(f"\nFLAGGED tracks (tag-only; nothing deleted):")
        print(f"{'pid':>6} | {'drift':>6} | {'turn':>6} | {'res_var':>8} | "
              f"{'span':>4} | {'skip%':>5} | {'suspect_z':>9}")
        print("-" * 62)
        for pid, r in flagged.sort_values("cumulative_drift", ascending=False).iterrows():
            print(f"{pid:>6} | {r['cumulative_drift']:>6.2f} | "
                  f"{r['turn_incoherence']:>6.2f} | {r['residual_var']:>8.2f} | "
                  f"{int(r['span']):>4} | {r['frac_skip']*100:>4.0f}% | "
                  f"{r['suspect_score']:>9.2f}")

    review = eligible[~eligible["flagged"]].sort_values("suspect_score", ascending=False)
    print(f"\nTop eligible NON-flagged by suspect_score:")
    print(f"{'pid':>6} | {'suspect':>7} | {'drift':>6} | {'turn':>6} | {'span':>4}")
    print("-" * 44)
    for pid, r in review.head(15).iterrows():
        print(f"{pid:>6} | {r['suspect_score']:>7.2f} | "
              f"{r['cumulative_drift']:>6.2f} | {r['turn_incoherence']:>6.2f} | "
              f"{int(r['span']):>4}")

    if args.write_clean:
        write_clean_geff(track_df, feats, args.run_id)
    if not args.no_napari:
        show_napari(track_df, feats, args.run_id)


if __name__ == "__main__":
    main()