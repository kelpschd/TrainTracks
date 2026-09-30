"""
Score tracks by MOTION CONSISTENCY WITH FLOW -- a post-hoc, per-track quality
signal that is deliberately ORTHOGONAL to what the motile solver optimized.

Why this isn't circular with the solver's flow_offset
-----------------------------------------------------
The solver minimizes a GLOBAL sum of per-edge terms (EdgeSelectedCost with the
flow residual + the -36/-72 edge_selected_constant bounty), traded off against
NodeAppearCost etc. So a high-residual edge can still be selected because it
beat the alternatives, not because it's good. And the solver never evaluates any
SECOND-ORDER, per-track property. This scorer measures three things the cost
function never saw:

  1. residual_var  -- variance of per-edge flow residual ALONG a track. One bad
                      link among good ones (spiky) vs. uniformly mediocre. The
                      solver optimized the SUM, not the spread.
  2. turn_incoherence -- how much the track's per-frame velocity direction
                      changes step to step. Real motion here follows the flow and
                      is directional; a mislink zigzags. Never in the objective.
  3. cumulative_drift -- how far the track wanders from the path you'd get by
                      integrating the flow field from its start. Per-edge one-step
                      residuals can all be acceptable while the track slowly walks
                      off its true cell over its full span.

We intentionally DE-EMPHASIZE mean one-step residual: it's the axis the solver
already squeezed, so it's the weakest discriminator. It's reported for
reference but kept out of the composite score.

Two outputs
-----------
1. suspect_score : a z-scored composite (over the ELIGIBLE population only),
   for RANKED REVIEW. Relative, uncalibrated -- use it to sort, not to cull.
2. flagged       : an AUTONOMOUS cull decision. A track is flagged only if it is
   eligible (span >= MIN_SPAN_FOR_SCORE) AND fails BOTH absolute cutoffs
   (drift AND turn). Absolute units, so a run's junk mix can't change the verdict
   on a given good track. Tag-only: nothing is deleted; the flag is written to
   the CSV and you filter on it downstream.

Conventions carried over from the rest of the pipeline
------------------------------------------------------
- flow.zarr / flow_raw is (T,Y,X,2). optical_flow.py writes output_zarr[i-1] =
  (motion frame i-1 -> i), so flow_raw[t] is the t -> t+1 displacement field.
- candidate_graph.py reads flow as (dy, dx) = (mean[...,1], mean[...,0]). We
  match that: flow_vec(t, y, x) returns (dy, dx).
- Skip edges (gap>1): every step is normalized to PER-FRAME velocity
  (displacement / gap) before angles/residuals, so a 2-frame chord is
  comparable to a 1-frame step. Drift integrates flow frame-by-frame across the
  gap (constant-velocity-per-frame, same assumption as add_flow_dist_attr).

Usage:
    python track_flow_score.py                 # uses RUN_ID below
    python track_flow_score.py <run_id>
    python track_flow_score.py <run_id> --no-napari
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
import geff
from funtracks.import_export import import_from_geff


# ============================================================================
# CONFIG
# ============================================================================

BASE = Path("/Users/kelpschdj/Documents/DataTecnica/TTU/TrainTracks")
RUNS_DIR = BASE / "tracking_runs" / "runs"
FLOW_PATH = BASE / "flow.zarr"

IMAGES_DIR = Path(
    "/Users/kelpschdj/Documents/DataTecnica/TTU/Data/Sphere/"
    "220725_i11w-hT-M33-I76_sg1035_d10sphere/raw_data"
)
IMG_NUM = 9

# Default run to score if none passed on the CLI.
RUN_ID = "20260916_105301_9452"   # -72 / pen10

# Manually validated ground-truth tracks (cols: frame, x, y, particle). Used by
# --gt to calibrate the flag cutoffs: score these through the SAME path and
# confirm they all land under the cutoffs (flagged=False).
GT_CSV = BASE / "sg100_Well5_1018_valid_tracks.csv"

# Composite weights. Each orthogonal feature is z-scored across the ELIGIBLE
# population before combining, so weights are relative importance, not absolute
# scale. Mean one-step residual is NOT in the composite on purpose.
COMPOSITE_WEIGHTS = dict(
    residual_var=1.0,
    turn_incoherence=1.0,
    cumulative_drift=1.0,
)

# Evidence floor. Tracks with span < this are NOT scored, NOT flagged, and are
# excluded from the z-score population entirely (passed through eligible=False).
# A span-3 track's residual_var is the variance of two numbers and its
# turn_incoherence is a single angle -- pure noise, and including them corrupts
# the population statistics that real candidates get judged against. You filter
# these short tracks out downstream anyway.
MIN_SPAN_FOR_SCORE = 5

# --- Flag cutoff (autonomous cull) ------------------------------------------
# TURN-ONLY flag, calibrated against the GT. The 11 validated tracks all turn
# <= 0.91 rad/step (strongly directional, as cells following the flow should be),
# so a cutoff of 1.4 leaves clear margin above known-good while staying below the
# ~1.57 rad (90deg/step) random-walk line. A span>=5 track that turns more than
# this, on average, wanders in a way no validated track does -- the mislink
# signature.
#
# Drift was DROPPED as a flag axis: the GT showed real tracks drift up to
# 9.25 px/frame from the flow-integrated path while moving dead straight (the
# flow field is smoothed, so consistent real motion accumulates "drift" that is
# actually correct). Any drift cutoff low enough to catch bad tracks would eat
# validated ones. cumulative_drift is still COMPUTED and written to the CSV as a
# reported feature -- just no longer part of the flag.
#
# CALIBRATION RECIPE (do once): run --gt and confirm all ~11 GT tracks land
# under TURN_FLAG_RADIANS (flagged=False). Proves the cutoff won't eat real
# biology; can't prove it catches every mislink (11 tracks is too few). Tag-only.
TURN_FLAG_RADIANS = 1.4         # mean step-to-step turn; GT max is 0.91,
                                # random walk ~1.57. Flag if a track exceeds this.

# Retained ONLY for the reported-feature readout and the --gt drift column; NOT
# used in the flag. Kept so the drift context is visible next to each track.
DRIFT_REPORT_PX_PER_FRAME = 8.0

# Minimum track span (frames) to KEEP in the clean GEFF. Independent of the
# flag: short tracks aren't "erratic," they're just not of interest, so they're
# dropped by span, not by turn. tracks_clean.geff removes flagged tracks AND
# span < this. Matches the span filter you were applying downstream.
MIN_SPAN_KEEP = 5

# Tracks needing at least this many EDGES for variance/turn stats to be defined.
MIN_EDGES_FOR_STATS = 2


# ============================================================================
# Load run (same shape as track_vis.load_run)
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


def geff_to_track_df(sg) -> pd.DataFrame:
    """funtracks solution graph -> [particle, frame, y, x], frame-sorted."""
    attrs = sg.graph.node_attrs(unpack=True).to_pandas()
    return pd.DataFrame({
        "particle": attrs["tracklet_id"],
        "frame": attrs["t"].astype(int),
        "y": attrs["pos_0"].astype(float),
        "x": attrs["pos_1"].astype(float),
    }).sort_values(["particle", "frame"]).reset_index(drop=True)


# ============================================================================
# Flow sampling
# ============================================================================

def make_flow_sampler(flow_raw: np.ndarray):
    """Return flow_vec(t, y, x) -> (dy, dx), matching candidate_graph's axis order.

    flow_raw is (T, Y, X, 2) with last axis (u, v) = (x-comp, y-comp) as OpenCV
    returns it. candidate_graph reads dy from [...,1] and dx from [...,0], so we
    do the same. Positions are rounded to int and clamped into bounds; frames
    past the last flow slice reuse the last slice (flow_raw has one slice per
    transition, so the final node's frame may index one past the end).
    """
    T, Y, X, _ = flow_raw.shape

    def flow_vec(t: int, y: float, x: float):
        tt = min(max(int(t), 0), T - 1)
        yy = min(max(int(round(y)), 0), Y - 1)
        xx = min(max(int(round(x)), 0), X - 1)
        v = flow_raw[tt, yy, xx]
        return float(v[1]), float(v[0])  # (dy, dx)

    return flow_vec


# ============================================================================
# Per-track features
# ============================================================================

def _unit(v: np.ndarray):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else None


def score_track(group: pd.DataFrame, flow_vec) -> dict:
    """Compute motion-consistency features for one particle's detections.

    group: rows for a single particle, columns [frame, y, x], frame-sorted.
    Returns a dict of features (NaN where undefined for short tracks).
    """
    g = group.sort_values("frame").reset_index(drop=True)
    n = len(g)
    frames = g["frame"].to_numpy()
    ys = g["y"].to_numpy()
    xs = g["x"].to_numpy()
    span = int(frames[-1] - frames[0] + 1)
    n_edges = n - 1

    base = dict(
        n_nodes=n, n_edges=n_edges, span=span,
        mean_residual=np.nan, residual_var=np.nan,
        turn_incoherence=np.nan, cumulative_drift=np.nan,
        max_step_gap=0, frac_skip=np.nan,
    )
    if n_edges < 1:
        return base

    # Per-edge quantities. Everything is normalized to PER-FRAME velocity so
    # skip edges are comparable to adjacent ones.
    residuals = []          # |observed per-frame vel - flow-predicted per-frame vel|
    velocities = []         # per-frame velocity vectors (dy, dx)
    gaps = []

    for i in range(n_edges):
        t0 = int(frames[i])
        gap = int(frames[i + 1] - frames[i])
        gap = max(gap, 1)
        gaps.append(gap)

        disp = np.array([ys[i + 1] - ys[i], xs[i + 1] - xs[i]])  # (dy, dx)
        vel = disp / gap                                          # per-frame velocity
        velocities.append(vel)

        # Flow-predicted per-frame velocity: average the flow the cell would
        # experience across the gap, sampled along the straight interpolated
        # path (constant-velocity-per-frame). For gap=1 this is just flow at the
        # source node -- exactly the solver's one-step model.
        pred_steps = []
        pos = np.array([ys[i], xs[i]])
        for k in range(gap):
            fdy, fdx = flow_vec(t0 + k, pos[0], pos[1])
            f = np.array([fdy, fdx])
            pred_steps.append(f)
            pos = pos + f  # advance along the flow's own prediction
        pred_vel = np.mean(pred_steps, axis=0)

        residuals.append(float(np.linalg.norm(vel - pred_vel)))

    residuals = np.array(residuals)
    gaps = np.array(gaps)

    # 1. Residual spread along the track (orthogonal to the solver's summed cost)
    mean_residual = float(np.mean(residuals))
    residual_var = float(np.var(residuals)) if n_edges >= MIN_EDGES_FOR_STATS else np.nan

    # 2. Turning incoherence: mean angle (radians) between consecutive per-frame
    #    velocity directions. Zero = perfectly straight; pi = full reversal.
    if n_edges >= MIN_EDGES_FOR_STATS:
        angles = []
        for i in range(n_edges - 1):
            a = _unit(velocities[i])
            b = _unit(velocities[i + 1])
            if a is None or b is None:
                continue  # stationary step has no direction; skip it
            cos = np.clip(np.dot(a, b), -1.0, 1.0)
            angles.append(float(np.arccos(cos)))
        turn_incoherence = float(np.mean(angles)) if angles else np.nan
    else:
        turn_incoherence = np.nan

    # 3. Cumulative drift: integrate the flow field from the track's START and
    #    compare the endpoint of that flow-predicted path to the track's actual
    #    endpoint, normalized by span so long and short tracks are comparable.
    pos = np.array([ys[0], xs[0]])
    t = int(frames[0])
    for i in range(n_edges):
        gap = int(gaps[i])
        for k in range(gap):
            fdy, fdx = flow_vec(t + k, pos[0], pos[1])
            pos = pos + np.array([fdy, fdx])
        t += gap
    actual_end = np.array([ys[-1], xs[-1]])
    total_drift = float(np.linalg.norm(pos - actual_end))
    cumulative_drift = total_drift / max(span - 1, 1)  # per-frame drift rate

    return dict(
        n_nodes=n, n_edges=n_edges, span=span,
        mean_residual=mean_residual,
        residual_var=residual_var,
        turn_incoherence=turn_incoherence,
        cumulative_drift=cumulative_drift,
        max_step_gap=int(gaps.max()),
        frac_skip=float(np.mean(gaps > 1)),
    )


def _zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std(ddof=0)
    if sd < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def score_all(track_df: pd.DataFrame, flow_vec) -> pd.DataFrame:
    rows = []
    for pid, group in track_df.groupby("particle"):
        feat = score_track(group, flow_vec)
        feat["particle"] = int(pid)
        rows.append(feat)
    feats = pd.DataFrame(rows).set_index("particle")

    # Evidence floor: only tracks with enough span AND computable stats enter the
    # scored population. Everything else is passed through with eligible=False --
    # not scored, not flagged, and (critically) NOT part of the z-score
    # normalization, so short-track noise no longer sets the mean/std that real
    # candidates are judged against.
    eligible = (
        (feats["span"] >= MIN_SPAN_FOR_SCORE)
        & feats["residual_var"].notna()
        & feats["turn_incoherence"].notna()
    )
    feats["eligible"] = eligible

    # Composite z-score (for ranked REVIEW, not the flag): standardized over the
    # eligible population only. Higher = more suspect relative to peers.
    comp = pd.Series(np.nan, index=feats.index)
    if eligible.any():
        z = pd.DataFrame(index=feats.index[eligible])
        for col, w in COMPOSITE_WEIGHTS.items():
            z[col] = _zscore(feats.loc[eligible, col]) * w
        comp.loc[eligible] = z.sum(axis=1)
    feats["suspect_score"] = comp

    # Autonomous flag (for the cull): TURN-ONLY, absolute cutoff. Drift was
    # dropped because the GT showed real tracks drift heavily while straight.
    # Only eligible tracks can be flagged; ineligible ones are flagged=False.
    feats["flagged"] = eligible & (feats["turn_incoherence"] >= TURN_FLAG_RADIANS)

    return feats.sort_values("suspect_score", ascending=False)


# ============================================================================
# Clean GEFF writer (the actual cull)
# ============================================================================

def write_clean_geff(track_df: pd.DataFrame, feats: pd.DataFrame,
                     run_id: str) -> Path:
    """Write a culled GEFF next to the run: flagged tracklets AND short tracks
    both removed.

    Two independent cuts, both applied here so the output needs no further
    filtering:
      1. flagged (turn-based) -- erratic tracks the cull identified.
      2. span < MIN_SPAN_KEEP  -- short tracks (not erratic, just not of
         interest); this is the span filter that used to live downstream.

    Built directly from the kept rows of track_df (the same [particle, frame,
    y, x] view every other script reads via geff_to_track_df), NOT by mutating
    the tracksdata GraphView -- GraphView.copy() isn't supported. Constructing a
    fresh nx graph from the DataFrame matches traccuracy_compare.track_df_to_nx,
    so it stays consistent with the pipeline. Original tracks.geff is untouched.
    """
    import networkx as nx

    flagged_ids = set(feats.index[feats["flagged"]].astype(int).tolist())
    short_ids = set(feats.index[feats["span"] < MIN_SPAN_KEEP].astype(int).tolist())
    drop_ids = flagged_ids | short_ids

    keep = track_df[~track_df["particle"].astype(int).isin(drop_ids)].copy()
    keep = keep.sort_values(["particle", "frame"]).reset_index(drop=True)

    # Fresh DiGraph: nodes carry t/y/x + tracklet_id; edges connect each kept
    # particle's detections in frame order (spanning gaps as skip edges).
    g = nx.DiGraph()
    next_id = 0
    for pid, group in keep.groupby("particle"):
        group = group.sort_values("frame")
        prev = None
        for _, row in group.iterrows():
            nid = next_id
            next_id += 1
            g.add_node(
                nid,
                t=int(row["frame"]),
                y=float(row["y"]),
                x=float(row["x"]),
                tracklet_id=int(pid),
            )
            if prev is not None:
                g.add_edge(prev, nid)
            prev = nid

    out = RUNS_DIR / run_id / "tracks_clean.geff"
    geff.write(g, str(out), zarr_format=3, overwrite=True)
    kept_tracks = keep["particle"].nunique()
    # short_ids and flagged_ids can overlap (a short track could also be flagged
    # -- but short tracks aren't eligible so aren't flagged; still, report unique).
    print(f"\nWrote clean GEFF: {out}")
    print(f"  removed {len(flagged_ids)} flagged + "
          f"{len(short_ids)} short (span<{MIN_SPAN_KEEP}) "
          f"= {len(drop_ids)} tracklets dropped")
    print(f"  kept {kept_tracks} tracks ({g.number_of_nodes()} nodes)")
    return out


# ============================================================================
# napari
# ============================================================================

def show_napari(track_df: pd.DataFrame, feats: pd.DataFrame, run_id: str):
    import napari
    import zarr as _zarr

    images = sorted(IMAGES_DIR.glob("*.zarr"))
    np_arr = np.array(_zarr.open(images[IMG_NUM], mode="r")["s0"])

    flow_frames = np.array(
        _zarr.open_group(FLOW_PATH, mode="r")["flow_frames_XY"]
    )

    # Attach per-node suspect_score + flagged so the Tracks layers can color by
    # them. napari colors tracks by a per-vertex feature; broadcast each track's
    # value to all its vertices.
    df = track_df.sort_values(["particle", "frame"]).reset_index(drop=True)
    df = df.merge(
        feats[["suspect_score", "flagged"]],
        left_on="particle", right_index=True, how="left",
    )
    fill = df["suspect_score"].min()
    df["suspect_score"] = df["suspect_score"].fillna(fill)
    df["flagged"] = df["flagged"].fillna(False).astype(float)

    data = df[["particle", "frame", "y", "x"]].to_numpy()
    properties = {
        "suspect_score": df["suspect_score"].to_numpy(),
        "flagged": df["flagged"].to_numpy(),
    }

    # The FINAL kept set: same two cuts as write_clean_geff (flagged + short),
    # so this layer is exactly what tracks_clean.geff contains.
    drop_ids = set(feats.index[feats["flagged"]].astype(int).tolist())
    drop_ids |= set(feats.index[feats["span"] < MIN_SPAN_KEEP].astype(int).tolist())
    final = df[~df["particle"].astype(int).isin(drop_ids)]

    viewer = napari.Viewer()
    viewer.add_image(np_arr[:, 0], name="Raw image")
    viewer.add_image(flow_frames, name="Flow")

    # All tracks, colored by continuous suspect_score (the "rank" view).
    viewer.add_tracks(
        data, properties=properties,
        color_by="suspect_score", colormap="inferno",
        name=f"all tracks (suspect_score)",
        visible=False,
    )
    # All tracks, colored by the flag boolean (the "cull decision" view).
    viewer.add_tracks(
        data, properties=properties,
        color_by="flagged", colormap="inferno",
        name=f"all tracks (flagged)",
        visible=False,
    )
    # The FINAL filtered set -- what survives into tracks_clean.geff.
    if len(final):
        viewer.add_tracks(
            final[["particle", "frame", "y", "x"]].to_numpy(),
            name=f"FINAL kept ({final['particle'].nunique()} tracks)",
        )

    # GT tracks on top, always visible, for side-by-side validation.
    try:
        gt = pd.read_csv(GT_CSV)
        gt.columns = gt.columns.str.strip()
        viewer.add_tracks(
            gt[["particle", "frame", "y", "x"]].to_numpy(),
            name=f"GT ({gt['particle'].nunique()} validated)",
        )
    except FileNotFoundError:
        print(f"(GT CSV not found at {GT_CSV}; skipping GT layer)")

    napari.run()


# ============================================================================
# GT calibration
# ============================================================================

def load_gt_track_df() -> pd.DataFrame:
    """Load the validated GT CSV (frame, x, y, particle) into the [particle,
    frame, y, x] shape score_track expects. Identical downstream path to a run,
    so drift/turn are computed exactly as they are for solution tracks."""
    gt = pd.read_csv(GT_CSV)
    gt.columns = gt.columns.str.strip()
    return pd.DataFrame({
        "particle": gt["particle"].astype(int),
        "frame": gt["frame"].astype(int),
        "y": gt["y"].astype(float),
        "x": gt["x"].astype(float),
    }).sort_values(["particle", "frame"]).reset_index(drop=True)


def calibrate_on_gt(flow_vec) -> None:
    """Score every GT track through score_track and report drift/turn vs cutoffs.

    The GT is the ONLY thing that maps to 'correct.' If every GT track lands
    under both cutoffs, the flag is safe (won't eat real biology). If a GT track
    would be flagged, the cutoff that catches it is too tight -- raise it above
    that track's value. This can't prove the cutoffs catch every mislink (11
    tracks is too few), only that they don't kill known-good tracks.
    """
    gt_df = load_gt_track_df()
    n_tracks = gt_df["particle"].nunique()
    print(f"GT: {n_tracks} validated tracks, {len(gt_df)} detections")
    print(f"Cutoff under test: turn>={TURN_FLAG_RADIANS} (turn-only flag; "
          f"drift reported but not used)\n")

    rows = []
    for pid, group in gt_df.groupby("particle"):
        feat = score_track(group, flow_vec)
        feat["particle"] = int(pid)
        rows.append(feat)
    gt_feats = pd.DataFrame(rows).set_index("particle")

    # Would each GT track be flagged under the current TURN-ONLY cutoff?
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

    # Headroom: the worst GT turn is the floor you must stay above.
    elig_gt = gt_feats[(gt_feats["span"] >= MIN_SPAN_FOR_SCORE)
                       & gt_feats["turn_incoherence"].notna()]
    if len(elig_gt):
        print(f"\nWorst GT (eligible) turn -- keep cutoff ABOVE this:")
        print(f"  max GT turn : {elig_gt['turn_incoherence'].max():.2f} rad "
              f"(current cutoff {TURN_FLAG_RADIANS})")
        print(f"  (drift, unused: GT ranges "
              f"{elig_gt['cumulative_drift'].min():.2f}"
              f"-{elig_gt['cumulative_drift'].max():.2f} px/frame -- "
              f"overlaps bad tracks, which is why it was dropped)")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Score tracks by motion consistency with the flow field."
    )
    parser.add_argument("run_id", nargs="?", default=RUN_ID,
                        help=f"run_id under {RUNS_DIR} (default: {RUN_ID})")
    parser.add_argument("--no-napari", action="store_true",
                        help="skip the napari viewer, write CSV only")
    parser.add_argument("--gt", action="store_true",
                        help="calibration mode: score the GT tracks through the "
                             "same path and report drift/turn vs the cutoffs, "
                             "instead of scoring a run")
    parser.add_argument("--write-clean", action="store_true",
                        help="write tracks_clean.geff next to the run with all "
                             "flagged tracklets removed (the actual cull; "
                             "original tracks.geff is left untouched)")
    args = parser.parse_args()

    flow_raw = np.array(zarr.open_group(FLOW_PATH, mode="r")["flow_raw"])
    flow_vec = make_flow_sampler(flow_raw)

    # --- GT calibration short-circuit ---
    if args.gt:
        print(f"flow_raw shape: {flow_raw.shape}")
        calibrate_on_gt(flow_vec)
        return

    sg, meta = load_run(args.run_id)
    track_df = geff_to_track_df(sg)
    print(f"Run {args.run_id}: {track_df['particle'].nunique()} tracks, "
          f"{len(track_df)} detections | "
          f"max_gap={meta.get('max_gap')} gap_penalty={meta.get('gap_penalty')} "
          f"edge_const={meta.get('edge_selected_constant')}")

    print(f"flow_raw shape: {flow_raw.shape}")

    feats = score_all(track_df, flow_vec)

    # Save CSV next to the run.
    out_csv = RUNS_DIR / args.run_id / "flow_scores.csv"
    feats.to_csv(out_csv)
    print(f"Wrote {out_csv}")

    # Population summary: three groups.
    eligible = feats[feats["eligible"]]
    too_short = feats[~feats["eligible"]]
    flagged = feats[feats["flagged"]]
    print(f"\nPopulations:")
    print(f"  eligible (span>={MIN_SPAN_FOR_SCORE}, scorable): {len(eligible)}")
    print(f"  too short / unscorable (passed through):    {len(too_short)}")
    print(f"  FLAGGED (turn>={TURN_FLAG_RADIANS}):                    "
          f"{len(flagged)}")

    # The autonomous cull result: every flagged track, worst drift first.
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

    # Ranked review of eligible-but-not-flagged: the score sorts these for
    # eyeballing, but they cleared the absolute cutoffs.
    review = eligible[~eligible["flagged"]].sort_values(
        "suspect_score", ascending=False
    )
    print(f"\nTop eligible NON-flagged by suspect_score "
          f"(review if you want, but they passed the cull):")
    print(f"{'pid':>6} | {'suspect':>7} | {'drift':>6} | {'turn':>6} | {'span':>4}")
    print("-" * 44)
    for pid, r in review.head(15).iterrows():
        print(f"{pid:>6} | {r['suspect_score']:>7.2f} | "
              f"{r['cumulative_drift']:>6.2f} | {r['turn_incoherence']:>6.2f} | "
              f"{int(r['span']):>4}")

    # The actual cull: write a reduced GEFF with flagged tracklets removed.
    if args.write_clean:
        write_clean_geff(track_df, feats, args.run_id)

    if not args.no_napari:
        show_napari(track_df, feats, args.run_id)


if __name__ == "__main__":
    main()