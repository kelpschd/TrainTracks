"""Per-track motion-consistency scoring and the clean-track selection rule.

Moved out of ``scripts/track_flow_score.py`` (and replaces the separate copy that lived
in ``batch_pipeline.py``) so that ONE definition of ``turn_incoherence`` is used
everywhere.

``turn_incoherence`` is the mean absolute signed turn angle (arctan2 of cross/dot)
between consecutive MOVING steps (steps faster than ``PAUSE_SPEED_PX_PER_FRAME``);
pauses are neutral. The 1.4 rad cutoff ``TURN_FLAG_RADIANS`` was calibrated against
this definition with ``track_flow_score.py --gt``. A track is flagged when it is
eligible (span >= ``MIN_SPAN_FOR_SCORE`` and stats available) and its turn_incoherence
is >= the cutoff; tracks with span < ``MIN_SPAN_KEEP`` are also dropped from the clean set.

Pure numpy/pandas apart from the GEFF writer, which imports networkx/geff lazily.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from TrainTracks.ids import track_uid_for_group

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
COMPOSITE_WEIGHTS = dict(
    residual_var=1.0,
    turn_incoherence=1.0,
    cumulative_drift=1.0,
)

MIN_SPAN_FOR_SCORE = 5

# --- Flag cutoff (autonomous cull): TURN-ONLY ------------------------------
# GT tracks are strongly directional (max turn well under 1.0 rad/step), so 1.4
# leaves margin above known-good and below the ~1.57 random-walk line. Calibrate
# with --gt: confirm all GT tracks land flagged=False.
TURN_FLAG_RADIANS = 1.4

# Reported next to tracks; NOT a flag axis (GT drifts up to ~9 px/frame straight).
DRIFT_REPORT_PX_PER_FRAME = 8.0

# Minimum span to KEEP in the clean GEFF (independent of the flag).
MIN_SPAN_KEEP = 5

MIN_EDGES_FOR_STATS = 2

# Pause: a step at/below this per-frame speed is neutral (not a turn).
PAUSE_SPEED_PX_PER_FRAME = 0.5

TURN_THRESHOLDS_DEG = (30.0, 45.0, 60.0)

SELF_REVISIT_RADIUS_PX = 5.0
SELF_REVISIT_MIN_FRAME_GAP = 3


def make_flow_sampler(flow_raw: np.ndarray):
    T, Y, X, _ = flow_raw.shape

    def flow_vec(t: int, y: float, x: float):
        tt = min(max(int(t), 0), T - 1)
        yy = min(max(int(round(y)), 0), Y - 1)
        xx = min(max(int(round(x)), 0), X - 1)
        v = flow_raw[tt, yy, xx]
        return float(v[1]), float(v[0])  # (dy, dx)

    return flow_vec

def _intensity_continuity(values: np.ndarray) -> dict:
    """Continuity features from a per-node intensity series along one track.

    cv       : std/mean -- overall steadiness.
    max_jump : largest single-step fractional change |dI| / median(I). A swap
               shows a big step; smooth bleaching does not. (A translation BURST
               can ALSO show a big step -- that's the calibration question.)
    n_big_steps : count of steps with fractional change > 0.5 (diagnostic).
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    out = dict(mean=np.nan, cv=np.nan, max_jump=np.nan, n_big_steps=0)
    if v.size == 0:
        return out
    mean = float(np.mean(v))
    out["mean"] = mean
    if v.size >= 2:
        sd = float(np.std(v, ddof=0))
        out["cv"] = float(sd / mean) if mean > 1e-9 else np.nan
        baseline = float(np.median(v))
        if baseline > 1e-9:
            frac_steps = np.abs(np.diff(v)) / baseline
            out["max_jump"] = float(np.max(frac_steps))
            out["n_big_steps"] = int(np.sum(frac_steps > 0.5))
    return out

def _unit(v: np.ndarray):
    n = np.linalg.norm(v)
    return v / n if n > 1e-8 else None

def _empty_intensity_feats() -> dict:
    d = {}
    for s in ("raw", "pre"):
        d[f"intensity_{s}_mean"] = np.nan
        d[f"intensity_{s}_cv"] = np.nan
        d[f"intensity_{s}_max_jump"] = np.nan
        d[f"intensity_{s}_n_big_steps"] = 0
    return d

def score_track(group: pd.DataFrame, flow_vec,
                intensity_raw=None, intensity_pre=None) -> dict:
    """Compute motion-consistency + (optional) intensity-continuity features."""
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
        turn_incoherence=np.nan,
        total_turn_radians=np.nan, total_turn_degrees=np.nan,
        net_turn_radians=np.nan, turn_per_moving_step=np.nan,
        n_turn_events_30deg=np.nan, n_turn_events_45deg=np.nan,
        n_turn_events_60deg=np.nan,
        n_motion_steps=0, n_pause_steps=0, pause_fraction=np.nan,
        path_length=np.nan, net_displacement=np.nan,
        path_straightness=np.nan, turning_density=np.nan,
        n_self_revisit_pairs=0, n_self_revisit_points=0,
        self_revisit_fraction=np.nan,
        cumulative_drift=np.nan, max_step_gap=0, frac_skip=np.nan,
    )
    base.update(_empty_intensity_feats())
    if n_edges < 1:
        return base

    residuals, velocities, gaps = [], [], []

    for i in range(n_edges):
        t0 = int(frames[i])
        gap = max(int(frames[i + 1] - frames[i]), 1)
        gaps.append(gap)

        disp = np.array([ys[i + 1] - ys[i], xs[i + 1] - xs[i]])
        vel = disp / gap
        velocities.append(vel)

        pred_steps = []
        pos = np.array([ys[i], xs[i]])
        for k in range(gap):
            fdy, fdx = flow_vec(t0 + k, pos[0], pos[1])
            f = np.array([fdy, fdx])
            pred_steps.append(f)
            pos = pos + f
        pred_vel = np.mean(pred_steps, axis=0)
        residuals.append(float(np.linalg.norm(vel - pred_vel)))

    residuals = np.array(residuals)
    gaps = np.array(gaps)

    mean_residual = float(np.mean(residuals))
    residual_var = float(np.var(residuals)) if n_edges >= MIN_EDGES_FOR_STATS else np.nan

    speeds = np.array([np.linalg.norm(v) for v in velocities], dtype=float)
    pause_mask = speeds <= PAUSE_SPEED_PX_PER_FRAME
    moving_indices = [i for i, is_pause in enumerate(pause_mask) if not is_pause]
    n_pause_steps = int(np.sum(pause_mask))
    n_motion_steps = int(len(moving_indices))
    pause_fraction = float(n_pause_steps / n_edges) if n_edges else np.nan

    signed_turns = []
    for j in range(len(moving_indices) - 1):
        a = _unit(velocities[moving_indices[j]])
        b = _unit(velocities[moving_indices[j + 1]])
        if a is None or b is None:
            continue
        cross = float(a[0] * b[1] - a[1] * b[0])
        dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
        signed_turns.append(float(np.arctan2(cross, dot)))

    abs_turns = np.abs(np.asarray(signed_turns, dtype=float))
    if abs_turns.size:
        total_turn_radians = float(np.sum(abs_turns))
        total_turn_degrees = float(np.degrees(total_turn_radians))
        net_turn_radians = float(np.abs(np.sum(np.asarray(signed_turns, dtype=float))))
        turn_per_moving_step = float(total_turn_radians / abs_turns.size)
        n_turn_events_30deg = int(np.sum(abs_turns >= np.deg2rad(30.0)))
        n_turn_events_45deg = int(np.sum(abs_turns >= np.deg2rad(45.0)))
        n_turn_events_60deg = int(np.sum(abs_turns >= np.deg2rad(60.0)))
    else:
        total_turn_radians = total_turn_degrees = net_turn_radians = np.nan
        turn_per_moving_step = np.nan
        n_turn_events_30deg = n_turn_events_45deg = n_turn_events_60deg = np.nan

    turn_incoherence = turn_per_moving_step

    step_lengths = np.array([np.linalg.norm(v) * gaps[i]
                             for i, v in enumerate(velocities)], dtype=float)
    path_length = float(np.sum(step_lengths))
    net_displacement = float(np.linalg.norm(np.array([ys[-1] - ys[0], xs[-1] - xs[0]])))
    path_straightness = float(net_displacement / path_length) if path_length > 1e-8 else np.nan
    turning_density = (
        float(total_turn_radians / path_length)
        if path_length > 1e-8 and np.isfinite(total_turn_radians) else np.nan
    )

    coords = np.column_stack([ys, xs])
    n_self_revisit_pairs = 0
    revisit_points = set()
    if n >= 2:
        radius_sq = SELF_REVISIT_RADIUS_PX ** 2
        for i in range(n):
            for j in range(i + 1, n):
                if abs(int(frames[j]) - int(frames[i])) < SELF_REVISIT_MIN_FRAME_GAP:
                    continue
                delta = coords[j] - coords[i]
                if float(np.dot(delta, delta)) <= radius_sq:
                    n_self_revisit_pairs += 1
                    revisit_points.add(i)
                    revisit_points.add(j)
    n_self_revisit_points = len(revisit_points)
    self_revisit_fraction = float(n_self_revisit_points / n) if n else np.nan

    pos = np.array([ys[0], xs[0]])
    t = int(frames[0])
    for i in range(n_edges):
        gap = int(gaps[i])
        for k in range(gap):
            fdy, fdx = flow_vec(t + k, pos[0], pos[1])
            pos = pos + np.array([fdy, fdx])
        t += gap
    actual_end = np.array([ys[-1], xs[-1]])
    cumulative_drift = float(np.linalg.norm(pos - actual_end) / max(span - 1, 1))

    # --- Intensity continuity (REPORT-ONLY) ---------------------------------
    intensity_feats = {}
    for src_name, sampler in (("raw", intensity_raw), ("pre", intensity_pre)):
        if sampler is None:
            intensity_feats[f"intensity_{src_name}_mean"] = np.nan
            intensity_feats[f"intensity_{src_name}_cv"] = np.nan
            intensity_feats[f"intensity_{src_name}_max_jump"] = np.nan
            intensity_feats[f"intensity_{src_name}_n_big_steps"] = 0
            continue
        vals = np.array([sampler(int(frames[k]), ys[k], xs[k])
                         for k in range(n)], dtype=float)
        c = _intensity_continuity(vals)
        intensity_feats[f"intensity_{src_name}_mean"] = c["mean"]
        intensity_feats[f"intensity_{src_name}_cv"] = c["cv"]
        intensity_feats[f"intensity_{src_name}_max_jump"] = c["max_jump"]
        intensity_feats[f"intensity_{src_name}_n_big_steps"] = c["n_big_steps"]

    return dict(
        n_nodes=n, n_edges=n_edges, span=span,
        mean_residual=mean_residual, residual_var=residual_var,
        turn_incoherence=turn_incoherence,
        total_turn_radians=total_turn_radians,
        total_turn_degrees=total_turn_degrees,
        net_turn_radians=net_turn_radians,
        turn_per_moving_step=turn_per_moving_step,
        n_turn_events_30deg=n_turn_events_30deg,
        n_turn_events_45deg=n_turn_events_45deg,
        n_turn_events_60deg=n_turn_events_60deg,
        n_motion_steps=n_motion_steps, n_pause_steps=n_pause_steps,
        pause_fraction=pause_fraction,
        path_length=path_length, net_displacement=net_displacement,
        path_straightness=path_straightness, turning_density=turning_density,
        n_self_revisit_pairs=n_self_revisit_pairs,
        n_self_revisit_points=n_self_revisit_points,
        self_revisit_fraction=self_revisit_fraction,
        cumulative_drift=cumulative_drift,
        max_step_gap=int(gaps.max()), frac_skip=float(np.mean(gaps > 1)),
        **intensity_feats,
    )

def _zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std(ddof=0)
    if sd < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd

def score_all(track_df: pd.DataFrame, flow_vec,
              intensity_raw=None, intensity_pre=None) -> pd.DataFrame:
    rows = []
    for pid, group in track_df.groupby("particle"):
        feat = score_track(group, flow_vec, intensity_raw, intensity_pre)
        feat["particle"] = int(pid)
        if "track_uid" in group.columns:
            feat["track_uid"] = str(group["track_uid"].iloc[0])
        else:
            feat["track_uid"] = track_uid_for_group(group)
        rows.append(feat)
    feats = pd.DataFrame(rows).set_index("particle")

    eligible = (
        (feats["span"] >= MIN_SPAN_FOR_SCORE)
        & feats["residual_var"].notna()
        & feats["turn_incoherence"].notna()
    )
    feats["eligible"] = eligible

    comp = pd.Series(np.nan, index=feats.index)
    if eligible.any():
        z = pd.DataFrame(index=feats.index[eligible])
        for col, w in COMPOSITE_WEIGHTS.items():
            z[col] = _zscore(feats.loc[eligible, col]) * w
        comp.loc[eligible] = z.sum(axis=1)
    feats["suspect_score"] = comp

    feats["flagged"] = eligible & (feats["turn_incoherence"] >= TURN_FLAG_RADIANS)
    return feats.sort_values("suspect_score", ascending=False)


def select_clean_tracks(track_df: pd.DataFrame, feats: pd.DataFrame,
                        min_span_keep: int = MIN_SPAN_KEEP):
    """Apply the clean-track rule. Returns (kept_track_df, flagged_ids, short_ids).

    Dropped: tracks with ``flagged`` True and tracks with ``span < min_span_keep``.
    ``feats`` is indexed by ``particle`` (as returned by ``score_all``).
    """
    for col in ("flagged", "span"):
        if col not in feats.columns:
            raise ValueError(f"Score table is missing required {col!r} column")
    flagged_ids = set(feats.index[feats["flagged"].fillna(False).astype(bool)].astype(int))
    short_ids = set(feats.index[feats["span"].astype(float) < min_span_keep].astype(int))
    drop_ids = flagged_ids | short_ids
    keep = track_df[~track_df["particle"].astype(int).isin(drop_ids)].copy()
    keep = keep.sort_values(["particle", "frame"]).reset_index(drop=True)
    return keep, flagged_ids, short_ids


def write_track_df_geff(track_df: pd.DataFrame, output_path) -> None:
    """Write a track DataFrame (particle, frame, y, x) as a GEFF of linear tracklets.

    ``particle`` is stored as both ``track_id`` and ``tracklet_id``.
    """
    import geff
    import networkx as nx

    g = nx.DiGraph()
    next_id = 0
    for pid, group in track_df.sort_values(["particle", "frame"]).groupby("particle", sort=True):
        prev = None
        for _, row in group.iterrows():
            nid = next_id
            next_id += 1
            g.add_node(nid, t=int(row["frame"]), y=float(row["y"]), x=float(row["x"]),
                       track_id=int(pid), tracklet_id=int(pid))
            if prev is not None:
                g.add_edge(prev, nid)
            prev = nid
    geff.write(g, str(output_path), zarr_format=3, overwrite=True)


def write_clean_geff(track_df: pd.DataFrame, feats: pd.DataFrame, output_path,
                     min_span_keep: int = MIN_SPAN_KEEP) -> dict:
    """Select the clean tracks and write them to ``output_path`` (the raw GEFF is never touched)."""
    keep, flagged_ids, short_ids = select_clean_tracks(track_df, feats, min_span_keep)
    write_track_df_geff(keep, output_path)
    drop_ids = flagged_ids | short_ids
    return {
        "clean_geff": str(Path(output_path)),
        "n_clean_tracks": int(keep["particle"].nunique()) if len(keep) else 0,
        "n_clean_detections": int(len(keep)),
        "n_short_removed": len(short_ids),
        "n_flagged_removed": len(flagged_ids),
        "n_tracks_removed": len(drop_ids),
    }
