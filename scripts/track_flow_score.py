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
import hashlib

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

# Manually validated ground-truth tracks (cols: frame, x, y, particle).
GT_CSV = BASE / "sg100_Well5_1018_valid_tracks.csv"

# Source movie for GT intensity sampling (the movie GT_CSV was validated on).
GT_MOVIE_ZARR = IMAGES_DIR / "sg100_Well5_1018.zarr"
ZARR_ARRAY_KEY = "s0"

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

# --- Intensity sampling (REPORT-ONLY) --------------------------------------
# Per-run outputs don't store pixel data, so intensity is sampled from the source
# .zarr. Two sources: raw detection channel, and `pre` = background-subtracted
# reconstruction matching detection (median-filtered mean projection subtracted).
# These MUST match preprocess_blob_det.py for `pre` to mean anything.
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


def track_uid_for_group(group: pd.DataFrame) -> str:
    g = group.sort_values("frame")[["frame", "y", "x"]]
    payload = [
        (int(r.frame), int(round(float(r.y) * 1_000_000)),
         int(round(float(r.x) * 1_000_000)))
        for r in g.itertuples(index=False)
    ]
    return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()


def add_track_uids(track_df: pd.DataFrame) -> pd.DataFrame:
    df = track_df.copy()
    uid_map = {
        int(pid): track_uid_for_group(group)
        for pid, group in df.groupby("particle", sort=False)
    }
    df["track_uid"] = df["particle"].astype(int).map(uid_map)
    return df


def geff_to_track_df(sg) -> pd.DataFrame:
    attrs = sg.graph.node_attrs(unpack=True).to_pandas()
    df = pd.DataFrame({
        "particle": attrs["tracklet_id"].astype(int),
        "frame": attrs["t"].astype(int),
        "y": attrs["pos_0"].astype(float),
        "x": attrs["pos_1"].astype(float),
    })
    return add_track_uids(df).sort_values(["particle", "frame"]).reset_index(drop=True)


# ============================================================================
# Flow sampling
# ============================================================================

def make_flow_sampler(flow_raw: np.ndarray):
    T, Y, X, _ = flow_raw.shape

    def flow_vec(t: int, y: float, x: float):
        tt = min(max(int(t), 0), T - 1)
        yy = min(max(int(round(y)), 0), Y - 1)
        xx = min(max(int(round(x)), 0), X - 1)
        v = flow_raw[tt, yy, xx]
        return float(v[1]), float(v[0])  # (dy, dx)

    return flow_vec


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


# ============================================================================
# Per-track features
# ============================================================================

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


# ============================================================================
# Clean GEFF writer
# ============================================================================

def write_clean_geff(track_df: pd.DataFrame, feats: pd.DataFrame,
                     run_id: str) -> Path:
    import networkx as nx

    flagged_ids = set(feats.index[feats["flagged"]].astype(int).tolist())
    short_ids = set(feats.index[feats["span"] < MIN_SPAN_KEEP].astype(int).tolist())
    drop_ids = flagged_ids | short_ids

    keep = track_df[~track_df["particle"].astype(int).isin(drop_ids)].copy()
    keep = keep.sort_values(["particle", "frame"]).reset_index(drop=True)

    g = nx.DiGraph()
    next_id = 0
    for pid, group in keep.groupby("particle"):
        group = group.sort_values("frame")
        prev = None
        for _, row in group.iterrows():
            nid = next_id
            next_id += 1
            g.add_node(nid, t=int(row["frame"]), y=float(row["y"]),
                       x=float(row["x"]), track_id=int(pid), tracklet_id=int(pid))
            if prev is not None:
                g.add_edge(prev, nid)
            prev = nid

    out = RUNS_DIR / run_id / "tracks_clean.geff"
    geff.write(g, str(out), zarr_format=3, overwrite=True)
    kept_tracks = keep["particle"].nunique()
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
    parser = argparse.ArgumentParser(
        description="Score tracks by motion consistency with the flow field."
    )
    parser.add_argument("run_id", nargs="?", default=RUN_ID)
    parser.add_argument("--no-napari", action="store_true")
    parser.add_argument("--gt", action="store_true",
                        help="GT calibration + intensity table")
    parser.add_argument("--write-clean", action="store_true")
    args = parser.parse_args()

    flow_raw = np.array(zarr.open_group(FLOW_PATH, mode="r")["flow_raw"])
    flow_vec = make_flow_sampler(flow_raw)

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