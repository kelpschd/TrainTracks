"""
Cotracking analysis: particle tracks vs. independently tracked organelle tracks.

Pure numpy/pandas on purpose. It does NOT touch segmentation, density filtering,
Kalman linking or track filtering -- feed it the DataFrames those steps already
produce.

Inputs (all pixel units, all with columns frame, x, y, particle):
    particle_tracks : particle tracks
    org_tracks      : organelle tracks (lysosome OR mitochondria; run once each)

Definitions
-----------
contact frame : particle and organelle centroid within `contact_px` on a frame
                where both are observed.
episode       : run of contact frames for one (particle, organelle) pair.
                Frames where either object is unobserved (track gap) may be
                bridged up to `max_gap` frames; a frame where BOTH are observed
                and apart ends the episode. Kept if it has >= `min_contact_frames`
                observed contact frames.
episode class : 'stationary_contact' organelle barely moved during the episode
                'co_transport'       organelle moved AND step directions agree
                                     (mean cosine >= cos_thresh)
                'moving_contact'     organelle moved but directions uncorrelated
                                     (co-residence, not coupled transport)

Cotracking is defined by proximity + persistence only. Velocity is NOT used to
define it, so comparing particle velocity inside vs. outside episodes is not
circular. Direction agreement is used only to sub-classify episodes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ["frame", "x", "y", "particle"]


# --------------------------------------------------------------------------
# 1. contact detection
# --------------------------------------------------------------------------
def _check(df, name):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns {missing}")
    return df[REQUIRED].copy()


def contact_frames(particle_tracks, org_tracks, contact_px):
    """Every (frame, particle, organelle) triple with centroid distance <= contact_px."""
    P = _check(particle_tracks, "particle_tracks")
    O = _check(org_tracks, "org_tracks")
    rows = []
    o_by_frame = {f: g for f, g in O.groupby("frame")}
    for f, pg in P.groupby("frame"):
        og = o_by_frame.get(f)
        if og is None:
            continue
        pxy = pg[["x", "y"]].to_numpy(float)
        oxy = og[["x", "y"]].to_numpy(float)
        d = np.hypot(pxy[:, None, 0] - oxy[None, :, 0], pxy[:, None, 1] - oxy[None, :, 1])
        pi, oi = np.nonzero(d <= contact_px)
        for a, b in zip(pi, oi):
            rows.append((f, pg["particle"].iat[a], og["particle"].iat[b], d[a, b]))
    return pd.DataFrame(rows, columns=["frame", "particle", "organelle", "dist"])


def _runs(frames, p_obs, o_obs, max_gap):
    """Split sorted contact frames into episodes, bridging only unobserved gaps."""
    runs, cur = [], [frames[0]]
    for f in frames[1:]:
        prev = cur[-1]
        gap = range(prev + 1, f)
        separated = any((g in p_obs) and (g in o_obs) for g in gap)
        if not separated and len(gap) <= max_gap:
            cur.append(f)
        else:
            runs.append(cur)
            cur = [f]
    runs.append(cur)
    return runs


# --------------------------------------------------------------------------
# 2. episodes + their motion descriptors
# --------------------------------------------------------------------------
def _xy_lookup(df):
    return {pid: g.sort_values("frame").set_index("frame")[["x", "y"]]
            for pid, g in df.groupby("particle")}


def _step_vectors(trk, frames):
    """Displacement per consecutive observed frame pair (dx, dy, frame_end)."""
    t = trk.loc[trk.index.isin(frames)]
    fr = t.index.to_numpy()
    xy = t[["x", "y"]].to_numpy(float)
    keep = np.diff(fr) == 1
    d = np.diff(xy, axis=0)[keep]
    return fr[1:][keep], d


def find_episodes(particle_tracks, org_tracks, contact_px=10.0, min_contact_frames=3,
                  max_gap=1, stationary_px=5.0, cos_thresh=0.5, eps_step=0.1):
    P = _check(particle_tracks, "particle_tracks")
    O = _check(org_tracks, "org_tracks")
    cf = contact_frames(P, O, contact_px)
    cols = ["particle", "organelle", "start", "end", "duration", "n_contact",
            "mean_dist", "org_net_disp", "particle_net_disp", "mean_cos", "n_cos_steps",
            "particle_mean_speed", "org_mean_speed", "episode_class"]
    if cf.empty:
        return pd.DataFrame(columns=cols)

    P_xy, O_xy = _xy_lookup(P), _xy_lookup(O)
    out = []
    for (pid, oid), g in cf.groupby(["particle", "organelle"]):
        g = g.sort_values("frame")
        p_obs, o_obs = set(P_xy[pid].index), set(O_xy[oid].index)
        for run in _runs(g["frame"].tolist(), p_obs, o_obs, max_gap):
            if len(run) < min_contact_frames:
                continue
            s, e = run[0], run[-1]
            span = range(s, e + 1)
            pt = P_xy[pid].loc[P_xy[pid].index.isin(span)]
            ot = O_xy[oid].loc[O_xy[oid].index.isin(span)]
            p_net = float(np.hypot(*(pt.iloc[-1] - pt.iloc[0]).to_numpy())) if len(pt) > 1 else 0.0
            o_net = float(np.hypot(*(ot.iloc[-1] - ot.iloc[0]).to_numpy())) if len(ot) > 1 else 0.0

            # direction agreement on frames where both took a 1-frame step
            fp, dp = _step_vectors(P_xy[pid], span)
            fo, do = _step_vectors(O_xy[oid], span)
            common, ip, io = np.intersect1d(fp, fo, return_indices=True)
            cos = []
            for a, b in zip(dp[ip], do[io]):
                na, nb = np.hypot(*a), np.hypot(*b)
                if na > eps_step and nb > eps_step:
                    cos.append(float(a @ b / (na * nb)))
            mean_cos = float(np.mean(cos)) if cos else np.nan

            if o_net < stationary_px:
                cls = "stationary_contact"
            elif np.isfinite(mean_cos) and mean_cos >= cos_thresh:
                cls = "co_transport"
            else:
                cls = "moving_contact"
            p_spd = float(np.mean(np.hypot(dp[:, 0], dp[:, 1]))) if len(dp) else np.nan
            o_spd = float(np.mean(np.hypot(do[:, 0], do[:, 1]))) if len(do) else np.nan
            out.append((pid, oid, s, e, e - s + 1, len(run),
                        float(g.loc[g["frame"].isin(run), "dist"].mean()),
                        o_net, p_net, mean_cos, len(cos), p_spd, o_spd, cls))
    return pd.DataFrame(out, columns=cols)


# --------------------------------------------------------------------------
# 3. particle behaviour before / during / between / after cotracking
# --------------------------------------------------------------------------
def particle_steps(particle_tracks, episodes):
    """One row per particle step (consecutive observed frames) labelled by state.

    state: 'during'  step lies inside an episode
           'before' / 'between' / 'after'  relative to that particle's episodes
           'never'   particle has no episodes
    speed is px/frame (displacement divided by frame gap).
    """
    P = _check(particle_tracks, "particle_tracks")
    rows = []
    for pid, g in P.groupby("particle"):
        g = g.sort_values("frame")
        fr = g["frame"].to_numpy()
        xy = g[["x", "y"]].to_numpy(float)
        ep = episodes[episodes["particle"] == pid]
        for i in range(1, len(fr)):
            f0, f1 = fr[i - 1], fr[i]
            dt = f1 - f0
            disp = xy[i] - xy[i - 1]
            if ep.empty:
                state = "never"
            elif ((ep["start"] <= f0) & (ep["end"] >= f1)).any():
                state = "during"
            elif f1 <= ep["start"].min():
                state = "before"
            elif f0 >= ep["end"].max():
                state = "after"
            else:
                state = "between"
            rows.append((pid, f0, f1, dt, float(np.hypot(*disp)) / dt, disp[0] / dt, disp[1] / dt, state))
    return pd.DataFrame(rows, columns=["particle", "frame_start", "frame_end", "dt",
                                       "speed", "vx", "vy", "state"])


def segment_summary(steps):
    """Per-particle-track mean speed and straightness by state (paired design)."""
    recs = []
    for (pid, state), g in steps.groupby(["particle", "state"]):
        net = np.hypot(g["vx"].mul(g["dt"]).sum(), g["vy"].mul(g["dt"]).sum())
        path = g["speed"].mul(g["dt"]).sum()
        recs.append((pid, state, len(g), g["speed"].mean(), net / path if path > 0 else np.nan))
    s = pd.DataFrame(recs, columns=["particle", "state", "n_steps", "mean_speed", "straightness"])
    return s


def paired_during_vs_outside(steps, min_steps=2):
    """Each particle that has BOTH 'during' and outside steps -> one paired row."""
    s = steps.copy()
    s["grp"] = np.where(s["state"] == "during", "during",
                        np.where(s["state"] == "never", "never", "outside"))
    cols = ["particle", "mean_during", "mean_outside", "size_during", "size_outside",
            "speed_during", "speed_outside", "delta_speed"]
    sub = s[s["grp"] != "never"]
    if sub.empty:
        return pd.DataFrame(columns=cols)
    g = sub.groupby(["particle", "grp"])["speed"].agg(["mean", "size"]).unstack()
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    # a subset may contain only 'during' or only 'outside' steps -> missing columns
    g = g.reindex(columns=["mean_during", "mean_outside", "size_during", "size_outside"])
    g = g.dropna()
    g = g[(g["size_during"] >= min_steps) & (g["size_outside"] >= min_steps)]
    g = g.rename(columns={"mean_during": "speed_during", "mean_outside": "speed_outside"})
    g["delta_speed"] = g["speed_during"] - g["speed_outside"]
    return g.reset_index()


def event_aligned_speed(steps, episodes, window=(-5, 10)):
    """Particle step speed vs. frame relative to episode onset (rel = frame_end - start).

    One row per (episode, step) so episodes are the unit of replication. Steps
    from the same particle can appear for several episodes; that is intended.
    """
    rows = []
    for _, ep in episodes.iterrows():
        s = steps[steps["particle"] == ep["particle"]]
        rel = s["frame_end"] - ep["start"]
        m = (rel >= window[0]) & (rel <= window[1])
        for r, sp in zip(rel[m], s.loc[m, "speed"]):
            rows.append((ep["particle"], ep["organelle"], ep["episode_class"], int(r), sp))
    return pd.DataFrame(rows, columns=["particle", "organelle", "episode_class", "rel_frame", "speed"])


# --------------------------------------------------------------------------
# 4. null model
# --------------------------------------------------------------------------
def _stats(particle_tracks, episodes):
    n_p = particle_tracks["particle"].nunique()
    p_with = episodes["particle"].nunique() if len(episodes) else 0
    frames_in = int(episodes["n_contact"].sum()) if len(episodes) else 0
    return {
        "n_episodes": len(episodes),
        "frac_particles_with_episode": p_with / n_p if n_p else np.nan,
        "particle_frames_in_episode": frames_in,
        "frac_particle_frames_in_episode": frames_in / len(particle_tracks) if len(particle_tracks) else np.nan,
    }


def shifted_null(particle_tracks, org_tracks, n_perm=200, mode="local",
                 shift_range=(40.0, 120.0), image_shape=None, seed=0, **episode_kw):
    """Rigidly translate the whole organelle track set and recompute episodes.

    mode='local'  shift by a random vector with |shift| in shift_range (px).
                  Keeps organelles in roughly the same cellular neighbourhood, so
                  the null respects local density (recommended default).
    mode='global' shift uniformly over the image, wrapping at the borders
                  (needs image_shape=(H, W)). Ignores density, so tends to be a
                  liberal null: it moves organelles into empty space.
    Only the translation is random; each organelle track keeps its own shape and
    the movie-wide temporal pattern, so frame alignment is preserved.
    """
    rng = np.random.default_rng(seed)
    O = _check(org_tracks, "org_tracks")
    res = []
    for _ in range(n_perm):
        if mode == "local":
            r = rng.uniform(*shift_range)
            th = rng.uniform(0, 2 * np.pi)
            dx, dy = r * np.cos(th), r * np.sin(th)
            Os = O.assign(x=O["x"] + dx, y=O["y"] + dy)
        elif mode == "global":
            if image_shape is None:
                raise ValueError("global mode needs image_shape=(H, W)")
            H, W = image_shape
            Os = O.assign(x=(O["x"] + rng.uniform(0, W)) % W, y=(O["y"] + rng.uniform(0, H)) % H)
        else:
            raise ValueError("mode must be 'local' or 'global'")
        ep = find_episodes(particle_tracks, Os, **episode_kw)
        res.append(_stats(particle_tracks, ep))
    return pd.DataFrame(res)


def compare_to_null(particle_tracks, observed_episodes, null_df):
    obs = _stats(_check(particle_tracks, "particle_tracks"), observed_episodes)
    rows = []
    for k, v in obs.items():
        nd = null_df[k].to_numpy(float)
        p = (1 + np.sum(nd >= v)) / (1 + len(nd))   # one-sided, enrichment
        rows.append((k, v, np.mean(nd), np.percentile(nd, 2.5), np.percentile(nd, 97.5),
                     v / np.mean(nd) if np.mean(nd) > 0 else np.nan, p))
    return pd.DataFrame(rows, columns=["metric", "observed", "null_mean", "null_p2.5",
                                       "null_p97.5", "fold_over_null", "p_enrichment"])


# --------------------------------------------------------------------------
# 5. leading vs. trailing: where is the particle relative to the organelle's heading?
# --------------------------------------------------------------------------
def episode_offsets(particle_tracks, org_tracks, episodes, org_delay_frames=0.0,
                    min_org_step=0.5):
    """Per contact-frame offset of the particle from the organelle, in the organelle's frame.

    heading : organelle displacement per frame (central difference over adjacent frames,
              one-sided at the ends of the track); needs |step| >= min_org_step px.
    along   : component of (particle - organelle) ALONG the heading, px.
              > 0 particle is AHEAD of the organelle, < 0 it is TRAILING.
    lateral : signed component perpendicular to the heading (+ = left of heading, in
              image coordinates with y pointing down), px.
    cos_ang : cos(angle between offset and heading) = along / distance.
              +1 always ahead, -1 always behind, ~0 no preferred side.

    org_delay_frames: if the organelle channel is imaged LATER than the particle channel
    by d frames, the organelle appears to have moved ahead; its position at the particle's
    time is o - v*d. Default 0 (no correction). Check the ND2 per-channel timestamps.
    """
    P_xy = _xy_lookup(_check(particle_tracks, "particle_tracks"))
    O_xy = _xy_lookup(_check(org_tracks, "org_tracks"))
    rows = []
    for _, ep in episodes.iterrows():
        pt, ot = P_xy[ep["particle"]], O_xy[ep["organelle"]]
        for f in range(int(ep["start"]), int(ep["end"]) + 1):
            if f not in pt.index or f not in ot.index:
                continue
            if (f - 1) in ot.index and (f + 1) in ot.index:
                v = (ot.loc[f + 1] - ot.loc[f - 1]).to_numpy(float) / 2.0
            elif (f + 1) in ot.index:
                v = (ot.loc[f + 1] - ot.loc[f]).to_numpy(float)
            elif (f - 1) in ot.index:
                v = (ot.loc[f] - ot.loc[f - 1]).to_numpy(float)
            else:
                continue
            sp = float(np.hypot(*v))
            o = ot.loc[f].to_numpy(float) - v * org_delay_frames
            rel = pt.loc[f].to_numpy(float) - o
            dist = float(np.hypot(*rel))
            if sp < min_org_step:
                along = lateral = cos_ang = np.nan
            else:
                h = v / sp
                along = float(rel @ h)
                lateral = float(rel @ np.array([-h[1], h[0]]))
                cos_ang = along / dist if dist > 0 else np.nan
            rows.append((ep["particle"], ep["organelle"], int(ep["start"]), ep["episode_class"],
                         f, sp, dist, along, lateral, cos_ang))
    return pd.DataFrame(rows, columns=["particle", "organelle", "start", "episode_class", "frame",
                                       "org_step", "dist", "along", "lateral", "cos_ang"])


def summarize_offsets(frames, min_frames=3, consistent_frac=0.8):
    """One row per episode. 'ahead' / 'trailing' only if the sign is consistent."""
    recs = []
    for (pid, oid, st), g in frames.dropna(subset=["along"]).groupby(["particle", "organelle", "start"]):
        if len(g) < min_frames:
            continue
        frac_ahead = float((g["along"] > 0).mean())
        m = float(g["along"].mean())
        if m > 0 and frac_ahead >= consistent_frac:
            lab = "ahead"
        elif m < 0 and (1 - frac_ahead) >= consistent_frac:
            lab = "trailing"
        else:
            lab = "mixed"
        recs.append((pid, oid, st, g["episode_class"].iat[0], len(g), m, float(g["along"].std(ddof=1)),
                     frac_ahead, float(g["lateral"].abs().mean()), float(g["dist"].mean()),
                     float(g["cos_ang"].mean()), lab))
    return pd.DataFrame(recs, columns=["particle", "organelle", "start", "episode_class", "n_frames",
                                       "mean_along", "sd_along", "frac_frames_ahead",
                                       "mean_abs_lateral", "mean_dist", "mean_cos_ang", "position"])
