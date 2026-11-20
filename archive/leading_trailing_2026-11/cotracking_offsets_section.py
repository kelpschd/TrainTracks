"""Leading vs trailing offsets: the two functions removed from cotracking/cotracking.py (2026-11-20).

Verbatim copy of section 5 of cotracking.py as of the pre-cleanup tag, kept for reference.
They use helpers of that module (_xy_lookup, _check, ...) and `import numpy as np`, `import pandas as pd`;
to run them, check out tag pre-cleanup-2026-11-19 (or paste them back into cotracking.py).
"""

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
