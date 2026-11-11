#!/usr/bin/env python3
"""
Are particles consistently AHEAD of or TRAILING the organelle they cotrack with?

Uses the files batch_cotracking.py already wrote per movie
(particle_tracks.csv, tracks_<organelle>.csv, episodes_<organelle>.csv); no re-tracking.

For every contact frame of a moving-organelle episode the particle's offset from the
organelle centroid is projected onto the organelle's direction of motion
(see cotracking.episode_offsets). Positive 'along' = particle ahead, negative = trailing.
Stationary organelles have no heading and are skipped.

Outputs under <cotracking_dir>/summary/:
    offsets_frames.csv      one row per contact frame
    offsets_episodes.csv    one row per episode (mean along, frac ahead, ahead/trailing/mixed)
    offsets_summary.csv     pooled across episodes, cluster bootstrap over movies

Usage:
    python leading_trailing.py /path/to/cotracking
    python leading_trailing.py /path/to/cotracking --org-delay-frames 0.1   # channel-timing correction
    python leading_trailing.py /path/to/cotracking --classes co_transport moving_contact

CHECK CHANNEL TIMING FIRST. If the organelle channel is acquired after the particle channel,
a moving organelle is recorded further along its path than the particle, which makes the
particle look like it is trailing. In a synthetic test a 0.2-frame lag made 22 of 24
zero-offset episodes read as 'trailing'. Get the per-channel timestamps from the ND2
(or acquisition settings) and pass the lag, in frames, to --org-delay-frames.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cotracking as ct  # noqa: E402

ORGS = ("lysosome", "mitochondria")
RNG = np.random.default_rng(0)


def boot_mean(values, clusters, n=5000):
    """Mean with 95% CI, resampling clusters (movies)."""
    df = pd.DataFrame({"v": values, "c": clusters}).dropna()
    if df.empty:
        return np.nan, np.nan, np.nan
    g = df.groupby("c")["v"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    idx = RNG.integers(0, len(s), (n, len(s)))
    m = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
    return s.sum() / c.sum(), *np.percentile(m, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--classes", nargs="+", default=["co_transport"],
                    help="episode classes to include (default: co_transport)")
    ap.add_argument("--org-delay-frames", type=float, default=0.0)
    ap.add_argument("--min-org-step", type=float, default=0.5, help="px/frame needed to define a heading")
    ap.add_argument("--pixel-size", type=float, default=0.108333333333333, help="um per px")
    args = ap.parse_args()

    root = args.root.expanduser().resolve()
    out = root / "summary"; out.mkdir(exist_ok=True)
    movies = sorted(p for p in root.iterdir() if p.is_dir() and (p / "particle_tracks.csv").exists())
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

    frames_all, eps_all = [], []
    for m in movies:
        P = pd.read_csv(m / "particle_tracks.csv")
        for o in ORGS:
            tp, ep_p = m / f"tracks_{o}.csv", m / f"episodes_{o}.csv"
            if not (tp.exists() and ep_p.exists()):
                continue
            O, ep = pd.read_csv(tp), pd.read_csv(ep_p)
            ep = ep[ep["episode_class"].isin(args.classes)]
            if ep.empty or O.empty:
                continue
            fr = ct.episode_offsets(P, O, ep, org_delay_frames=args.org_delay_frames,
                                    min_org_step=args.min_org_step)
            if fr.empty:
                continue
            fr["movie"], fr["organelle_type"] = m.name, o
            frames_all.append(fr)
            es = ct.summarize_offsets(fr)
            es["movie"], es["organelle_type"] = m.name, o
            eps_all.append(es)

    if not eps_all:
        print("No moving-organelle episodes to analyse."); return
    frames = pd.concat(frames_all, ignore_index=True)
    eps = pd.concat(eps_all, ignore_index=True)
    eps["mean_along_um"] = eps["mean_along"] * args.pixel_size
    eps["guide"] = eps["movie"].str.split("_Well").str[0]
    frames.to_csv(out / "offsets_frames.csv", index=False)
    eps.to_csv(out / "offsets_episodes.csv", index=False)

    rows = []
    for (o, g), d in list(eps.groupby(["organelle_type", "guide"])) + [((o, "all"), d) for o, d in eps.groupby("organelle_type")]:
        m_al, lo, hi = boot_mean(d["mean_along_um"], d["movie"])
        m_cs, clo, chi = boot_mean(d["mean_cos_ang"], d["movie"])
        pos = d["position"].value_counts()
        rows.append((o, g, len(d), d["movie"].nunique(), m_al, lo, hi, m_cs, clo, chi,
                     pos.get("ahead", 0), pos.get("trailing", 0), pos.get("mixed", 0),
                     (d["mean_along"] > 0).mean()))
    summ = pd.DataFrame(rows, columns=[
        "organelle", "guide", "n_episodes", "n_movies", "mean_along_um", "ci_lo", "ci_hi",
        "mean_cos_angle", "cos_ci_lo", "cos_ci_hi", "n_ahead", "n_trailing", "n_mixed",
        "frac_episodes_ahead"])
    summ.to_csv(out / "offsets_summary.csv", index=False)
    print(f"classes: {args.classes} | org_delay_frames: {args.org_delay_frames}\n")
    print(summ.round(3).to_string(index=False))
    try:
        from scipy.stats import binomtest
        for _, r in summ[summ.guide == "all"].iterrows():
            k, n = int(r.n_ahead), int(r.n_ahead + r.n_trailing)
            if n:
                print(f"{r.organelle}: ahead vs trailing among consistent episodes "
                      f"{k}/{n}, binomial p = {binomtest(k, n, 0.5).pvalue:.3g}")
    except Exception:
        pass

    # Timing diagnostic: a channel lag of d frames adds -d * (organelle speed) to 'along',
    # so the offset would scale with speed (slope ~ -d, intercept ~ 0). A speed-independent
    # offset (intercept != 0, slope ~ 0) points to real geometry instead.
    print("\nalong (px) = intercept + slope * organelle speed (px/frame); episode-level means")
    for o, d in eps.groupby("organelle_type"):
        spd = frames.dropna(subset=["along"]).query("organelle_type == @o") \
                    .groupby(["movie", "particle", "organelle", "start"])["org_step"].mean()
        e = d.set_index(["movie", "particle", "organelle", "start"]).join(spd.rename("org_speed")).dropna(subset=["org_speed"])
        if len(e) >= 5 and e["org_speed"].std() > 0:
            slope, icpt = np.polyfit(e["org_speed"], e["mean_along"], 1)
            print(f"  {o}: intercept {icpt:+.2f} px, slope {slope:+.3f} frames (n={len(e)} episodes)")
        else:
            print(f"  {o}: too few episodes for the timing diagnostic")
    print(f"\nWrote offsets_*.csv to {out}")


if __name__ == "__main__":
    main()
