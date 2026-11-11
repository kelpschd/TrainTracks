#!/usr/bin/env python3
"""
Pool batch_cotracking.py outputs across movies and ask what cotracking does to
particle velocity.

Statistical unit: MOVIE (cluster). Tracks are nested in movies, so every
confidence interval is a cluster bootstrap that resamples movies, then takes
all tracks of the sampled movies. Within a movie each particle track counts once.

Tables written to <out>/summary/ :
    enrichment.csv            pooled observed episodes vs pooled shifted-null
    track_speed.csv           one row per particle track: cotracking flag, speeds
    speed_cotracking_vs_never.csv   track mean speed, cotracking vs never-cotracking
    paired_during_vs_outside.csv    within-track speed, during vs outside episodes
    paired_summary.csv        cluster-bootstrap mean paired delta, by organelle/class
    event_aligned_speed.csv   speed vs frame relative to episode onset
    episodes_all.csv          every episode, with movie/group
Figure: cotracking_velocity.png

Usage:  python summarize_cotracking_velocity.py /path/to/cotracking [--min-steps 2]
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


def group_of(movie: str) -> str:
    return movie.split("_Well")[0]            # sg100 / sg1035


# ---------------------------------------------------------------------------
# cluster bootstrap
# ---------------------------------------------------------------------------
def _cluster_stats(df, value, cluster, grp=None):
    if grp is None:
        g = df.groupby(cluster)[value].agg(["sum", "count"])
        return g["sum"].to_numpy(float), g["count"].to_numpy(float)
    parts = []
    for k in (0, 1):
        sub = df[df[grp] == k]
        g = sub.groupby(cluster)[value].agg(["sum", "count"]).reindex(
            df[cluster].unique(), fill_value=0)
        parts += [g["sum"].to_numpy(float), g["count"].to_numpy(float)]
    return parts


def boot_mean(df, value, cluster="movie", n=5000):
    """Mean of `value` with a cluster-bootstrap 95% CI."""
    df = df.dropna(subset=[value])
    if df.empty:
        return np.nan, np.nan, np.nan, 0, 0
    s, c = _cluster_stats(df, value, cluster)
    idx = RNG.integers(0, len(s), (n, len(s)))
    m = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
    return s.sum() / c.sum(), *np.percentile(m, [2.5, 97.5]), int(c.sum()), len(s)


def boot_diff(df, value, flag, cluster="movie", n=5000):
    """mean(flag==1) - mean(flag==0) with cluster-bootstrap CI."""
    df = df.dropna(subset=[value]).copy()
    df[flag] = df[flag].astype(int)
    if df[flag].nunique() < 2:
        return np.nan, np.nan, np.nan
    s1, c1, s0, c0 = [None] * 4
    s0, c0, s1, c1 = _cluster_stats(df, value, cluster, grp=flag)
    obs = s1.sum() / c1.sum() - s0.sum() / c0.sum()
    idx = RNG.integers(0, len(s1), (n, len(s1)))
    a, b = c1[idx].sum(1), c0[idx].sum(1)
    ok = (a > 0) & (b > 0)
    d = s1[idx].sum(1)[ok] / a[ok] - s0[idx].sum(1)[ok] / b[ok]
    return obs, *np.percentile(d, [2.5, 97.5])


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_all(root: Path):
    movies = sorted(p for p in root.iterdir() if p.is_dir() and (p / "particle_tracks.csv").exists())
    if not movies:
        raise FileNotFoundError(f"No per-movie outputs in {root}")
    data = {"P": {}, "ep": {o: [] for o in ORGS}, "null": {o: [] for o in ORGS}}
    for m in movies:
        data["P"][m.name] = pd.read_csv(m / "particle_tracks.csv")
        for o in ORGS:
            e = pd.read_csv(m / f"episodes_{o}.csv"); e["movie"] = m.name
            n = pd.read_csv(m / f"null_{o}.csv"); n["movie"] = m.name
            data["ep"][o].append(e); data["null"][o].append(n)
    def _cat(frames):
        keep = [f for f in frames if len(f)]
        return pd.concat(keep, ignore_index=True) if keep else frames[0]
    data["ep"] = {o: _cat(v) for o, v in data["ep"].items()}
    data["null"] = {o: _cat(v) for o, v in data["null"].items()}
    return data


# ---------------------------------------------------------------------------
# analyses
# ---------------------------------------------------------------------------
def enrichment(data, movies=None, label="all"):
    rows = []
    label_grp = label
    movies = list(data["P"]) if movies is None else list(movies)
    for o in ORGS:
        ep = data["ep"][o][data["ep"][o].movie.isin(movies)]
        nulls = data["null"][o][data["null"][o].movie.isin(movies)]
        n_perm = min((len(g) for _, g in nulls.groupby("movie") if len(g)), default=0)
        if n_perm == 0:
            continue
        obs_particles = ep[["movie", "particle"]].drop_duplicates().shape[0]
        n_particles = sum(data["P"][m].particle.nunique() for m in movies)
        # pooled null: permutation k across movies summed
        stack = {k: np.zeros(n_perm) for k in ("n_episodes", "particles_with", "frames")}
        for _, g in nulls.groupby("movie"):
            g = g.reset_index(drop=True).iloc[:n_perm]
            stack["n_episodes"] += g["n_episodes"].to_numpy()
            stack["particles_with"] += (g["frac_particles_with_episode"] * g["n_particles"]).to_numpy()
            stack["frames"] += g["particle_frames_in_episode"].to_numpy()
        obs = {"n_episodes": len(ep), "particles_with": obs_particles,
               "frames": int(ep["n_contact"].sum()) if len(ep) else 0}
        for k, label in (("n_episodes", "episodes"), ("particles_with", "particle tracks with >=1 episode"),
                         ("frames", "particle frames in episode")):
            nd = stack[k]
            rows.append((label_grp, o, label, obs[k], nd.mean(), *np.percentile(nd, [2.5, 97.5]),
                         obs[k] / nd.mean() if nd.mean() > 0 else np.nan,
                         (1 + (nd >= obs[k]).sum()) / (1 + len(nd)), n_particles))
    return pd.DataFrame(rows, columns=["movie_group", "organelle", "metric", "observed", "null_mean",
                                       "null_p2.5", "null_p97.5", "fold_over_null",
                                       "p_enrichment", "n_particle_tracks_total"])


def track_speed_table(data):
    rows = []
    for o in ORGS:
        ep = data["ep"][o]
        for movie, P in data["P"].items():
            st = ct.particle_steps(P, ep[ep.movie == movie])
            if st.empty:
                continue
            for pid, g in st.groupby("particle"):
                during = g[g.state == "during"]; outside = g[~g.state.isin(["during"])]
                rows.append((o, movie, pid, len(g), g.speed.mean(),
                             int((g.state != "never").any()),
                             during.speed.mean() if len(during) else np.nan, len(during),
                             outside.speed.mean() if len(outside) else np.nan, len(outside)))
    return pd.DataFrame(rows, columns=["organelle", "movie", "particle", "n_steps", "mean_speed",
                                       "cotracks", "speed_during", "n_during", "speed_outside", "n_outside"])


def cotracking_vs_never(ts):
    rows = []
    for o in ORGS:
        t = ts[ts.organelle == o]
        d, lo, hi = boot_diff(t, "mean_speed", "cotracks")
        for flag in (0, 1):
            m, l, h, n, k = boot_mean(t[t.cotracks == flag], "mean_speed")
            rows.append((o, "cotracking" if flag else "never", n, k, m, l, h))
        rows.append((o, "difference (cotracking - never)", len(t), t.movie.nunique(), d, lo, hi))
    return pd.DataFrame(rows, columns=["organelle", "group", "n_tracks", "n_movies",
                                       "mean_speed_px_per_frame", "ci95_lo", "ci95_hi"])


def paired_tables(data, min_steps):
    """Within-track during vs outside, overall and by episode class."""
    pair_rows, sum_rows = [], []
    for o in ORGS:
        ep_all = data["ep"][o]
        classes = [("all", None)] + [(c, c) for c in sorted(ep_all.episode_class.dropna().unique())]
        for label, cls in classes:
            parts = []
            for movie, P in data["P"].items():
                ep = ep_all[ep_all.movie == movie]
                if cls is not None:
                    ep = ep[ep.episode_class == cls]
                if ep.empty:
                    continue
                st = ct.particle_steps(P, ep)
                pr = ct.paired_during_vs_outside(st, min_steps=min_steps)
                if len(pr):
                    pr["movie"] = movie
                    parts.append(pr)
            if not parts:
                sum_rows.append((o, label, 0, 0, np.nan, np.nan, np.nan, np.nan, np.nan))
                continue
            pr = pd.concat(parts, ignore_index=True)
            pr["organelle"], pr["episode_class"] = o, label
            pair_rows.append(pr)
            m, lo, hi, n, k = boot_mean(pr, "delta_speed")
            ratio = (pr.speed_during / pr.speed_outside.replace(0, np.nan)).median()
            sum_rows.append((o, label, n, k, m, lo, hi, (pr.delta_speed > 0).mean(), ratio))
    paired = pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()
    summ = pd.DataFrame(sum_rows, columns=["organelle", "episode_class", "n_tracks", "n_movies",
                                           "mean_delta_speed", "ci95_lo", "ci95_hi",
                                           "frac_tracks_faster_during", "median_speed_ratio"])
    return paired, summ


def aligned_table(data):
    parts = []
    for o in ORGS:
        ep_all = data["ep"][o]
        for movie, P in data["P"].items():
            ep = ep_all[ep_all.movie == movie]
            if ep.empty:
                continue
            ev = ct.event_aligned_speed(ct.particle_steps(P, ep), ep)
            ev["organelle"], ev["movie"] = o, movie
            parts.append(ev)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------
def make_figure(out, ts, paired, aligned, ep_lys):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))

    t = ts[ts.organelle == "lysosome"]
    data = [t[t.cotracks == 0].mean_speed.dropna(), t[t.cotracks == 1].mean_speed.dropna()]
    ax[0].boxplot(data, showfliers=False)
    ax[0].set_xticks([1, 2]); ax[0].set_xticklabels([f"never\n(n={len(data[0])})", f"cotracks\n(n={len(data[1])})"])
    for i, d in enumerate(data):
        ax[0].scatter(np.random.default_rng(1).normal(i + 1, .05, len(d)), d, s=10, alpha=.5)
    ax[0].set_ylabel("track mean speed (px/frame)"); ax[0].set_title("Lysosome: cotracking vs never")

    pl = paired[(paired.get("organelle") == "lysosome") & (paired.get("episode_class") == "all")] if len(paired) else paired
    if len(pl):
        for _, r in pl.iterrows():
            ax[1].plot([0, 1], [r.speed_outside, r.speed_during], "-o", color="gray", alpha=.5, ms=3)
        ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["outside", "during"])
        ax[1].set_ylabel("speed (px/frame)"); ax[1].set_title(f"Within-track paired (n={len(pl)} tracks)")
    else:
        ax[1].text(.5, .5, "no paired tracks", ha="center"); ax[1].set_axis_off()

    if len(aligned):
        al = aligned[aligned.organelle == "lysosome"]
        for cls, g in al.groupby("episode_class"):
            m = g.groupby("rel_frame").speed.agg(["mean", "sem", "size"])
            m = m[m["size"] >= 3]
            ax[2].errorbar(m.index, m["mean"], yerr=m["sem"], marker="o", ms=3, capsize=2, label=f"{cls}")
        ax[2].axvline(0, color="k", ls=":")
        ax[2].set_xlabel("frame relative to cotracking onset"); ax[2].set_ylabel("particle speed (px/frame)")
        ax[2].set_title("Event-aligned (lysosome)"); ax[2].legend(fontsize=8)

    if len(ep_lys):
        ax[3].scatter(ep_lys.org_mean_speed, ep_lys.particle_mean_speed, s=18, alpha=.6)
        lim = np.nanmax([ep_lys.org_mean_speed.max(), ep_lys.particle_mean_speed.max()])
        ax[3].plot([0, lim], [0, lim], "k:")
        ax[3].set_xlabel("lysosome speed in episode"); ax[3].set_ylabel("particle speed in episode")
        ax[3].set_title("Speed match within episodes")
    fig.tight_layout(); fig.savefig(out / "cotracking_velocity.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--min-steps", type=int, default=2,
                    help="min particle steps in each of during/outside to enter the paired analysis")
    args = ap.parse_args()
    root = args.root.expanduser().resolve()
    out = root / "summary"; out.mkdir(exist_ok=True)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)

    data = load_all(root)
    print(f"{len(data['P'])} movies, {sum(p.particle.nunique() for p in data['P'].values())} particle tracks\n")

    ep_all = pd.concat([e.assign(organelle_type=o) for o, e in data["ep"].items()], ignore_index=True)
    ep_all["group"] = ep_all["movie"].map(group_of)
    ep_all.to_csv(out / "episodes_all.csv", index=False)

    groups = sorted({group_of(m) for m in data["P"]})
    enr = pd.concat([enrichment(data)] + [enrichment(data, [m for m in data["P"] if group_of(m) == g], g)
                                         for g in groups if len(groups) > 1], ignore_index=True)
    enr.to_csv(out / "enrichment.csv", index=False)
    print("ENRICHMENT (pooled observed vs pooled shifted-null; per movie group)\n", enr.round(3).to_string(index=False), "\n")
    for o in ORGS:
        e = data["ep"][o]
        if len(e):
            print(f"{o} episodes by class:", e.episode_class.value_counts().to_dict(),
                  "| by movie group:", e.movie.map(group_of).value_counts().to_dict())
    print()

    ts = track_speed_table(data); ts.to_csv(out / "track_speed.csv", index=False)
    cv = cotracking_vs_never(ts); cv.to_csv(out / "speed_cotracking_vs_never.csv", index=False)
    print("TRACK MEAN SPEED, cotracking vs never (px/frame, cluster-bootstrap over movies)\n",
          cv.round(2).to_string(index=False), "\n")

    paired, psum = paired_tables(data, args.min_steps)
    if len(paired):
        paired.to_csv(out / "paired_during_vs_outside.csv", index=False)
    psum.to_csv(out / "paired_summary.csv", index=False)
    print("WITHIN-TRACK PAIRED: speed during minus speed outside episodes (px/frame)\n",
          psum.round(2).to_string(index=False), "\n")

    al = aligned_table(data)
    if len(al):
        al.to_csv(out / "event_aligned_speed.csv", index=False)
    make_figure(out, ts, paired, al, data["ep"]["lysosome"])
    print(f"Wrote tables and cotracking_velocity.png to {out}")


if __name__ == "__main__":
    main()
