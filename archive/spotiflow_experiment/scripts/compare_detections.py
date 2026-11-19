# compare_detections.py
#
# Detection-level comparison of blob CSVs (t, y, x): LoG vs Spotiflow.
# No tracking involved. For each CSV it reports
#   * total spots, spots/frame
#   * GT DETECTION RECALL: fraction of hand-validated GT points (GT_CSV) that
#     have a detection within MATCH_TOL px in the same frame. This is the
#     fairest detector-level metric you have, because the GT only covers
#     validated tracks (precision is not measurable -- same caveat as
#     traccuracy_compare.py).
#   * pairwise agreement vs the reference CSV: one-to-one matches within
#     MATCH_TOL px (per frame, Hungarian), reported as % of reference found
#     and % of other that is also in reference.
#
# Usage: edit CSVS below, then  python scripts/compare_detections.py

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree, distance_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/Users/kelpschdj/Documents/DataTecnica/TTU/TrainTracks")
GT_CSV = BASE / "sg100_Well5_1018_valid_tracks.csv"      # frame, x, y, particle
OUT_DIR = BASE / "detection_comparison"

# label -> CSV. First entry is the reference for pairwise agreement.
CSVS = {
    "LoG_0.00021":      BASE / "blobs/blobs_0_00021.csv",
    "LoG_0.00019":      BASE / "blobs/blobs_0_00019.csv",
    "LoG_0.00017":      BASE / "blobs/blobs_0_00017.csv",
    "LoG_0.00015":      BASE / "blobs/blobs_0_00015.csv",
    "SF_general_raw_p0.5":   BASE / "blobs/blobs_spotiflow_general_raw_p0_5.csv",
    "SF_general_raw_p0.3":   BASE / "blobs/blobs_spotiflow_general_raw_p0_3.csv",
    "SF_general_bgsub_p0.5": BASE / "blobs/blobs_spotiflow_general_bgsub_p0_5.csv",
}

MATCH_TOL = 5.0     # px. Detection-to-GT / detection-to-detection tolerance.
                    # (traccuracy uses 15 px for track nodes; this is tighter
                    # because it asks about localization, not track matching.)


def gt_recall(det: pd.DataFrame, gt: pd.DataFrame, tol: float):
    hits, dists = 0, []
    for _, r in gt.iterrows():
        d = det[det["t"] == r["frame"]]
        if len(d) == 0:
            continue
        dist = np.hypot(d["y"].to_numpy() - r["y"], d["x"].to_numpy() - r["x"]).min()
        dists.append(dist)
        hits += dist <= tol
    return hits / len(gt), (np.median([x for x in dists if x <= tol]) if hits else np.nan)


def match_counts(ref: pd.DataFrame, oth: pd.DataFrame, tol: float) -> int:
    n = 0
    for t in sorted(set(ref["t"]) & set(oth["t"])):
        a = ref.loc[ref["t"] == t, ["y", "x"]].to_numpy(float)
        b = oth.loc[oth["t"] == t, ["y", "x"]].to_numpy(float)
        D = distance_matrix(a, b)
        D[D > tol] = 1e6
        r, c = linear_sum_assignment(D)
        n += int((D[r, c] < 1e6).sum())
    return n


def main():
    OUT_DIR.mkdir(exist_ok=True)
    gt = pd.read_csv(GT_CSV, encoding="utf-8-sig")
    dets = {k: pd.read_csv(v) for k, v in CSVS.items() if Path(v).exists()}
    missing = [k for k, v in CSVS.items() if not Path(v).exists()]
    if missing:
        print("Skipping (file not found):", ", ".join(missing))
    ref_name = next(iter(dets))
    ref = dets[ref_name]

    rows = []
    for name, d in dets.items():
        rec, med = gt_recall(d, gt, MATCH_TOL)
        m = match_counts(ref, d, MATCH_TOL)
        rows.append(dict(
            detector=name,
            n_spots=len(d),
            spots_per_frame=len(d) / max(d["t"].nunique(), 1),
            gt_det_recall=rec,
            median_loc_err_px=med,
            **{f"pct_of_{ref_name}_found": m / len(ref),
               "pct_of_this_in_ref": m / len(d)},
        ))
    summary = pd.DataFrame(rows).set_index("detector")
    pd.set_option("display.width", None, "display.max_columns", None)
    print(summary.round(3))
    summary.to_csv(OUT_DIR / "detection_summary.csv")

    # Spots per frame
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, d in dets.items():
        c = d.groupby("t").size()
        ax.plot(c.index, c.values, label=name, marker=".")
    ax.set_xlabel("frame"); ax.set_ylabel("spots"); ax.set_yscale("log"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT_DIR / "spots_per_frame.png", dpi=150); plt.close(fig)

    # Recall vs spot count (one point per detector)
    fig, ax = plt.subplots(figsize=(5, 4))
    for name, row in summary.iterrows():
        ax.scatter(row["n_spots"], row["gt_det_recall"],
                   marker="s" if name.startswith("SF") else "o")
        ax.annotate(name, (row["n_spots"], row["gt_det_recall"]), fontsize=6)
    ax.set_xscale("log"); ax.set_xlabel("total spots"); ax.set_ylabel(f"GT recall (<= {MATCH_TOL:g}px)")
    fig.tight_layout(); fig.savefig(OUT_DIR / "gt_recall_vs_spots.png", dpi=150); plt.close(fig)
    print(f"Saved outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
