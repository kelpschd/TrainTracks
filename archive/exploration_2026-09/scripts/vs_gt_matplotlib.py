"""
Visualize traccuracy vs-GT results, reading straight from traccuracy_vs_gt.csv
so the figure regenerates after re-running the comparison (e.g. after adding
skip-edge / gap-closing to the tracker).

Four panels:
  1. Recall + target effectiveness across thresholds, with the
     node-recall -> target-effectiveness FRAGMENTATION BAND shaded. That band
     is the headline: it should NARROW once gap-closing stitches tracklets
     back together (node recall stays put, target effectiveness climbs).
  2a/2b. Recovered vs missed NODES and EDGES as absolute counts, so the small
     sample (11 GT tracks) is visually honest -- threshold differences are a
     handful of detections.
  3. Precision / purity vs predicted-detection count, shown together to make
     clear the precision rise is a partial-GT ARTIFACT, not improvement.

Trustworthy metrics: recall-side + target effectiveness.
Artifact (partial GT): precision, purity, F1.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

BASE = Path("/Users/kelpschdj/Documents/DataTecnica/TTU/TrainTracks")
CSV = BASE / "traccuracy_vs_gt.csv"
OUT = BASE / "vs_gt_metrics.png"

# GT totals (also derivable from the CSV columns below, but pinned for labels)
GT_NODES = 168
GT_EDGES = 157


def load():
    df = pd.read_csv(CSV)
    df.columns = df.columns.str.strip()
    # clean threshold label: "0_00015" -> "0.00015"
    df["thr_label"] = df["threshold"].astype(str).str.replace("_", ".", regex=False)
    return df.reset_index(drop=True)


def plot(df):
    x = np.arange(len(df))
    labels = df["thr_label"].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Tracking accuracy vs. validated tracks  "
        f"(11 GT tracks · {GT_NODES} nodes · {GT_EDGES} edges · match dist 15 px)",
        fontsize=13, fontweight="bold",
    )

    # -----------------------------------------------------------------
    # Panel 1: recall lines + fragmentation band
    # -----------------------------------------------------------------
    ax = axes[0, 0]
    node_r = df["Node Recall"].to_numpy()
    edge_r = df["Edge Recall"].to_numpy()
    targ_e = df["target_effectiveness"].to_numpy()

    # fragmentation band: node recall (cells found) down to target
    # effectiveness (how much of a GT track a single tracklet covers)
    ax.fill_between(
        x, targ_e, node_r, color="#f97316", alpha=0.15,
        label="fragmentation gap\n(node recall − target eff.)",
    )
    for i in x:  # annotate the band width at each threshold
        gap = node_r[i] - targ_e[i]
        ax.annotate(
            f"{gap*100:.0f} pts",
            (x[i], (node_r[i] + targ_e[i]) / 2),
            ha="center", va="center", fontsize=8, color="#c2410c",
        )

    ax.plot(x, node_r, "-o", color="#2563eb", lw=2.5, label="Node recall")
    ax.plot(x, edge_r, "-o", color="#dc2626", lw=2.5, label="Edge recall")
    ax.plot(x, targ_e, "--o", color="#16a34a", lw=2.5, label="Target effectiveness")

    ax.set_title("1 · Recall & fragmentation band\n"
                 "(band should NARROW after gap-closing)", fontsize=10)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("blob detection threshold")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=8, loc="center right")

    # -----------------------------------------------------------------
    # Panel 2a: nodes recovered vs missed (counts)
    # -----------------------------------------------------------------
    ax = axes[0, 1]
    tp_n = df["True Positive Nodes"].to_numpy()
    fn_n = df["False Negative Nodes"].to_numpy()
    ax.bar(x, tp_n, color="#2563eb", label="Recovered (TP)")
    ax.bar(x, fn_n, bottom=tp_n, color="#e5e7eb", label="Missed (FN)")
    for i in x:
        ax.text(x[i], tp_n[i] / 2, str(tp_n[i]), ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
    ax.set_title(f"2a · Validated NODES recovered (of {GT_NODES})\n"
                 "differences are a few detections — within noise", fontsize=10)
    ax.set_ylim(0, GT_NODES)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("blob detection threshold")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.legend(fontsize=8)

    # -----------------------------------------------------------------
    # Panel 2b: edges recovered vs missed (counts)
    # -----------------------------------------------------------------
    ax = axes[1, 0]
    tp_e = df["True Positive Edges"].to_numpy()
    fn_e = df["False Negative Edges"].to_numpy()
    ax.bar(x, tp_e, color="#dc2626", label="Recovered (TP)")
    ax.bar(x, fn_e, bottom=tp_e, color="#e5e7eb", label="Missed (FN)")
    for i in x:
        ax.text(x[i], tp_e[i] / 2, str(tp_e[i]), ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
    ax.set_title(f"2b · Validated EDGES recovered (of {GT_EDGES})\n"
                 "grey slice bigger than nodes — the linking gap", fontsize=10)
    ax.set_ylim(0, GT_EDGES)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("blob detection threshold")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.legend(fontsize=8)

    # -----------------------------------------------------------------
    # Panel 3: precision/purity artifact vs detection count
    # -----------------------------------------------------------------
    ax = axes[1, 1]
    width = 0.35
    ax.bar(x - width/2, df["Node Precision"], width,
           color="#f59e0b", label="Node precision")
    ax.bar(x + width/2, df["track_purity"], width,
           color="#fbbf24", label="Track purity")
    ax.set_ylabel("precision / purity")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, max(df["track_purity"].max(), df["Node Precision"].max()) * 1.3)

    ax2 = ax.twinx()
    ax2.plot(x, df["pred_detections"], "-s", color="#6b7280", lw=2,
             label="Predicted detections")
    ax2.set_ylabel("predicted detections (total)")

    ax.set_title("3 · Precision/purity rise is an ARTIFACT\n"
                 "climbs only because detection count falls", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("blob detection threshold")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    # merge legends from both axes
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT}")
    plt.show()


if __name__ == "__main__":
    df = load()
    plot(df)
