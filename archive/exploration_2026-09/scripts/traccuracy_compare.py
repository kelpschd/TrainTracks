"""
Compare blob-threshold tracking solutions with traccuracy, two ways:

  1. GT vs each threshold   -> ACCURACY (how well each threshold recovers the
                               manually validated tracks)
  2. threshold vs threshold -> AGREEMENT (how much the solutions differ from
                               each other; NEITHER is ground truth)

IMPORTANT CAVEATS on interpretation
-----------------------------------
(1) GT comparisons: the GT covers only ~11 validated tracks while each solution
    contains many more. Every unmatched predicted track is counted as a false
    positive even when it's a legitimate un-validated track. So precision /
    purity / FP-based numbers (and CTC/AOGM-style summaries) are overly
    pessimistic and NOT trustworthy here. Trust the GT->pred direction: of the
    11 validated tracks, how many does each threshold recover, and how
    completely -- i.e. Target Effectiveness in TrackOverlapMetrics and
    recall-side BasicMetrics. Those are fair across thresholds because the
    matching distance is identical for all (MATCH_THRESHOLD).

(2) threshold-vs-threshold comparisons: neither side is truth, so run_metrics'
    gt/pred labels are just "reference" and "other." The numbers describe
    DIFFERENCE/AGREEMENT between two solutions, not correctness. They are also
    ASYMMETRIC: 15-vs-17 != 17-vs-15 (FP/FN and the two directions swap). Read
    the pairwise matrix as "how much of A does B reproduce," not "B is better
    than A."

Verified against traccuracy's documented run_metrics / PointMatcher /
TrackOverlapMetrics / BasicMetrics API. The TrackingGraph and PointMatcher
constructor signatures print on startup so you can confirm argument names match
your installed version (developed against v0.4.x).
"""

import inspect
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

from traccuracy import run_metrics, TrackingGraph
from traccuracy.matchers import PointMatcher
from traccuracy.metrics import TrackOverlapMetrics, BasicMetrics
from funtracks.import_export import import_from_geff


# ---------------------------
# Config
# ---------------------------

BASE = Path("/Users/kelpschdj/Documents/DataTecnica/TTU/TrainTracks")
GT_CSV = "/Users/kelpschdj/Documents/DataTecnica/TTU/TrainTracks/sg100_Well5_1018_valid_tracks.csv"  # columns: frame, x, y, particle
THRESHOLDS = ["0_00015", "0_00017", "0_00019", "0_00021"]

# Single matching distance (pixels) used for ALL four comparisons. Holding this
# constant is what makes the cross-threshold comparison meaningful. Anchor it to
# your cell size / localization error (blob detection used ~5 px dilation).
MATCH_THRESHOLD = 15.0


# ---------------------------
# Graph construction (shared by GT and predictions -> identical structure)
# ---------------------------

def track_df_to_nx(df):
    """
    track_df (particle, frame, y, x) -> networkx.DiGraph.
    Nodes carry t/y/x; edges connect each particle's detections in frame order
    (spanning gaps as skip edges).

    Node IDs must be integers (traccuracy validates this), so we use a running
    counter -- the value is arbitrary, only uniqueness matters.
    """
    g = nx.DiGraph()
    df = df.sort_values(["particle", "frame"]).reset_index(drop=True)
    next_id = 0
    for pid, group in df.groupby("particle"):
        group = group.sort_values("frame")
        prev = None
        for _, row in group.iterrows():
            nid = next_id
            next_id += 1
            g.add_node(nid, t=int(row["frame"]), y=float(row["y"]), x=float(row["x"]))
            if prev is not None:
                g.add_edge(prev, nid)
            prev = nid
    return g


def geff_to_track_df(sg):
    """funtracks solution graph -> track_df (particle, frame, y, x)."""
    attrs = sg.graph.node_attrs(unpack=True).to_pandas()
    return pd.DataFrame({
        "particle": attrs["tracklet_id"],
        "frame": attrs["t"].astype(int),
        "y": attrs["pos_0"].astype(float),
        "x": attrs["pos_1"].astype(float),
    })


def make_tracking_graph(nx_graph, name):
    """
    Wrap a networkx graph in a traccuracy TrackingGraph.

    NOTE: confirm these kwargs against your installed traccuracy. The signature
    is printed on startup. Common form is:
        TrackingGraph(graph, frame_key="t", location_keys=("y", "x"), name=...)
    """
    return TrackingGraph(
        nx_graph,
        frame_key="t",
        location_keys=("y", "x"),
        name=name,
    )


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    # Print the two constructor signatures we care about, so mismatches are
    # obvious immediately rather than as a cryptic TypeError later.
    print("TrackingGraph signature:", inspect.signature(TrackingGraph.__init__))
    print("PointMatcher signature: ", inspect.signature(PointMatcher.__init__))
    print("-" * 60)

    # Ground truth
    gt_df = pd.read_csv(GT_CSV)
    gt_graph = make_tracking_graph(track_df_to_nx(gt_df), name="GT")
    print(f"GT: {gt_df['particle'].nunique()} tracks, {len(gt_df)} detections")
    print("-" * 60)

def _extract_results(r):
    """
    Normalize one metric result into a flat dict, tolerating the couple of
    shapes traccuracy versions return: an object with `.results`, a dict with
    a "results" key, or a plain dict of metric_name -> value.
    """
    if hasattr(r, "results"):
        return dict(r.results)
    if isinstance(r, dict):
        if "results" in r and isinstance(r["results"], dict):
            return dict(r["results"])
        return dict(r)
    raise TypeError(f"Unexpected metric result type: {type(r)}")


def run_pair(gt_nx, pred_nx, gt_name, pred_name, threshold):
    """
    Run both metrics on one (reference, other) pair; return a flat dict.

    Builds FRESH TrackingGraph objects and a fresh matcher every call.
    traccuracy annotates graphs in place with error flags during run_metrics,
    so reusing a graph across calls leaves stale annotations and triggers
    "Only GT graph has node errors annotated". Rebuilding each time avoids that.
    """
    gt_graph = make_tracking_graph(gt_nx.copy(), name=gt_name)
    pred_graph = make_tracking_graph(pred_nx.copy(), name=pred_name)
    matcher = PointMatcher(threshold=threshold)

    results, _ = run_metrics(
        gt_data=gt_graph,
        pred_data=pred_graph,
        matcher=matcher,
        metrics=[TrackOverlapMetrics(), BasicMetrics()],
    )

    flat = {}
    # `results` may be a list of per-metric results, or already a single dict.
    if isinstance(results, dict):
        for v in results.values():
            if isinstance(v, dict):
                flat.update(v)
        if not flat:  # was already flat
            flat.update(results)
    else:
        for r in results:
            flat.update(_extract_results(r))
    return flat


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    print("TrackingGraph signature:", inspect.signature(TrackingGraph.__init__))
    print("PointMatcher signature: ", inspect.signature(PointMatcher.__init__))
    print("-" * 60)

    # Build GT networkx graph (TrackingGraphs are built fresh per comparison)
    gt_df = pd.read_csv(GT_CSV)
    gt_nx = track_df_to_nx(gt_df)
    print(f"GT: {gt_df['particle'].nunique()} tracks, {len(gt_df)} detections")

    # Build every threshold networkx graph once
    pred_nx = {}
    pred_info = {}
    for thr in THRESHOLDS:
        sg = import_from_geff(
            directory=str(BASE / "tracking_runs" / f"blobs_{thr}.geff"),
            node_name_map={"time": "t", "pos": ["y", "x"]},
        )
        pred_df = geff_to_track_df(sg)
        pred_nx[thr] = track_df_to_nx(pred_df)
        pred_info[thr] = (pred_df["particle"].nunique(), len(pred_df))
        print(f"  blobs_{thr}: {pred_info[thr][0]} tracks, "
              f"{pred_info[thr][1]} detections")

    # -------------------------------------------------------------
    # (1) ACCURACY: GT vs each threshold
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("(1) ACCURACY -- GT vs each threshold")
    print("    Trust Target Effectiveness / recall; distrust precision/FP")
    print("=" * 60)

    gt_rows = []
    for thr in THRESHOLDS:
        flat = run_pair(gt_nx, pred_nx[thr], "GT", f"pred_{thr}", MATCH_THRESHOLD)
        flat["threshold"] = thr
        flat["pred_tracks"], flat["pred_detections"] = pred_info[thr]
        gt_rows.append(flat)
        print(f"\n--- GT vs {thr} ---")
        for k, v in flat.items():
            print(f"  {k}: {v}")

    gt_summary = pd.DataFrame(gt_rows).set_index("threshold")

    # -------------------------------------------------------------
    # (2) AGREEMENT: threshold vs threshold (asymmetric, both directions)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("(2) AGREEMENT -- threshold vs threshold (NOT accuracy)")
    print("    'ref vs other' = how much of ref the other reproduces.")
    print("    Asymmetric: both directions computed.")
    print("=" * 60)

    pair_rows = []
    for ref, other in itertools.permutations(THRESHOLDS, 2):
        flat = run_pair(
            pred_nx[ref], pred_nx[other],
            f"ref_{ref}", f"other_{other}", MATCH_THRESHOLD,
        )
        flat["reference"] = ref
        flat["other"] = other
        pair_rows.append(flat)
        print(f"\n--- ref {ref} vs other {other} ---")
        for k, v in flat.items():
            print(f"  {k}: {v}")

    pair_summary = pd.DataFrame(pair_rows).set_index(["reference", "other"])

    # -------------------------------------------------------------
    # Save + print summaries
    # -------------------------------------------------------------
    pd.set_option("display.max_columns", None, "display.width", None)

    print("\n" + "=" * 60)
    print("SUMMARY (1): ACCURACY vs GT")
    print("=" * 60)
    print(gt_summary)
    gt_summary.to_csv(BASE / "traccuracy_vs_gt.csv")

    print("\n" + "=" * 60)
    print("SUMMARY (2): AGREEMENT between thresholds")
    print("=" * 60)
    print(pair_summary)
    pair_summary.to_csv(BASE / "traccuracy_pairwise.csv")

    print(f"\nSaved: {BASE / 'traccuracy_vs_gt.csv'}")
    print(f"Saved: {BASE / 'traccuracy_pairwise.csv'}")