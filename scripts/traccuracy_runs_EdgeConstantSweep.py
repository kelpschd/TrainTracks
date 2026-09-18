"""
Compare tracking RUNS (from candidate_graph.py's run system) with traccuracy,
two ways:

  1. GT vs each run    -> ACCURACY (how well each run recovers the manually
                          validated tracks)
  2. run vs run        -> AGREEMENT (how much the solutions differ; neither is
                          ground truth)

Runs are specified by run_id; each run's GEFF path is read from its
metadata.json, so this stays in sync with the run system automatically.

IMPORTANT CAVEATS
-----------------
(1) GT comparisons: GT covers only ~11 validated tracks while each solution has
    many more, so every unmatched predicted track counts as a false positive
    even when it's a legitimate un-validated track. Precision / purity / FP
    numbers are overly pessimistic -- DISTRUST them. Trust recall-side metrics
    and Target Effectiveness (GT->pred direction). Fair across runs because
    MATCH_THRESHOLD is identical for all.
(2) run-vs-run: neither side is truth; numbers describe DIFFERENCE, not
    correctness, and are ASYMMETRIC (A-vs-B != B-vs-A).

The TrackingGraph / PointMatcher signatures print on startup so you can confirm
argument names match your installed traccuracy (developed against v0.4.x).
"""

import json
import inspect
import itertools
from pathlib import Path

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
RUNS_DIR = BASE / "tracking_runs" / "runs"
GT_CSV = BASE / "sg100_Well5_1018_valid_tracks.csv"  # cols: frame, x, y, particle

# Runs to compare: {label: run_id}. Edge-constant sweep candidates.
RUNS = {
    "pen20": "20260916_104812_d776",
    "pen10": "20260916_105301_9452",
    "pen5": "20260916_105750_a1ad",
    "pen2": "20260916_110242_2203",
    "pen0": "20260916_110734_4622",
}

# Single matching distance (px), identical for ALL comparisons -- this is what
# makes cross-run numbers comparable. Anchor to cell size / localization error.
MATCH_THRESHOLD = 15.0

# Whether to also run the (n choose 2)*2 run-vs-run agreement comparisons.
DO_PAIRWISE = True


# ---------------------------
# Graph construction (shared by GT and predictions -> identical structure)
# ---------------------------

def track_df_to_nx(df):
    """
    track_df (particle, frame, y, x) -> networkx.DiGraph.
    Integer node IDs (traccuracy requires int); edges connect each particle's
    detections in frame order.
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
    """Wrap a networkx graph in a traccuracy TrackingGraph."""
    return TrackingGraph(
        nx_graph, frame_key="t", location_keys=("y", "x"), name=name,
    )


def load_run_nx(run_id):
    """run_id -> (networkx graph for traccuracy, metadata dict)."""
    run_dir = RUNS_DIR / run_id
    with open(run_dir / "metadata.json") as f:
        meta = json.load(f)
    sg = import_from_geff(
        directory=str(run_dir / "tracks.geff"),
        node_name_map={"time": "t", "pos": ["y", "x"]},
    )
    return track_df_to_nx(geff_to_track_df(sg)), meta


# ---------------------------
# Metric running
# ---------------------------

def _extract_results(r):
    if hasattr(r, "results"):
        return dict(r.results)
    if isinstance(r, dict):
        if "results" in r and isinstance(r["results"], dict):
            return dict(r["results"])
        return dict(r)
    raise TypeError(f"Unexpected metric result type: {type(r)}")


def run_pair(ref_nx, other_nx, ref_name, other_name, threshold):
    """
    Both metrics on one (reference, other) pair -> flat dict.
    Fresh TrackingGraphs + matcher each call (traccuracy annotates in place, so
    reuse leaves stale annotations and errors out).
    """
    ref_graph = make_tracking_graph(ref_nx.copy(), name=ref_name)
    other_graph = make_tracking_graph(other_nx.copy(), name=other_name)
    matcher = PointMatcher(threshold=threshold)

    results, _ = run_metrics(
        gt_data=ref_graph,
        pred_data=other_graph,
        matcher=matcher,
        metrics=[TrackOverlapMetrics(), BasicMetrics()],
    )

    flat = {}
    if isinstance(results, dict):
        for v in results.values():
            if isinstance(v, dict):
                flat.update(v)
        if not flat:
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

    # GT
    gt_df = pd.read_csv(GT_CSV)
    gt_nx = track_df_to_nx(gt_df)
    print(f"GT: {gt_df['particle'].nunique()} tracks, {len(gt_df)} detections")

    # Runs (networkx + metadata), loaded once
    run_nx = {}
    for label, run_id in RUNS.items():
        nxg, meta = load_run_nx(run_id)
        run_nx[label] = nxg
        print(f"  {label} ({run_id}): {nxg.number_of_nodes()} nodes | "
              f"edge_const={meta.get('edge_selected_constant')} "
              f"max_gap={meta.get('max_gap')} gap_penalty={meta.get('gap_penalty')} "
              f"skip_edges={meta.get('n_skip_edges')}")

    # -------------------------------------------------------------
    # (1) ACCURACY: GT vs each run
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("(1) ACCURACY -- GT vs each run")
    print("    Trust Target Effectiveness / recall; distrust precision/FP")
    print("=" * 60)

    gt_rows = []
    for label in RUNS:
        flat = run_pair(gt_nx, run_nx[label], "GT", label, MATCH_THRESHOLD)
        flat["run"] = label
        gt_rows.append(flat)

    gt_summary = pd.DataFrame(gt_rows).set_index("run")

    # focused view of the trustworthy columns
    trust_cols = [c for c in [
        "Node Recall", "Edge Recall", "target_effectiveness",
        "track_fractions", "True Positive Nodes", "False Negative Nodes",
        "True Positive Edges", "False Negative Edges",
    ] if c in gt_summary.columns]

    pd.set_option("display.max_columns", None, "display.width", None)
    print("\nTrustworthy metrics (recall side):")
    print(gt_summary[trust_cols])
    print("\nFull table:")
    print(gt_summary)
    gt_summary.to_csv(BASE / "traccuracy_runs_vs_gt.csv")

    # -------------------------------------------------------------
    # (2) AGREEMENT: run vs run
    # -------------------------------------------------------------
    if DO_PAIRWISE and len(RUNS) > 1:
        print("\n" + "=" * 60)
        print("(2) AGREEMENT -- run vs run (NOT accuracy, asymmetric)")
        print("=" * 60)

        pair_rows = []
        for ref, other in itertools.permutations(RUNS.keys(), 2):
            flat = run_pair(run_nx[ref], run_nx[other],
                            f"ref_{ref}", f"other_{other}", MATCH_THRESHOLD)
            flat["reference"] = ref
            flat["other"] = other
            pair_rows.append(flat)

        pair_summary = pd.DataFrame(pair_rows).set_index(["reference", "other"])
        print(pair_summary)
        pair_summary.to_csv(BASE / "traccuracy_runs_pairwise.csv")

    print(f"\nSaved: {BASE / 'traccuracy_runs_vs_gt.csv'}")
    if DO_PAIRWISE and len(RUNS) > 1:
        print(f"Saved: {BASE / 'traccuracy_runs_pairwise.csv'}")
