"""
Sweep gap_penalty at a FIXED edge_selected_constant, to tune gap-closing (skip
edges) in a controlled way. One logged run per penalty value.

Base edge constant is held at a value where skips are ~off at high penalty, so
LOWERING the penalty unlocks skip edges gradually and you can attribute skip
selection to the penalty (not to the edge constant forcing them).

The summary highlights n_skip_edges and median_track_len -- median moving off
its floor is the signal that gap-closing is actually stitching tracks, not just
growing the tail.

Usage:
    python sweep_gap_penalty.py /path/to/blobs_0_00017.csv
"""

import copy
import argparse
from pathlib import Path

import candidate_graph as cg


# Fixed edge constant for this sweep (chosen: skips ~off at high penalty here,
# so the penalty controls skip selection). -60 sat right at the skip threshold.
EDGE_CONSTANT = -72.0

# Penalty values, high -> low. High vetoes skips; low unlocks them.
GAP_PENALTIES = [20.0, 10.0, 5.0, 2.0, 0.0]

# Make sure gaps are actually offered in the candidate graph.
MAX_GAP = 2


def run_one(blobs_csv: Path, gap_penalty: float) -> dict:
    """run() with edge_selected_constant + gap_penalty + max_gap overridden."""
    original = cg.FIXED_PARAMS
    patched = copy.deepcopy(original)
    patched["edge_selected_constant"] = EDGE_CONSTANT
    patched["gap_penalty"] = gap_penalty
    patched["max_gap"] = MAX_GAP
    cg.FIXED_PARAMS = patched
    try:
        note = f"gap_penalty sweep: ec{EDGE_CONSTANT} pen{gap_penalty} maxgap{MAX_GAP}"
        return cg.run(blobs_csv, note=note)
    finally:
        cg.FIXED_PARAMS = original


def main():
    parser = argparse.ArgumentParser(description="Sweep gap_penalty at fixed edge const.")
    parser.add_argument("blobs_csv", type=Path, help="Path to a blob CSV (t, y, x)")
    args = parser.parse_args()
    if not args.blobs_csv.exists():
        parser.error(f"Blob CSV not found: {args.blobs_csv}")

    print(f"Edge constant fixed at {EDGE_CONSTANT}, max_gap={MAX_GAP}")

    results = []
    for pen in GAP_PENALTIES:
        print("\n" + "=" * 70)
        print(f"RUN: gap_penalty = {pen}  (edge_const {EDGE_CONSTANT})")
        print("=" * 70)
        results.append(run_one(args.blobs_csv, pen))

    # -----------------------------------------------------------------
    # Summary: does lowering the penalty unlock skips + lengthen tracks?
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"GAP_PENALTY SWEEP SUMMARY (edge_const {EDGE_CONSTANT})")
    print("watch: skip_edges rise + median_len move off floor = gap-closing works")
    print("(then confirm CORRECTNESS with traccuracy -- longer != correct)")
    print("=" * 70)
    print(f"{'gap_pen':>8} | {'skip_edges':>10} | {'frac_skip':>9} | "
          f"{'mean_len':>8} | {'median':>6} | {'max':>4} | "
          f"{'long>=10':>8} | {'tracks/det':>10} | run_id")
    print("-" * 110)
    for rec in results:
        print(f"{rec['gap_penalty']:>8} | "
              f"{rec['n_skip_edges']:>10} | "
              f"{rec['frac_skip_edges']:>9.4f} | "
              f"{rec['mean_track_len']:>8.3f} | "
              f"{rec['median_track_len']:>6.1f} | "
              f"{rec['max_track_len']:>4} | "
              f"{rec['n_long_tracks']:>8} | "
              f"{rec['tracks_per_detection']:>10.4f} | "
              f"{rec['run_id']}")

    # Ready-to-paste RUNS dict for track_vis / traccuracy_compare
    print("\nPaste into track_vis.py / traccuracy_compare.py:")
    print("RUNS = {")
    for rec in results:
        label = f"pen{rec['gap_penalty']:g}"
        print(f'    "{label}": "{rec["run_id"]}",')
    print("}")


if __name__ == "__main__":
    main()
