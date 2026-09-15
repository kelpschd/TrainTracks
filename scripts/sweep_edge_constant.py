"""
Sweep edge_selected_constant over several values on ONE blob CSV, producing one
logged run per value via candidate_graph.run(). Each run is isolated (own run_id
+ dir + metadata), so nothing overwrites anything.

After it finishes it prints the run_ids and the two fragmentation readouts
(mean_track_len, tracks_per_detection) side by side, and a ready-to-paste RUNS
dict for track_vis.py.

Usage:
    python sweep_edge_constant.py /path/to/blobs_0_00017.csv
"""

import copy
import argparse
from pathlib import Path

import candidate_graph as cg


# Values to sweep (both directions around the original -36).
EDGE_CONSTANTS = [-12.0, -24.0, -36.0, -48.0, -60.0]


def run_one(blobs_csv: Path, edge_constant: float) -> dict:
    """Run candidate_graph.run() with edge_selected_constant overridden.

    We temporarily swap the module-level FIXED_PARAMS so run() picks up the
    override, then restore it -- keeps each run independent and correctly logged.
    """
    original = cg.FIXED_PARAMS
    patched = copy.deepcopy(original)
    patched["edge_selected_constant"] = edge_constant
    cg.FIXED_PARAMS = patched
    try:
        note = f"edge_const sweep: {edge_constant}"
        return cg.run(blobs_csv, note=note)
    finally:
        cg.FIXED_PARAMS = original  # always restore, even if run() raises


def main():
    parser = argparse.ArgumentParser(description="Sweep edge_selected_constant.")
    parser.add_argument("blobs_csv", type=Path, help="Path to a blob CSV (t, y, x)")
    args = parser.parse_args()

    if not args.blobs_csv.exists():
        parser.error(f"Blob CSV not found: {args.blobs_csv}")

    results = []
    for ec in EDGE_CONSTANTS:
        print("\n" + "=" * 70)
        print(f"RUN: edge_selected_constant = {ec}")
        print("=" * 70)
        rec = run_one(args.blobs_csv, ec)
        results.append(rec)

    # -----------------------------------------------------------------
    # Summary: the fragmentation-vs-constant curve
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY (read mean_track_len UP / tracks_per_detection DOWN")
    print("as less fragmentation -- but confirm correctness later with traccuracy)")
    print("=" * 70)
    print(f"{'edge_const':>11} | {'mean_len':>8} | {'median':>6} | "
          f"{'max':>4} | {'n_tracks':>8} | {'tracks/det':>10} | run_id")
    print("-" * 90)
    for rec in results:
        print(f"{rec['edge_selected_constant']:>11} | "
              f"{rec['mean_track_len']:>8.3f} | "
              f"{rec['median_track_len']:>6.1f} | "
              f"{rec['max_track_len']:>4} | "
              f"{rec['n_tracks']:>8} | "
              f"{rec['tracks_per_detection']:>10.4f} | "
              f"{rec['run_id']}")

    # -----------------------------------------------------------------
    # Ready-to-paste RUNS dict for track_vis.py
    # -----------------------------------------------------------------
    print("\nPaste into track_vis.py:")
    print("RUNS = {")
    for rec in results:
        label = f"ec{int(rec['edge_selected_constant'])}"
        print(f'    "{label}": "{rec["run_id"]}",')
    print("}")


if __name__ == "__main__":
    main()
