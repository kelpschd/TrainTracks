# spotiflow_track_compare.py
#
# Tracking-level comparison vs GT. Pass the run IDs (from tracking_log.csv)
# of runs made with candidate_graph.py on each blob CSV; scores each with
# traccuracy exactly as traccuracy_compare.py does (same matcher, same
# MATCH_THRESHOLD, same metrics) and writes one summary CSV.
#
# Workflow:
#   python scripts/candidate_graph.py blobs/blobs_0_00021.csv --note "LoG 0.00021"
#   python scripts/candidate_graph.py blobs/blobs_spotiflow_general_raw_p0_5.csv --note "SF general raw p0.5"
#   ...
#   python scripts/spotiflow_track_compare.py            # scores every run in the log
#   python scripts/spotiflow_track_compare.py --notes-contain SF LoG

import argparse
import json
from pathlib import Path

import pandas as pd

from funtracks.import_export import import_from_geff
from traccuracy_compare import (
    track_df_to_nx, geff_to_track_df, run_pair, GT_CSV, MATCH_THRESHOLD, BASE,
)

RUNS_DIR = BASE / "tracking_runs" / "runs"


def load_runs() -> pd.DataFrame:
    """One row per run, read from each run's metadata.json.

    (tracking_log.csv is NOT used: its header predates later-added columns, so
    newer rows have more fields than the header and pandas can't parse it.)
    """
    recs = []
    for meta in sorted(RUNS_DIR.glob("*/metadata.json")):
        try:
            recs.append(json.loads(meta.read_text()))
        except Exception as e:
            print(f"  skipping unreadable {meta}: {e}")
    return pd.DataFrame(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes-contain", nargs="*", default=None,
                    help="Only score runs whose note contains one of these")
    args = ap.parse_args()

    log = load_runs()
    log = log[log["output_geff"].notna()]
    if args.notes_contain:
        mask = log["note"].fillna("").apply(lambda s: any(k in s for k in args.notes_contain))
        log = log[mask]
    print(f"Scoring {len(log)} runs")

    gt_nx = track_df_to_nx(pd.read_csv(GT_CSV, encoding="utf-8-sig"))
    rows = []
    for _, r in log.iterrows():
        geff_path = Path(r["output_geff"])
        if not geff_path.exists():
            print(f"  missing: {geff_path}"); continue
        sg = import_from_geff(directory=str(geff_path),
                              node_name_map={"time": "t", "pos": ["y", "x"]})
        pred_df = geff_to_track_df(sg)
        flat = run_pair(gt_nx, track_df_to_nx(pred_df), "GT", "pred", MATCH_THRESHOLD)
        flat.update(label=r.get("note") or Path(r["input_csv"]).stem,
                    input_csv=Path(r["input_csv"]).name,
                    pred_tracks=pred_df["particle"].nunique(),
                    pred_detections=len(pred_df))
        rows.append(flat)
        print(f"  {flat['label']}: recall(node)={flat.get('Node Recall', float('nan')):.3f} "
              f"TE={flat.get('target_effectiveness', float('nan')):.3f} "
              f"tracks={flat['pred_tracks']}")

    out = pd.DataFrame(rows).set_index("label")
    keep = ["input_csv", "target_effectiveness", "track_fractions", "Node Recall",
            "Edge Recall", "Total GT Nodes", "True Positive Nodes",
            "pred_tracks", "pred_detections"]
    pd.set_option("display.width", None, "display.max_columns", None)
    print(out[[c for c in keep if c in out.columns]].round(3))
    dest = BASE / "traccuracy_spotiflow_vs_gt.csv"
    out.to_csv(dest)
    print(f"Saved {dest}")


if __name__ == "__main__":
    main()
