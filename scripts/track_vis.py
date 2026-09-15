"""
Visualize one or more tracking runs (from candidate_graph.py's run system)
against the raw image, optical flow, each run's blob detections, and the
manually validated GT tracks.

Runs are loaded by run_id from tracking_runs/runs/<run_id>/. Each run's
metadata.json tells us which blob CSV it used, so blobs are pulled per run
without re-specifying them here. Hardcoding run_ids keeps this reproducible:
the script records exactly which runs produced a given view.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import napari
import zarr
from funtracks.import_export import import_from_geff


# ---------------------------
# Config
# ---------------------------

BASE = Path("/Users/kelpschdj/Documents/DataTecnica/TTU/TrainTracks")
RUNS_DIR = BASE / "tracking_runs" / "runs"

IMAGES_DIR = Path(
    "/Users/kelpschdj/Documents/DataTecnica/TTU/Data/Sphere/"
    "220725_i11w-hT-M33-I76_sg1035_d10sphere/raw_data"
)
IMG_NUM = 9

GT_CSV = BASE / "sg100_Well5_1018_valid_tracks.csv"  # cols: frame, x, y, particle

# Runs to visualize: {label: run_id}. Fill in run_ids from tracking_log.csv.
RUNS = {
    "n12": "20260914_171731_4900",
    "n24": "20260914_172215_e116",
    "n36": "20260914_172659_2e3a",
    "n48": "20260914_173145_a485",
    "n60": "20260914_173629_c280",
    # "gap2_pen20":   "REPLACE_WITH_RUN_ID",
}

# Distinct colors per run (cycled if more runs than colors).
COLORS = ["cyan", "yellow", "magenta", "green", "orange"]


# ---------------------------
# Helpers
# ---------------------------

def geff_to_tracks_array(sg):
    """funtracks solution graph -> napari Tracks array [track_id, t, y, x]."""
    attrs = sg.graph.node_attrs(unpack=True).to_pandas()
    df = pd.DataFrame({
        "particle": attrs["tracklet_id"],
        "frame": attrs["t"].astype(int),
        "y": attrs["pos_0"].astype(float),
        "x": attrs["pos_1"].astype(float),
    }).sort_values(["particle", "frame"])
    return df[["particle", "frame", "y", "x"]].to_numpy()


def load_run(run_id):
    """Return (solution_graph, metadata_dict) for a run_id."""
    run_dir = RUNS_DIR / run_id
    with open(run_dir / "metadata.json") as f:
        meta = json.load(f)
    sg = import_from_geff(
        directory=str(run_dir / "tracks.geff"),
        node_name_map={"time": "t", "pos": ["y", "x"]},
    )
    return sg, meta


# ---------------------------
# Main
# ---------------------------

if __name__ == "__main__":
    # raw image
    images = sorted(IMAGES_DIR.glob("*.zarr"))
    np_arr = np.array(zarr.open(images[IMG_NUM], mode="r")["s0"])
    print("Raw image loaded!")

    # flow frames
    flow_frames = np.array(
        zarr.open_group(BASE / "flow.zarr", mode="r")["flow_frames_XY"]
    )
    print("Flow frames loaded!")

    # GT tracks
    gt_df = pd.read_csv(GT_CSV)
    print(f"GT loaded: {gt_df['particle'].nunique()} tracks, {len(gt_df)} detections")

    # runs (+ their blob CSVs, read from each run's metadata)
    runs = {}
    for label, run_id in RUNS.items():
        sg, meta = load_run(run_id)
        blobs = pd.read_csv(meta["input_csv"])
        runs[label] = dict(sg=sg, meta=meta, blobs=blobs)
        print(f"{label} ({run_id}): {len(sg.nodes())} nodes | "
              f"blobs: {Path(meta['input_csv']).name} | "
              f"max_gap={meta.get('max_gap')} gap_penalty={meta.get('gap_penalty')}")

    # ---------------------------
    # napari
    # ---------------------------
    viewer = napari.Viewer()
    viewer.add_image(np_arr[:, 0], name="Raw image")
    viewer.add_image(flow_frames, name="Flow")

    for i, (label, r) in enumerate(runs.items()):
        color = COLORS[i % len(COLORS)]
        viewer.add_points(
            r["blobs"], size=30, face_color="transparent",
            border_color=color, border_width=0.1, name=f"blobs_{label}",
        )
        viewer.add_tracks(geff_to_tracks_array(r["sg"]), name=f"tracks_{label}")

    viewer.add_tracks(
        gt_df[["particle", "frame", "y", "x"]].to_numpy(),
        name="manually_validated_tracks",
    )

    napari.run()