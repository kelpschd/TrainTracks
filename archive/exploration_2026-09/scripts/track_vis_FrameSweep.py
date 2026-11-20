"""
Visualize one or more tracking runs (from candidate_graph.py's run system)
against the raw image, optical flow, each run's blob detections, and the
manually validated GT tracks.

Also previews several minimum-track-SPAN cutoffs as separate layers, so you can
pick a short-track filter by eye. Span = (max frame - min frame + 1) per track,
NOT node count -- a real track that bridges a gap via skip edges has few nodes
but a long span, and filtering on node count would wrongly delete it.

Runs are loaded by run_id from tracking_runs/runs/<run_id>/.
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

# Run to preview span cutoffs on (one run keeps the viewer readable).
RUN_ID = "20260916_105301_9452"   # -72 / pen10

# Minimum span (frames) a track must cover to be KEPT, one layer each.
# A track survives cutoff C if (max_frame - min_frame + 1) >= C.
SPAN_CUTOFFS = [0, 5, 7, 9, 11]

# Colors for the cutoff layers (cycled).
CUTOFF_COLORS = ["gray", "orange", "yellow", "cyan", "green", "magenta"]


# ---------------------------
# Helpers
# ---------------------------

def geff_to_track_df(sg):
    """funtracks solution graph -> DataFrame [particle, frame, y, x]."""
    attrs = sg.graph.node_attrs(unpack=True).to_pandas()
    return pd.DataFrame({
        "particle": attrs["tracklet_id"],
        "frame": attrs["t"].astype(int),
        "y": attrs["pos_0"].astype(float),
        "x": attrs["pos_1"].astype(float),
    }).sort_values(["particle", "frame"]).reset_index(drop=True)


def track_spans(df):
    """particle -> span (max frame - min frame + 1)."""
    g = df.groupby("particle")["frame"]
    return (g.max() - g.min() + 1)


def filter_by_span(df, min_span):
    """Keep only tracks whose frame-span >= min_span."""
    spans = track_spans(df)
    keep = spans[spans >= min_span].index
    return df[df["particle"].isin(keep)]


def df_to_tracks_array(df):
    """DataFrame -> napari Tracks array [track_id, t, y, x]."""
    return df[["particle", "frame", "y", "x"]].to_numpy()


def load_run(run_id):
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

    # run + its blobs
    sg, meta = load_run(RUN_ID)
    blobs = pd.read_csv(meta["input_csv"])
    track_df = geff_to_track_df(sg)
    print(f"Run {RUN_ID}: {track_df['particle'].nunique()} tracks | "
          f"edge_const={meta.get('edge_selected_constant')} "
          f"gap_penalty={meta.get('gap_penalty')} "
          f"blobs={Path(meta['input_csv']).name}")

    # report how many tracks survive each cutoff (quick text preview)
    spans = track_spans(track_df)
    total = len(spans)
    print("\nSpan cutoff preview (tracks kept / removed):")
    for c in SPAN_CUTOFFS:
        kept = int((spans >= c).sum())
        print(f"  span >= {c}: keep {kept:5d} / remove {total - kept:5d} "
              f"({kept/total:.1%} kept)")
    print()

    # ---------------------------
    # napari
    # ---------------------------
    viewer = napari.Viewer()
    viewer.add_image(np_arr[:, 0], name="Raw image")
    viewer.add_image(flow_frames, name="Flow")

    viewer.add_points(
        blobs, size=30, face_color="transparent",
        border_color="red", border_width=0.1, name="blobs",
    )

    # one Tracks layer per cutoff -- toggle them in the layer list to compare.
    # start with all but the first hidden so the viewer isn't a tangle.
    for i, c in enumerate(SPAN_CUTOFFS):
        fdf = filter_by_span(track_df, c)
        color = CUTOFF_COLORS[i % len(CUTOFF_COLORS)]
        layer = viewer.add_tracks(
            df_to_tracks_array(fdf),
            name=f"tracks_span>={c}",
        )
        layer.visible = (i == 0)  # show only the first by default

    # GT on top, always visible
    viewer.add_tracks(
        gt_df[["particle", "frame", "y", "x"]].to_numpy(),
        name="manually_validated_tracks",
    )

    napari.run()