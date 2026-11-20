# view_tracks_napari.py
#
# View the solved tracks from several runs (LoG vs Spotiflow) plus the GT
# tracks on the raw movie, in one napari session.
#
# Runs are found by the note passed to candidate_graph.py (--note), read from
# tracking_runs/runs/*/metadata.json. If several runs share a note, the most
# recent one is used.
#
# Layers per run:
#   <label> tracks >=N fr   tracks with at least MIN_LEN frames (visible)
#   <label> ALL tracks      every track, incl. 1-2 frame junk (hidden; toggle on)
# plus:
#   GT tracks               your 11 validated tracks (thick, magenta-ish)
#
# Track IDs are re-numbered 0..N-1 here (small ints), so the float64 / 63-bit
# funtracks-ID problem in your TODO doesn't affect this viewer. Original IDs
# are kept in the layer's features as `orig_id`.
#
# Usage: python scripts/view_tracks_napari.py

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
import napari

from funtracks.import_export import import_from_geff
from traccuracy_compare import geff_to_track_df

BASE = Path("/Users/kelpschdj/Documents/DataTecnica/TTU/TrainTracks")
RUNS_DIR = BASE / "tracking_runs" / "runs"
GT_CSV = BASE / "sg100_Well5_1018_valid_tracks.csv"
INPUT_ZARR = Path(
    "/Users/kelpschdj/Documents/DataTecnica/TTU/Data/Sphere/"
    "220725_i11w-hT-M33-I76_sg1035_d10sphere/raw_data/sg100_Well5_1018.zarr"
)
CHANNEL = 0
MIN_LEN = 6            # frames; "long" track threshold for the default-visible layer
TAIL = 40              # tail length in frames (>= movie length shows full track)

# label -> note used in candidate_graph.py --note
RUNS = {
    "LoG 0.00021":         "LoG 0.00021",
    "SF raw p0.3 bf10":    "SF raw p0.3 bf10",
    "SF bgsub p0.3 bf10":  "SF bgsub p0.3 bf10",
}


def latest_run_geff(note: str):
    best = None
    for meta in RUNS_DIR.glob("*/metadata.json"):
        m = json.loads(meta.read_text())
        if m.get("note") == note and (best is None or m["timestamp"] > best["timestamp"]):
            best = m
    return Path(best["output_geff"]) if best else None


def to_tracks_array(df: pd.DataFrame):
    """df(particle, frame, y, x) -> napari (N,4) [id,t,y,x] + features."""
    df = df.copy()
    df["orig_id"] = df["particle"]
    df["tid"] = pd.factorize(df["particle"])[0]
    df = df.sort_values(["tid", "frame"]).reset_index(drop=True)
    data = df[["tid", "frame", "y", "x"]].to_numpy(float)
    return data, df


viewer = napari.Viewer()
stack = np.asarray(zarr.open(str(INPUT_ZARR), mode="r")["s0"])
viewer.add_image(stack[:, CHANNEL], name="raw", blending="additive")

for label, note in RUNS.items():
    geff_path = latest_run_geff(note)
    if geff_path is None or not geff_path.exists():
        print(f"no run found for note {note!r}; skipped")
        continue
    sg = import_from_geff(directory=str(geff_path),
                          node_name_map={"time": "t", "pos": ["y", "x"]})
    df = geff_to_track_df(sg)
    data, d = to_tracks_array(df)
    lens = d.groupby("tid")["frame"].transform("size").to_numpy()
    n_all = d["tid"].nunique()
    n_long = d.loc[lens >= MIN_LEN, "tid"].nunique()
    print(f"{label}: {n_all} tracks, {n_long} with >= {MIN_LEN} frames, "
          f"{len(d)} detections")

    long_mask = lens >= MIN_LEN
    viewer.add_tracks(data[long_mask], name=f"{label} tracks >={MIN_LEN} fr ({n_long})",
                      tail_length=TAIL, tail_width=3, colormap="turbo",
                      features={"orig_id": d.loc[long_mask, "orig_id"].to_numpy()})
    viewer.add_tracks(data, name=f"{label} ALL tracks ({n_all})",
                      tail_length=TAIL, tail_width=2, colormap="turbo",
                      features={"orig_id": d["orig_id"].to_numpy()}, visible=False)

gt = pd.read_csv(GT_CSV, encoding="utf-8-sig")
gt_data, gt_d = to_tracks_array(gt[["particle", "frame", "y", "x"]])
viewer.add_tracks(gt_data, name=f"GT tracks ({gt['particle'].nunique()})",
                  tail_length=TAIL, tail_width=6, colormap="magma",
                  features={"orig_id": gt_d["orig_id"].to_numpy()})

napari.run()
