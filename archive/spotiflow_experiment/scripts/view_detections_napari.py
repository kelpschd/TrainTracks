# view_detections_napari.py
#
# Overlay detections on the raw movie to see WHERE LoG and Spotiflow differ
# and which GT points each one misses.
#
# Layers (toggle in napari's layer list):
#   raw                  channel 0 movie
#   bgsub                your preprocess_stack() output (hidden by default)
#   LoG ...              your filtered LoG CSVs (hollow circles)
#   SF filtered ...      Spotiflow after the dense-bin filter (hollow circles)
#   SF all (by prob)     UNFILTERED Spotiflow spots, colored by probability
#                        (viridis; hidden by default). Click-select a point and
#                        read its prob in the layer's feature table.
#   GT                   validated GT points (magenta squares), with particle id
#   GT missed by SF/LoG  GT points with no detection within TOL px (red X)
#
# Usage: python scripts/view_detections_napari.py
# Tip: step through frames with the slider; press the layer's eye icon to toggle.

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path

import numpy as np
import pandas as pd
import zarr
import napari

BASE = Path("/Users/kelpschdj/Documents/DataTecnica/TTU/TrainTracks")
INPUT_ZARR = Path(
    "/Users/kelpschdj/Documents/DataTecnica/TTU/Data/Sphere/"
    "220725_i11w-hT-M33-I76_sg1035_d10sphere/raw_data/sg100_Well5_1018.zarr"
)
CHANNEL = 0
SHOW_BGSUB = True           # also load the background-subtracted movie (slower start)
TOL = 5.0                   # px, same as compare_detections.py

# name -> (csv, edge color).  Missing files are skipped.
LAYERS = {
    "LoG 0.00021":           (BASE / "blobs/blobs_0_00021.csv", "yellow"),
    "SF filtered raw p0.3":  (BASE / "blobs/blobs_spotiflow_general_raw_p0_3.csv", "cyan"),
    "SF filtered raw p0.5":  (BASE / "blobs/blobs_spotiflow_general_raw_p0_5.csv", "lime"),
    "SF filtered bgsub p0.3": (BASE / "blobs/blobs_spotiflow_general_bgsub_p0_3.csv", "orange"),
}
SF_ALL_CSV = BASE / "blobs/spotiflow_all_general_raw.csv"
GT_CSV = BASE / "sg100_Well5_1018_valid_tracks.csv"
# Which detector to use for the "GT missed" layer:
MISSED_REFERENCE = {"LoG 0.00021": BASE / "blobs/blobs_0_00021.csv",
                    "SF filtered raw p0.3": BASE / "blobs/blobs_spotiflow_general_raw_p0_3.csv"}


def missed_gt(det_csv: Path, gt: pd.DataFrame) -> np.ndarray:
    det = pd.read_csv(det_csv)
    by_t = {t: g[["y", "x"]].to_numpy(float) for t, g in det.groupby("t")}
    out = []
    for _, r in gt.iterrows():
        d = by_t.get(r["frame"])
        if d is None or np.hypot(d[:, 0] - r["y"], d[:, 1] - r["x"]).min() > TOL:
            out.append([r["frame"], r["y"], r["x"]])
    return np.array(out).reshape(-1, 3)


viewer = napari.Viewer()
stack = np.asarray(zarr.open(str(INPUT_ZARR), mode="r")["s0"])      # (T,C,Y,X)
viewer.add_image(stack[:, CHANNEL], name="raw", blending="additive")

if SHOW_BGSUB:
    from preprocess_blob_det import preprocess_stack
    viewer.add_image(preprocess_stack(stack, blob_channel=CHANNEL), name="bgsub",
                     visible=False, blending="additive")

for name, (csv, color) in LAYERS.items():
    if not Path(csv).exists():
        print("missing, skipped:", csv)
        continue
    pts = pd.read_csv(csv)[["t", "y", "x"]].to_numpy()
    viewer.add_points(pts, name=f"{name} (n={len(pts)})", size=12,
                      face_color="transparent", border_color=color,
                      border_width=0.15, border_width_is_relative=True)

if SF_ALL_CSV.exists():
    sf = pd.read_csv(SF_ALL_CSV)
    viewer.add_points(sf[["t", "y", "x"]].to_numpy(), name="SF all (by prob)",
                      size=8, features={"prob": sf["prob"].to_numpy()},
                      face_color="prob", face_colormap="viridis",
                      border_color="transparent", visible=False)

gt = pd.read_csv(GT_CSV, encoding="utf-8-sig")
viewer.add_points(gt[["frame", "y", "x"]].to_numpy(), name="GT", size=20,
                  features={"particle": gt["particle"].to_numpy()},
                  face_color="transparent", border_color="magenta",
                  symbol="square", border_width=0.1, border_width_is_relative=True)

for label, csv in MISSED_REFERENCE.items():
    if Path(csv).exists():
        m = missed_gt(csv, gt)
        viewer.add_points(m, name=f"GT missed by {label} (n={len(m)})", size=26,
                          face_color="transparent", border_color="red", symbol="cross",
                          border_width=0.1, border_width_is_relative=True, visible=False)

napari.run()
