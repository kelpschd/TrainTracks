from pathlib import Path

import numpy as np
import pandas as pd
import napari
import zarr
from funtracks.import_export import import_from_geff
from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer


# ---------------------------
# geff -> track_df conversion
# ---------------------------

def geff_tracks_to_track_df(
    tracks,
    time_attr: str = "t",
    pos_attrs: tuple[str, ...] = ("pos_0", "pos_1"),
    track_attr: str = "tracklet_id",
) -> pd.DataFrame:
    """
    Convert a funtracks Tracks/SolutionTracks object into a DataFrame with
    columns [particle, frame, y, x].

    time_attr: with node_name_map={"time": "t", ...} at import, this ends up
        named "t" (not "time") -- the import code keeps the original GEFF
        property name for scalar attrs.
    pos_attrs: after node_attrs(unpack=True), a combined "pos" array attribute
        becomes "pos_0", "pos_1" (..."pos_2" for 3D), in the order passed as
        node_name_map["pos"] at import time (e.g. ["y","x"] -> pos_0=y, pos_1=x).
    track_attr: "tracklet_id" is funtracks' auto-computed linear-segment track
        identity, also what colors/segments the napari Tracks layer.
    """
    if len(pos_attrs) not in (2, 3):
        raise ValueError(f"Expected 2 (y,x) or 3 (z,y,x) pos_attrs, got {pos_attrs}")

    attrs_df = tracks.graph.node_attrs(unpack=True).to_pandas()

    missing = [c for c in (time_attr, track_attr, *pos_attrs) if c not in attrs_df.columns]
    if missing:
        raise KeyError(
            f"Expected columns {missing} not found in node_attrs(); "
            f"available columns: {list(attrs_df.columns)}. "
            "Run tracks.graph.node_attrs(unpack=True) yourself to check names "
            "if your node_name_map/axis order differs from the default."
        )

    out = pd.DataFrame(
        {
            "particle": attrs_df[track_attr],
            "frame": attrs_df[time_attr].astype(int),
            "y": attrs_df[pos_attrs[-2]].astype(float),
            "x": attrs_df[pos_attrs[-1]].astype(float),
        }
    )
    return out.sort_values(["particle", "frame"]).reset_index(drop=True)


def graph_to_tracks_array(solution_graph):
    """funtracks graph -> napari Tracks array [track_id, t, y, x]."""
    df = geff_tracks_to_track_df(solution_graph)
    return df[["particle", "frame", "y", "x"]].to_numpy()


# ---------------------------
# Paths / config
# ---------------------------

BASE = Path("/Users/kelpschdj/Documents/DataTecnica/TTU/TrainTracks")
IMAGES_DIR = Path(
    "/Users/kelpschdj/Documents/DataTecnica/TTU/Data/Sphere/"
    "220725_i11w-hT-M33-I76_sg1035_d10sphere/raw_data"
)
IMG_NUM = 9

# blob detection thresholds to compare, keyed by their filename suffix
THRESHOLDS = ["0_00015", "0_00017", "0_00019", "0_00021"]

# distinct border colors so layers are separable on the canvas
BLOB_COLORS = ["red", "cyan", "yellow", "magenta"]


if __name__ == "__main__":
    # raw image
    images = sorted(IMAGES_DIR.glob("*.zarr"))
    zarr_root = zarr.open(images[IMG_NUM], mode="r")
    np_arr = np.array(zarr_root["s0"])
    print("Raw image loaded!")

    # blobs per threshold
    blobs = {
        thr: pd.read_csv(BASE / "blobs" / f"blobs_{thr}.csv")
        for thr in THRESHOLDS
    }
    print("Annotated blobs loaded!")

    # flow frames
    flow_root = zarr.open_group(BASE / "flow.zarr", mode="r")
    flow_frames = flow_root["flow_frames_XY"]
    print("Flow frames loaded!")

    # solution graphs per threshold
    solution_graphs = {}
    for thr in THRESHOLDS:
        sg = import_from_geff(
            directory=str(BASE / "tracking_runs" / f"blobs_{thr}.geff"),
            node_name_map={"time": "t", "pos": ["y", "x"]},
        )
        solution_graphs[thr] = sg
        print(f"blobs_{thr}: {len(sg.nodes())} nodes")

    # TODO: load in the validated (ground truth) tracks here

    # ---------------------------
    # napari
    # ---------------------------
    viewer = napari.Viewer()
    tracks_viewer = TracksViewer.get_instance(viewer)

    viewer.add_image(np_arr[:, 0], name="Raw image")
    viewer.add_image(np.array(flow_frames), name="Flow")

    for thr, color in zip(THRESHOLDS, BLOB_COLORS):
        viewer.add_points(
            blobs[thr], size=30, face_color="transparent",
            border_color=color, border_width=0.1, name=f"blobs_{thr}",
        )

    for thr in THRESHOLDS:
        viewer.add_tracks(
            graph_to_tracks_array(solution_graphs[thr]), name=f"tracks_{thr}"
        )

    napari.run()