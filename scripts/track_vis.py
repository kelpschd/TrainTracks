from pathlib import Path

# import motile
# import geff
import napari
import zarr
import pandas as pd
import numpy as np
from funtracks.import_export import import_from_geff
# from motile_tracker.motile.backend import MotileRun
from motile_tracker.data_views.views.tree_view.tree_widget import TreeWidget
from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer

# currently have a solution graph in GEFF and the raw image
# want to visualize solution graph tracks

if __name__ == "__main__":
    images_dir = Path("/mnt/efs/dl_jrc/student_data/S-DK/Sphere/220725_i11w-hT-M33-I76_sg1035_d10sphere")
    images = list(sorted(images_dir.glob('*.zarr')))
    img_num = 9
    zarr_root = zarr.open(images[img_num], mode = 'r')
    arr = zarr_root['s0']
    np_arr = np.array(arr)
    print("Raw image loaded!")

    blobs_fp = "/mnt/efs/dl_jrc/student_data/S-DK/Sphere/220725_i11w-hT-M33-I76_sg1035_d10sphere/sg100_Well5_1018_blobs.csv"
    blobs_df = pd.read_csv(blobs_fp)
    print(blobs_df)
    blobs_np = blobs_df.to_numpy().astype(np.uint16)
    print(blobs_np)
    print("Annotated blobs loaded!")

    # load in solution_graph.geff
    solution_graph = import_from_geff(
        directory="/home/S-DK/TrainTracks/solution_graph.geff",
        node_name_map={"time": "t", "pos": ["y", "x"]})

    # check everything in napari
    viewer = napari.Viewer()

    widget = TreeWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Lineage View", area="right")
    tracks_viewer = TracksViewer.get_instance(viewer)

    viewer.add_image(np_arr[:,0], name = "Raw image")
    viewer.add_points(blobs_df, size = 30, face_color = "transparent", border_color="red", border_width=0.1)
    tracks_viewer.tracks_list.add_tracks(solution_graph, "test_data_1")
    napari.run()