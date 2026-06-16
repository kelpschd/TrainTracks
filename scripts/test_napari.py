from pathlib import Path

import motile
import geff
import napari
import zarr
import pandas as pd
import numpy as np
from funtracks.import_export import import_from_geff
# from motile_tracker.motile.backend import MotileRun
# from motile_tracker.data_views.views.tree_view.tree_widget import TreeWidget
# from motile_tracker.data_views.views_coordinator.tracks_viewer import TracksViewer

# currently have a solution graph in GEFF and the raw image
# want to visualize solution graph tracks

if __name__ == "__main__":
    viewer = napari.Viewer()
    napari.run()