import napari
import zarr
import scipy
import skimage

import numpy as np
import pandas as pd
import networkx as nx

from pathlib import Path
from tqdm.auto import tqdm
from typing import Iterable, Any
from skimage.morphology import disk, binary_dilation
from geff import write

# Load in detected spots
# csv -> numpy array
blobs_fp = "/mnt/efs/dl_jrc/student_data/S-DK/Sphere/220725_i11w-hT-M33-I76_sg1035_d10sphere/sg100_Well5_1018_blobs.csv"
blobs_df = pd.read_csv(blobs_fp)
print(blobs_df)
blobs_np = blobs_df.to_numpy().astype(np.uint16)
print(blobs_np)
print("Annotated blobs loaded!")

# import flow zarr
flow_path = Path("/home/S-DK/TrainTracks/flow.zarr")
flow_root = zarr.open_group(flow_path, mode='r')
flow_raw = flow_root['flow_raw'] 
flow_arr = np.array(flow_raw)
print(flow_arr.dtype)
print(flow_arr.shape)
print(type(flow_arr))

mask = np.zeros(flow_raw.shape[:-1], dtype = np.uint16)
mask[blobs_np[:, 0], blobs_np[:, 1], blobs_np[:, 2]] = True

radius = 5 # will consider making this bigger to get a nicer average flow
dilated = np.zeros(mask.shape, dtype = np.uint16)
for frame in range(0, mask.shape[0]):
    dilated[frame] = binary_dilation(mask[frame], footprint = disk(radius))

# Check to see if nodes and blobs align
# import raw img for reference
images_dir = Path("/mnt/efs/dl_jrc/student_data/S-DK/Sphere/220725_i11w-hT-M33-I76_sg1035_d10sphere")
images = list(sorted(images_dir.glob('*.zarr')))
img_num = 9
zarr_root = zarr.open(images[img_num], mode = 'r')
arr = zarr_root['s0']
np_arr = np.array(arr)
print("Raw image loaded!")

# set up cand_graph to be viewed in napari - remember my images are TYX 
# points_array = np.array([[data["t"], data["y"], data["x"]] for node, data in cand_graph.nodes(data=True)])
# cand_points_layer = napari.layers.Points(data=points_array, name="cand_points")

# Generate cand_graph 
# Generate it with flow attached
cand_graph = nx.DiGraph()
last_node_id = 0

for frame in range(0, dilated.shape[0]): 
    segmentation = skimage.measure.label(dilated[frame])
    flow_frame = flow_arr[frame]
    props = skimage.measure.regionprops(segmentation)
    last_node_id += len(props)
    
    for regionprop in props:
        node_id = int(regionprop.label)
        region = segmentation[frame] == node_id 
        centroid = (float(regionprop.centroid[0]),
                    float(regionprop.centroid[1]))
        flow = (float(np.mean(flow_frame[region][:1])),
                float(np.mean(flow_frame[region][:0])))
        attrs = {
            "t": frame,
            "x": centroid[1],
            "y": centroid[0],
            "flow": flow,
        }

        node_id += int(last_node_id)
        print(node_id)
        cand_graph.add_node(node_id, **attrs)
print(cand_graph)


quit()

for idx, row in tqdm(blobs_df.iterrows(), total=len(blobs_df), desc="Building candidate graph"):
    flow = calculate_flow_offset_pseudomask(row, flow_raw, 5)
    # take flow zarr and slice for each time point (frame) basically just bring in the whole row to the function
    # flow = # function goes here
    attrs = {
        "t": int(row["frame"]),
        "x": row["x"],
        "y": row["y"],
        "flow": flow,
    }
    cand_graph.add_node(idx, **attrs)
print(cand_graph)
# nx.write_graphml(cand_graph, 'cand_graph.graphml') # doesn't like the flow tuple...
# jen says we can write this to geff
write(
    cand_graph,
    "cand_graph.geff",
    zarr_format=2  # 2 or 3
)

# set up cand_graph to be viewed in napari - remember my images are TYX 
points_array = np.array([[data["t"], data["y"], data["x"]] for node, data in cand_graph.nodes(data=True)])
cand_points_layer = napari.layers.Points(data=points_array, name="cand_points")

# check everything in napari - candidate nodes should be in the center of the detected blobs (red circles)
viewer = napari.Viewer()
viewer.add_image(np_arr[:,0], name = "Raw image")
viewer.add_points(blobs_df, size = 30, face_color = "transparent", border_color="red", border_width=0.1)
viewer.add_labels(dilated)
viewer.add_layer(cand_points_layer)
napari.run()

# Add edges
def _compute_node_frame_dict(cand_graph: nx.DiGraph) -> dict[int, list[Any]]:
    """Compute dictionary from time frames to node ids for candidate graph.

    Args:
        cand_graph (nx.DiGraph): A networkx graph

    Returns:
        dict[int, list[Any]]: A mapping from time frames to lists of node ids.
    """
    node_frame_dict: dict[int, list[Any]] = {}
    for node, data in cand_graph.nodes(data=True):
        t = data["t"]
        if t not in node_frame_dict:
            node_frame_dict[t] = []
        node_frame_dict[t].append(node)
    return node_frame_dict

def create_kdtree(cand_graph: nx.DiGraph, node_ids: Iterable[Any]) -> scipy.spatial.KDTree:
    positions = [[cand_graph.nodes[node]["x"], cand_graph.nodes[node]["y"]] for node in node_ids]
    return scipy.spatial.KDTree(positions)

def add_cand_edges(
    cand_graph: nx.DiGraph,
    max_edge_distance: float,
) -> None:
    """Add candidate edges to a candidate graph by connecting all nodes in adjacent
    frames that are closer than max_edge_distance. Also adds attributes to the edges.

    Args:
        cand_graph (nx.DiGraph): Candidate graph with only nodes populated. Will
            be modified in-place to add edges.
        max_edge_distance (float): Maximum distance that objects can travel between
            frames. All nodes within this distance in adjacent frames will by connected
            with a candidate edge.
    """
    print("Extracting candidate edges")
    node_frame_dict = _compute_node_frame_dict(cand_graph)

    frames = sorted(node_frame_dict.keys())
    prev_node_ids = node_frame_dict[frames[0]]
    prev_kdtree = create_kdtree(cand_graph, prev_node_ids)
    for frame in tqdm(frames):
        #print(f"Current frame: {frame+1}")
        if frame + 1 not in node_frame_dict:
            continue
        next_node_ids = node_frame_dict[frame + 1]
        # print(f"Length of next node ids: {len(next_node_ids)}")
        next_kdtree = create_kdtree(cand_graph, next_node_ids)

        matched_indices = prev_kdtree.query_ball_tree(next_kdtree, max_edge_distance)

        for prev_node_id, next_node_indices in zip(prev_node_ids, matched_indices):
            for next_node_index in next_node_indices:
                next_node_id = next_node_ids[next_node_index]
                # print(f"Candidate edge: [{prev_node_id}, {next_node_id}]")
                cand_graph.add_edge(prev_node_id, next_node_id)

        prev_node_ids = next_node_ids
        prev_kdtree = next_kdtree

print("Finding candidate edges")
add_cand_edges(cand_graph, max_edge_distance=30)
print(f"Our candidate graph has {cand_graph.number_of_nodes()} nodes and {cand_graph.number_of_edges()} edges")
# this is much fewer edges than we really expected but it might be 
# related to how much flashing I see in blob detection, we may try
# to add skip connections - will have to see if there is code avaialbe
# for that already