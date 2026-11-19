"""The faster candidate-graph builder must give exactly the same graph as the original.

``_reference_build_candidate_graph`` is a copy of the original implementation (a full-frame
boolean mask per region). The current one reads each region's pixels from ``regionprops.coords``.
Same nodes, centroids and flow attributes, bit for bit.
"""
import numpy as np
import pytest

pytest.importorskip("networkx")
pytest.importorskip("skimage")

import networkx as nx  # noqa: E402
import skimage.measure  # noqa: E402
from skimage.morphology import binary_dilation, disk  # noqa: E402

from TrainTracks.tracking import build_candidate_graph  # noqa: E402


def _reference_build_candidate_graph(blobs_np, flow_arr, dilation_radius, n_frames):
    Y, X = flow_arr.shape[1], flow_arr.shape[2]
    mask = np.zeros((n_frames, Y, X), dtype=np.uint16)
    mask[blobs_np[:, 0], blobs_np[:, 1], blobs_np[:, 2]] = True
    dilated = np.zeros(mask.shape, dtype=np.uint16)
    footprint = disk(dilation_radius)
    for frame in range(n_frames):
        dilated[frame] = binary_dilation(mask[frame], footprint=footprint)
    g = nx.DiGraph()
    last = 0
    for frame in range(n_frames):
        seg = skimage.measure.label(dilated[frame])
        props = skimage.measure.regionprops(seg)
        flow_frame = flow_arr[frame] if frame < flow_arr.shape[0] else None
        for rp in props:
            region = seg == rp.label
            centroid = (float(rp.centroid[0]), float(rp.centroid[1]))
            if flow_frame is None:
                fv = (0.0, 0.0)
            else:
                fv = (float(np.mean(flow_frame[region][..., 1])), float(np.mean(flow_frame[region][..., 0])))
            g.add_node(int(rp.label) + last, t=frame, x=centroid[1], y=centroid[0], flow=fv)
        last += len(props)
    return g


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_new_candidate_graph_equals_the_original(seed):
    rng = np.random.default_rng(seed)
    T, Y, X = 5, 160, 160
    flow = rng.normal(0, 3, (T - 1, Y, X, 2)).astype("float32")
    n = 150
    blobs = np.column_stack([rng.integers(0, T, n), rng.integers(5, Y - 5, n), rng.integers(5, X - 5, n)]).astype(np.int64)
    blobs = np.unique(blobs, axis=0)   # clusters of nearby blobs merge into one region, as in real data
    new = build_candidate_graph(blobs, flow, dilation_radius=5, n_frames=T)
    ref = _reference_build_candidate_graph(blobs, flow, 5, T)
    assert new.number_of_nodes() == ref.number_of_nodes() > 20
    assert dict(new.nodes(data=True)) == dict(ref.nodes(data=True))   # exact float equality
