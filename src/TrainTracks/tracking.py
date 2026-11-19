"""Candidate graph, motile solve and track bookkeeping.

Moved unchanged from ``scripts/batch_pipeline.py`` (the code that produced every existing
``tracks.geff``; parameters: ``TRACKING_PARAMS`` in the pipeline, stored in each movie's
``metadata.json`` as ``tracking_params``).

Steps: one candidate node per connected blob region after disk dilation (``dilation_radius``;
this merges puncta within about 2 x radius px, see TODO Tier 2), candidate edges to the next
``max_gap`` frames within ``max_edge_distance`` x gap px, edge cost from the distance to the
flow-predicted position, motile ILP with max 1 parent / 1 child. ``run_tracking`` chains them.

networkx / scipy / scikit-image / motile are imported lazily inside the functions.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from TrainTracks.ids import assign_sequential_track_ids


def build_candidate_graph(
    blobs_np: np.ndarray,
    flow_arr: np.ndarray,
    dilation_radius: int,
    n_frames: int,
) -> nx.DiGraph:
    """
    blobs (t,y,x) + flow (T-1,Y,X,2) -> candidate DiGraph.

    The final image frame has no outgoing flow by definition, so the final
    frame's node flow attribute is set to (0,0). This keeps all T image frames
    represented while preserving the correct flow convention: flow[t] is t->t+1.
    """
    import networkx as nx
    import skimage.measure
    from skimage.morphology import binary_dilation, disk

    if flow_arr.ndim != 4 or flow_arr.shape[-1] != 2:
        raise ValueError(f"Expected flow shape (T-1,Y,X,2); got {flow_arr.shape}")
    if n_frames != flow_arr.shape[0] + 1:
        raise ValueError(
            f"n_frames={n_frames} inconsistent with flow length {flow_arr.shape[0]}"
        )

    Y, X = flow_arr.shape[1], flow_arr.shape[2]
    mask = np.zeros((n_frames, Y, X), dtype=np.uint16)

    if len(blobs_np):
        if np.any(blobs_np[:, 0] < 0) or np.any(blobs_np[:, 0] >= n_frames):
            raise ValueError("Blob frame index out of bounds")
        if np.any(blobs_np[:, 1] < 0) or np.any(blobs_np[:, 1] >= Y):
            raise ValueError("Blob y index out of bounds")
        if np.any(blobs_np[:, 2] < 0) or np.any(blobs_np[:, 2] >= X):
            raise ValueError("Blob x index out of bounds")
        mask[blobs_np[:, 0], blobs_np[:, 1], blobs_np[:, 2]] = True

    dilated = np.zeros(mask.shape, dtype=np.uint16)
    footprint = disk(dilation_radius)
    for frame in range(n_frames):
        dilated[frame] = binary_dilation(mask[frame], footprint=footprint)

    cand_graph = nx.DiGraph()
    last_node_id = 0

    for frame in range(n_frames):
        segmentation = skimage.measure.label(dilated[frame])
        props = skimage.measure.regionprops(segmentation)

        # Only frames that can be a source for an outgoing edge have actual flow.
        flow_frame = flow_arr[frame] if frame < flow_arr.shape[0] else None

        for regionprop in props:
            node_id = int(regionprop.label) + last_node_id
            centroid = (
                float(regionprop.centroid[0]),
                float(regionprop.centroid[1]),
            )

            if flow_frame is None:
                flow_vec = (0.0, 0.0)
            else:
                # Read the region's own pixels (regionprops.coords is in row-major order, the same
                # order a full-frame boolean mask gives). Building a full 2048x2048 mask per region,
                # as this used to, made this step ~90% of the run time of a movie; the values are
                # bit-identical (tests/test_tracking_equivalence.py).
                rr, cc = regionprop.coords[:, 0], regionprop.coords[:, 1]
                region_flow = flow_frame[rr, cc]
                flow_vec = (
                    float(np.mean(region_flow[..., 1])),
                    float(np.mean(region_flow[..., 0])),
                )

            cand_graph.add_node(
                node_id,
                t=frame,
                x=centroid[1],
                y=centroid[0],
                flow=flow_vec,
            )

        # Advance the global offset only after this frame's nodes are created.
        last_node_id += len(props)

    if cand_graph.number_of_nodes() != len(cand_graph.nodes):
        raise RuntimeError("Candidate graph node count is internally inconsistent")

    return cand_graph


def _compute_node_frame_dict(cand_graph: nx.DiGraph) -> dict[int, list[int]]:
    frame_dict: dict[int, list[int]] = {}
    for node, data in cand_graph.nodes(data=True):
        frame_dict.setdefault(data["t"], []).append(node)
    return frame_dict


def _create_kdtree(cand_graph: nx.DiGraph, node_ids: list[int]) -> scipy.spatial.KDTree:
    import scipy.spatial

    positions = [[cand_graph.nodes[n]["x"], cand_graph.nodes[n]["y"]] for n in node_ids]
    return scipy.spatial.KDTree(positions)


def add_candidate_edges(
    cand_graph: nx.DiGraph,
    max_edge_distance: float,
    max_gap: int,
) -> None:
    node_frame_dict = _compute_node_frame_dict(cand_graph)
    frames = sorted(node_frame_dict.keys())
    if not frames:
        return

    kdtrees = {
        f: _create_kdtree(cand_graph, node_frame_dict[f])
        for f in frames
        if node_frame_dict[f]
    }

    for frame in frames:
        src_ids = node_frame_dict[frame]
        if not src_ids:
            continue
        src_tree = kdtrees[frame]

        for gap in range(1, max_gap + 1):
            target = frame + gap
            if target not in node_frame_dict:
                continue
            tgt_ids = node_frame_dict[target]
            if not tgt_ids:
                continue
            tgt_tree = kdtrees[target]
            matched = src_tree.query_ball_tree(
                tgt_tree, max_edge_distance * gap
            )
            for src_id, tgt_idxs in zip(src_ids, matched):
                for j in tgt_idxs:
                    cand_graph.add_edge(src_id, tgt_ids[j], gap=gap)


def add_flow_dist_attr(cand_graph: nx.DiGraph, gap_penalty: float) -> None:
    for u, v in cand_graph.edges:
        node_u = cand_graph.nodes[u]
        node_v = cand_graph.nodes[v]
        gap = cand_graph.edges[u, v].get("gap", 1)
        pos_u = np.array([node_u["y"], node_u["x"]])
        pos_v = np.array([node_v["y"], node_v["x"]])
        predicted = pos_u + np.array(node_u["flow"]) * gap
        base = float(np.linalg.norm(predicted - pos_v))
        cand_graph.edges[u, v]["flow_offset"] = (
            base + gap_penalty * (gap - 1)
        )


def solve_tracks(cand_graph: nx.DiGraph, params: dict) -> nx.DiGraph:
    import motile
    import motile.constraints
    import motile.costs
    import networkx as nx

    if cand_graph.number_of_nodes() == 0:
        return nx.DiGraph()

    cand_trackgraph = motile.TrackGraph(cand_graph, frame_attribute="t")
    add_flow_dist_attr(cand_trackgraph, params["gap_penalty"])

    solver = motile.Solver(cand_trackgraph)
    solver.add_cost(
        motile.costs.NodeSelectedCost(
            weight=params["node_selected_weight"]
        )
    )
    solver.add_cost(
        motile.costs.EdgeSelectedCost(
            weight=params["edge_selected_weight"],
            attribute=params["edge_flow_attribute"],
            constant=params["edge_selected_constant"],
        )
    )
    solver.add_cost(
        motile.costs.NodeAppearCost(
            constant=params["node_appear_constant"]
        )
    )
    solver.add_constraint(
        motile.constraints.MaxParents(params["max_parents"])
    )
    solver.add_constraint(
        motile.constraints.MaxChildren(params["max_children"])
    )
    solver.solve()
    return solver.get_selected_subgraph().to_nx_graph()


def annotate_solution_track_ids(solution_nx: nx.DiGraph) -> nx.DiGraph:
    """Persist the same canonical geometry-derived track ID on every node."""
    df = solution_to_track_df(solution_nx)
    if df.empty:
        return solution_nx
    # Match rows back to solution nodes by (t, y, x). Coordinates originate in
    # the solution graph, so this mapping is exact for a freshly solved graph.
    lookup = {}
    for pid, group in df.groupby("particle"):
        for row in group.itertuples(index=False):
            lookup[(int(row.frame), float(row.y), float(row.x))] = int(pid)
    for node, data in solution_nx.nodes(data=True):
        key = (int(data["t"]), float(data["y"]), float(data["x"]))
        data["track_id"] = lookup[key]
        data["tracklet_id"] = lookup[key]
    return solution_nx


def track_metrics(solution_nx: nx.DiGraph, long_min_frames: int) -> dict:
    import networkx as nx

    n_nodes = solution_nx.number_of_nodes()
    n_edges = solution_nx.number_of_edges()
    components = list(nx.weakly_connected_components(solution_nx))
    n_tracks = len(components)

    if n_tracks == 0:
        return {
            "n_detections_tracked": 0,
            "n_edges": 0,
            "n_tracks": 0,
            "mean_track_len": 0.0,
            "median_track_len": 0.0,
            "max_track_len": 0,
            "n_long_tracks": 0,
            "frac_nodes_in_long": 0.0,
            "tracks_per_detection": 0.0,
            "n_skip_edges": 0,
            "frac_skip_edges": 0.0,
        }

    lengths = []
    nodes_in_long = 0
    for comp in components:
        frames = [solution_nx.nodes[n]["t"] for n in comp]
        span = max(frames) - min(frames) + 1
        lengths.append(span)
        if span >= long_min_frames:
            nodes_in_long += len(comp)

    lengths = np.asarray(lengths)
    n_skip = sum(
        1
        for u, v in solution_nx.edges
        if abs(solution_nx.nodes[v]["t"] - solution_nx.nodes[u]["t"]) > 1
    )

    return {
        "n_detections_tracked": int(n_nodes),
        "n_edges": int(n_edges),
        "n_tracks": int(n_tracks),
        "mean_track_len": float(np.mean(lengths)),
        "median_track_len": float(np.median(lengths)),
        "max_track_len": int(np.max(lengths)),
        "n_long_tracks": int(np.sum(lengths >= long_min_frames)),
        "frac_nodes_in_long": float(nodes_in_long / n_nodes),
        "tracks_per_detection": float(n_tracks / n_nodes),
        "n_skip_edges": int(n_skip),
        "frac_skip_edges": float(n_skip / n_edges) if n_edges else 0.0,
    }


def solution_to_track_df(solution_nx: nx.DiGraph) -> pd.DataFrame:
    """Convert solved NetworkX graph to rows with stable track IDs."""
    import networkx as nx

    if solution_nx.number_of_nodes() == 0:
        return pd.DataFrame(columns=["particle", "frame", "y", "x"])

    rows = []
    for old_pid, comp in enumerate(nx.weakly_connected_components(solution_nx)):
        for node in comp:
            d = solution_nx.nodes[node]
            rows.append({
                "particle": old_pid,
                "frame": int(d["t"]),
                "y": float(d["y"]),
                "x": float(d["x"]),
            })
    return assign_sequential_track_ids(
        pd.DataFrame(rows).sort_values(["particle", "frame"]).reset_index(drop=True)
    )


def run_tracking(blobs_df: pd.DataFrame, flow_raw: np.ndarray, n_frames: int, params: dict):
    """Blobs (t, y, x) + flow (T-1, Y, X, 2) -> solved track graph.

    The sequence ``process_movie`` ran inline. ``params`` is ``TRACKING_PARAMS``.
    Returns ``(solution_nx, candidate_stats)``; the stats include the seconds spent in the graph,
    edge and solve steps (``seconds_graph`` / ``seconds_edges`` / ``seconds_solve``).
    """
    t0 = time.perf_counter()
    blobs_np = blobs_df[["t", "y", "x"]].to_numpy(dtype=np.int64)
    cand = build_candidate_graph(
        blobs_np,
        flow_raw,
        dilation_radius=params["dilation_radius"],
        n_frames=n_frames,
    )
    t1 = time.perf_counter()
    add_candidate_edges(
        cand,
        max_edge_distance=params["max_edge_distance"],
        max_gap=params["max_gap"],
    )
    t2 = time.perf_counter()
    stats = {"n_candidate_nodes": cand.number_of_nodes(), "n_candidate_edges": cand.number_of_edges()}
    solution = solve_tracks(cand, params)
    t3 = time.perf_counter()
    stats.update(
        seconds_graph=round(t1 - t0, 2), seconds_edges=round(t2 - t1, 2), seconds_solve=round(t3 - t2, 2)
    )
    return solution, stats
