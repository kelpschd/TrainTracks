"""Dense optical flow (Farneback) for a (T, C, Y, X) movie.

Moved unchanged from ``scripts/batch_pipeline.py``; this is the function that produced
every existing ``flow.zarr`` (parameters: ``FLOW_PARAMS`` in the pipeline, stored in each
movie's ``metadata.json`` as ``flow_params``). Output has T-1 frames: ``flow[t]`` is the
displacement field for frame t -> t+1, last axis (dx, dy) as returned by OpenCV.
The detection channel is hard-coded to channel 0 (see TODO Tier 4).

cv2 / scikit-image / zarr are imported lazily so the module imports cheaply.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def compute_optical_flow(img_stack: np.ndarray, params: dict) -> np.ndarray:
    """
    Compute flow with shape (T-1, Y, X, 2), where flow[t] is frame t -> t+1.
    """
    import cv2
    from skimage.exposure import equalize_adapthist

    if img_stack.ndim != 4:
        raise ValueError(f"Expected (T,C,Y,X); got {img_stack.shape}")
    if img_stack.shape[0] < 2:
        raise ValueError("Need at least 2 frames to compute optical flow")

    gray = img_stack[:, 0]
    min_int_val = gray.min()
    max_int_val = gray.max()
    denom = max_int_val - min_int_val
    if denom == 0:
        arr_norm = np.zeros_like(gray, dtype=np.uint8)
    else:
        arr_norm = (
            (gray - min_int_val) / denom * 255
        ).astype(np.uint8)

    kernel_size = list(params["kernel_size"])
    clip_limit = params["clip_limit"]
    prev = equalize_adapthist(
        arr_norm[0], kernel_size=kernel_size, clip_limit=clip_limit
    )
    prev = (prev * 255).astype(np.uint8)

    T, Y, X = arr_norm.shape
    flow = np.zeros((T - 1, Y, X, 2), dtype=np.float32)
    prev_flow = None

    for i in range(1, T):
        curr = equalize_adapthist(
            arr_norm[i], kernel_size=kernel_size, clip_limit=clip_limit
        )
        curr = (curr * 255).astype(np.uint8)

        current_flow = cv2.calcOpticalFlowFarneback(
            prev,
            curr,
            None,
            pyr_scale=params["pyr_scale"],
            levels=params["levels"],
            winsize=params["winsize"],
            iterations=params["iterations"],
            poly_n=params["poly_n"],
            poly_sigma=params["poly_sigma"],
            flags=params["flags"],
        )

        sigma = params["temporal_smoothing_sigma"]
        if prev_flow is not None:
            current_flow = sigma * current_flow + (1.0 - sigma) * prev_flow

        flow[i - 1] = current_flow.astype(np.float32)
        prev = curr
        prev_flow = current_flow

    return flow


def write_flow_zarr(flow: np.ndarray, path: Path) -> None:
    import zarr

    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(path, mode="w")
    arr = root.create_array(
        "flow_raw",
        shape=flow.shape,
        dtype=np.float32,
        chunks=(1, flow.shape[1], flow.shape[2], 2),
    )
    arr[:] = flow
