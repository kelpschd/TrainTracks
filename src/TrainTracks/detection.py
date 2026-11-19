"""Blob detection: background subtraction, LoG detection, dense-blob filter.

These are the functions ``scripts/batch_pipeline.py`` used to produce every existing
``blobs.csv`` (moved here unchanged; the parameters live in ``DETECTION_PARAMS`` of the
pipeline and are recorded in each movie's ``metadata.json`` as ``detection_params``).
Heavy dependencies (scipy, scikit-image, dask) are imported lazily so the module can be
imported cheaply.

``filter_dense_blobs`` is intentional and stays (decided 2026-10-26): it floor-bins x/y
over ALL frames of the movie and clears every blob in any ``bin_size`` x ``bin_size``
px bin that holds >= ``blob_filter`` detections. Consequence for the methods: a real
particle that pauses in one bin for >= ``blob_filter`` frames is removed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def preprocess_stack(
    img_stack: np.ndarray,
    blob_channel: int,
    mask_channel: int,
    median_filter_size: int,
    min_object_size: int,
    dilation_radius: int,
    apply_mask: bool,
) -> np.ndarray:
    """Background-subtract the blob channel of a (T, C, Y, X) stack -> (T, Y, X) uint16.

    Background = median-filtered mean projection of the blob channel, subtracted from
    every frame (negatives clipped to 0). If ``apply_mask``, a region from the mask
    channel's max projection (Otsu -> fill holes -> remove small -> dilate) zeroes
    everything outside it.
    """
    from scipy import ndimage as ndi
    from scipy.ndimage import median_filter
    from skimage import filters, morphology
    from skimage.morphology import disk

    if img_stack.ndim != 4:
        raise ValueError(f"Expected (T,C,Y,X); got {img_stack.shape}")
    if blob_channel >= img_stack.shape[1]:
        raise ValueError(
            f"blob_channel={blob_channel} but image has {img_stack.shape[1]} channels"
        )

    mean_proj = np.mean(img_stack[:, blob_channel], axis=0)
    background = median_filter(mean_proj, size=median_filter_size)

    preprocessed = np.stack([
        np.clip(frame - background, 0, None).astype(np.uint16)
        for frame in img_stack[:, blob_channel]
    ])

    if apply_mask:
        if mask_channel >= img_stack.shape[1]:
            raise ValueError(
                f"mask_channel={mask_channel} but image has {img_stack.shape[1]} channels"
            )
        ch_mask_proj = np.max(img_stack[:, mask_channel], axis=0)
        thresh = filters.threshold_otsu(ch_mask_proj)
        binary = ch_mask_proj > thresh
        filled = ndi.binary_fill_holes(binary)
        large = morphology.remove_small_objects(
            filled, min_size=min_object_size
        )
        dilated = morphology.binary_dilation(
            large, disk(dilation_radius)
        )
        preprocessed *= dilated.astype(np.uint16)

    return preprocessed


def detect_blobs_in_stack(
    image_stack: np.ndarray,
    threshold: float,
    min_sigma: float,
    max_sigma: float,
    gaussian_sigma: float,
    normalize: bool,
    use_gaussian_blur: bool,
) -> pd.DataFrame:
    """Per-frame LoG blob detection (dask). Returns columns frame, x, y, size.

    ``size`` is the blob radius estimate (sigma * sqrt(2)); it is dropped by
    ``blobs_df_to_tyx``.
    """
    from dask import compute, delayed
    from dask.diagnostics import ProgressBar
    from skimage import filters
    from skimage.feature import blob_log

    @delayed
    def process_frame(t, frame):
        work = frame
        if normalize:
            work = (work - np.min(work)) / (
                np.max(work) - np.min(work) + 1e-8
            )
        if use_gaussian_blur:
            work = filters.gaussian(work, sigma=gaussian_sigma)
        blobs = blob_log(
            work,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            threshold=threshold,
        )
        return [
            {
                "frame": t,
                "x": x,
                "y": y,
                "size": sigma * np.sqrt(2),
            }
            for y, x, sigma in blobs
        ]

    tasks = [process_frame(t, frame) for t, frame in enumerate(image_stack)]
    with ProgressBar():
        results = compute(*tasks)
    positions = [entry for sublist in results for entry in sublist]
    return pd.DataFrame(positions)


def filter_dense_blobs(
    blob_df: pd.DataFrame,
    bin_size: int,
    blob_filter: int,
) -> pd.DataFrame:
    """Drop every blob in any ``bin_size`` px bin holding >= ``blob_filter`` blobs (all frames pooled)."""
    if len(blob_df) == 0:
        return blob_df

    rounded = blob_df.copy()
    rounded["x_rounded"] = (rounded["x"] // bin_size * bin_size).astype(int)
    rounded["y_rounded"] = (rounded["y"] // bin_size * bin_size).astype(int)

    counts = (
        rounded.groupby(["x_rounded", "y_rounded"])
        .size()
        .reset_index(name="count")
    )
    merged = rounded.merge(counts, on=["x_rounded", "y_rounded"])
    return merged[merged["count"] < blob_filter].reset_index(drop=True)


def blobs_df_to_tyx(df: pd.DataFrame) -> pd.DataFrame:
    """['frame', 'x', 'y', 'size'] -> integer ['t', 'y', 'x'] (the blobs.csv format)."""
    if len(df) == 0:
        return pd.DataFrame(columns=["t", "y", "x"], dtype=int)
    return pd.DataFrame(
        {
            "t": df["frame"].astype(int),
            "y": np.round(df["y"]).astype(int),
            "x": np.round(df["x"]).astype(int),
        }
    )


def run_detection(img_stack: np.ndarray, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full detection step as run by the batch pipeline.

    ``params`` is the pipeline's ``DETECTION_PARAMS`` (also stored as ``detection_params``
    in metadata.json). Returns ``(raw_blobs, blobs_tyx)``: the unfiltered detections
    (frame, x, y, size) and the density-filtered detections in blobs.csv format (t, y, x).
    """
    pre = preprocess_stack(
        img_stack,
        blob_channel=params["blob_channel"],
        mask_channel=params["mask_channel"],
        median_filter_size=params["median_filter_size"],
        min_object_size=params["min_object_size"],
        dilation_radius=params["mask_dilation_radius"],
        apply_mask=params["apply_mask"],
    )
    raw_blobs = detect_blobs_in_stack(
        pre,
        threshold=params["threshold"],
        min_sigma=params["min_sigma"],
        max_sigma=params["max_sigma"],
        gaussian_sigma=params["gaussian_sigma"],
        normalize=params["normalize"],
        use_gaussian_blur=params["use_gaussian_blur"],
    )
    filtered = filter_dense_blobs(
        raw_blobs,
        bin_size=params["bin_size"],
        blob_filter=params["blob_filter"],
    )
    return raw_blobs, blobs_df_to_tyx(filtered)
