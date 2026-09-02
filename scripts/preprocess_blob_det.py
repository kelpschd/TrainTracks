# detect_blobs.py
#
# Input : a .zarr image (T, C, Y, X)
# Output: a CSV of detected blobs with columns [t, y, x] -- drop-in for
#         candidate_graph.py (which indexes a mask as mask[t, y, x]).
#
# Pipeline:  load zarr -> background-subtract blob channel -> LoG blob detection
#            (dask, per frame) -> convert to (t, y, x) integers -> write CSV
#
# Masking from a second channel is available in preprocess_stack() but OFF by
# default (APPLY_MASK = False): background subtraction only, per your setup.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy import ndimage as ndi
from skimage import filters, morphology
from skimage.morphology import disk
from skimage.feature import blob_log

import dask
from dask import delayed, compute
from dask.diagnostics import ProgressBar

import zarr


# ============================================================================
# CONFIG -- edit these, then run `python detect_blobs.py`
# ============================================================================

# Input image zarr (T, C, Y, X). Point at the .zarr directory itself.
INPUT_ZARR = Path(
    "/Users/kelpschdj/Documents/DataTecnica/TTU/Data/Sphere/"
    "220725_i11w-hT-M33-I76_sg1035_d10sphere/raw_data/"
    "sg100_Well5_1018.zarr"   # <-- set to your actual .zarr filename
)
ZARR_ARRAY_KEY = "s0"          # array inside the zarr group (matches your scripts)

# Output CSV directory (t, y, x). Filename is auto-generated as
# blobs_<THRESHOLD>.csv so each threshold's output is kept separate.
OUTPUT_DIR = Path(
    "/Users/kelpschdj/Documents/DataTecnica/TTU/TrainTracks/blobs"
)

# --- Preprocessing ------------------------------------------------------------
BLOB_CHANNEL = 0               # channel to detect blobs on
MASK_CHANNEL = 0               # only used if APPLY_MASK = True
MEDIAN_FILTER_SIZE = 10        # background estimate via median filter of mean proj
APPLY_MASK = False             # background subtraction only (no mask)
MIN_OBJECT_SIZE = 400          # mask param (unused when APPLY_MASK=False)
MASK_DILATION_RADIUS = 5       # mask param (unused when APPLY_MASK=False)

# --- Detection (blob_log) -----------------------------------------------------
THRESHOLD = 0.00015            # your working value - this was orginally 0.00021
MIN_SIGMA = 1
MAX_SIGMA = 3
GAUSSIAN_SIGMA = 1             # pre-blur sigma; only used if USE_GAUSSIAN_BLUR
USE_GAUSSIAN_BLUR = False      # your note: blur "mucks with blob detection"
NORMALIZE = False              # detect on raw bg-subtracted intensities (not [0,1])

# --- Density filtering --------------------------------------------------------
# Drops blobs in over-dense local neighborhoods (noise floods): any bin_size x
# bin_size pixel bin holding >= BLOB_FILTER blobs has ALL its blobs removed.
BIN_SIZE = 5
BLOB_FILTER = 5


# ============================================================================
# Preprocessing
# ============================================================================

def preprocess_stack(
    img_stack: np.ndarray,
    blob_channel: int = BLOB_CHANNEL,
    mask_channel: int = MASK_CHANNEL,
    median_filter_size: int = MEDIAN_FILTER_SIZE,
    min_object_size: int = MIN_OBJECT_SIZE,
    dilation_radius: int = MASK_DILATION_RADIUS,
    apply_mask: bool = APPLY_MASK,
) -> np.ndarray:
    """Background-subtract the blob channel; optionally mask via another channel.

    img_stack : (T, C, Y, X). Returns (T, Y, X) uint16.

    Background = median-filtered mean projection of the blob channel; it's
    subtracted from every frame (negatives clipped to 0). If apply_mask, a
    binary region from the mask channel's max projection (Otsu -> fill -> remove
    small -> dilate) multiplies the result to zero out everything outside it.
    """
    # Mean projection for background subtraction
    mean_proj = np.mean(img_stack[:, blob_channel], axis=0)
    background = median_filter(mean_proj, size=median_filter_size)

    # Subtract background, clip negatives
    preprocessed = np.stack([
        np.clip(frame - background, 0, None).astype(np.uint16)
        for frame in img_stack[:, blob_channel]
    ])

    if apply_mask:
        ch_mask_proj = np.max(img_stack[:, mask_channel], axis=0)
        thresh = filters.threshold_otsu(ch_mask_proj)
        binary = ch_mask_proj > thresh
        filled = ndi.binary_fill_holes(binary)
        large = morphology.remove_small_objects(filled, min_size=min_object_size)
        dilated = morphology.binary_dilation(large, disk(dilation_radius))
        preprocessed *= dilated.astype(np.uint16)

    return preprocessed


# ============================================================================
# Detection
# ============================================================================

def detect_blobs_in_stack(
    image_stack: np.ndarray,
    threshold: float = THRESHOLD,
    min_sigma: float = MIN_SIGMA,
    max_sigma: float = MAX_SIGMA,
    gaussian_sigma: float = GAUSSIAN_SIGMA,
    normalize: bool = NORMALIZE,
    use_gaussian_blur: bool = USE_GAUSSIAN_BLUR,
) -> pd.DataFrame:
    """Per-frame LoG blob detection via dask. Returns ['frame','x','y','size'].

    image_stack : (T, Y, X). Each frame is optionally min-max normalized (and
    optionally pre-blurred) before blob_log. `size` is the blob's radius
    estimate (sigma * sqrt(2)); it's dropped when writing the (t,y,x) CSV.
    """
    @delayed
    def process_frame(t, frame):
        if normalize:
            frame = (frame - np.min(frame)) / (np.max(frame) - np.min(frame) + 1e-8)
        if use_gaussian_blur:
            frame = filters.gaussian(frame, sigma=gaussian_sigma)
        blobs = blob_log(frame, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
        return [{"frame": t, "x": x, "y": y, "size": sigma * np.sqrt(2)}
                for y, x, sigma in blobs]

    tasks = [process_frame(t, frame) for t, frame in enumerate(image_stack)]
    with ProgressBar():
        results = compute(*tasks)
    positions = [entry for sublist in results for entry in sublist]
    print(f"Puncta counting complete: {len(positions)} total puncta were found")
    return pd.DataFrame(positions)


def filter_dense_blobs(
    blob_df: pd.DataFrame,
    bin_size: int = BIN_SIZE,
    blob_filter: int = BLOB_FILTER,
) -> pd.DataFrame:
    """Remove blobs in over-dense local neighborhoods (your filter_dense_blobs).

    Positions are floor-binned to a bin_size grid; any bin holding >= blob_filter
    blobs has ALL of its blobs dropped. This clears the dense noise clusters that
    otherwise dominate the raw count. The @capture_metadata decorator is omitted.
    """
    if len(blob_df) == 0:
        return blob_df

    rounded = blob_df.copy()
    rounded["x_rounded"] = (rounded["x"] // bin_size * bin_size).astype(int)
    rounded["y_rounded"] = (rounded["y"] // bin_size * bin_size).astype(int)

    counts = rounded.groupby(["x_rounded", "y_rounded"]).size().reset_index(name="count")
    merged = rounded.merge(counts, on=["x_rounded", "y_rounded"])
    filtered = merged[merged["count"] < blob_filter].reset_index(drop=True)

    print(f"Puncta filtering complete: {len(filtered)} puncta remain")
    return filtered


# ============================================================================
# Conversion + driver
# ============================================================================

def blobs_df_to_tyx(df: pd.DataFrame) -> pd.DataFrame:
    """['frame','x','y','size'] -> ['t','y','x'] integer, drop-in for the pipeline.

    candidate_graph.py indexes mask[t, y, x], so column ORDER matters here:
    frame->t, then y, then x, with size dropped and coords rounded to int.
    """
    if len(df) == 0:
        return pd.DataFrame(columns=["t", "y", "x"], dtype=int)
    return pd.DataFrame({
        "t": df["frame"].astype(int),
        "y": np.round(df["y"]).astype(int),
        "x": np.round(df["x"]).astype(int),
    })


def main():
    print(f"Loading {INPUT_ZARR} [{ZARR_ARRAY_KEY}]")
    zarr_root = zarr.open(str(INPUT_ZARR), mode="r")
    img_stack = np.array(zarr_root[ZARR_ARRAY_KEY])   # (T, C, Y, X)
    print(f"Image shape (T,C,Y,X): {img_stack.shape}")

    if img_stack.ndim != 4:
        raise ValueError(f"Expected (T,C,Y,X); got {img_stack.shape}")
    if BLOB_CHANNEL >= img_stack.shape[1]:
        raise ValueError(
            f"BLOB_CHANNEL={BLOB_CHANNEL} but image has only "
            f"{img_stack.shape[1]} channels."
        )

    print(f"Preprocessing (background subtraction"
          f"{', masking' if APPLY_MASK else ''}) on channel {BLOB_CHANNEL}...")
    pre = preprocess_stack(img_stack)   # (T, Y, X)
    print(f"Preprocessed stack shape (T,Y,X): {pre.shape}")

    print(f"Detecting blobs (threshold={THRESHOLD:g})...")
    blobs_df = detect_blobs_in_stack(pre)

    print(f"Filtering dense blobs (bin_size={BIN_SIZE}, blob_filter={BLOB_FILTER})...")
    blobs_df = filter_dense_blobs(blobs_df)

    tyx = blobs_df_to_tyx(blobs_df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    thr_str = f"{THRESHOLD:g}".replace(".", "_")
    output_csv = OUTPUT_DIR / f"blobs_{thr_str}.csv"
    tyx.to_csv(output_csv, index=False)
    print(f"Wrote {len(tyx)} blobs -> {output_csv}")


if __name__ == "__main__":
    main()