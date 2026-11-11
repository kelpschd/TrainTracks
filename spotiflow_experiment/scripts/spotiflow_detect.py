# spotiflow_detect.py
#
# Drop-in alternative to preprocess_blob_det.py: runs Spotiflow instead of
# LoG blob_log, and writes (t, y, x) CSVs that candidate_graph.py can consume.
#
# Input : a .zarr image (T, C, Y, X)   (same one preprocess_blob_det.py uses)
# Output (all in OUTPUT_DIR):
#   spotiflow_all_<model>.csv            every spot above MIN_PROB, with
#                                        t, y, x (sub-pixel), prob, intens
#   blobs_spotiflow_<model>_p<prob>.csv  one per PROB_THRESHOLDS entry,
#                                        columns t, y, x (ints), dense-bin
#                                        filtered if APPLY_DENSE_FILTER
#
# Spotiflow is run ONCE per frame at the lowest probability threshold; the
# higher thresholds are derived by filtering on each spot's probability, so a
# threshold sweep costs nothing extra (this mirrors your LoG threshold sweep).
#
# Install (in the TrainTracks env, on your Mac):
#   pip install spotiflow          # pulls in torch
#
# Usage:
#   python scripts/spotiflow_detect.py
#   python scripts/spotiflow_detect.py --max-frames 3     # quick smoke test

from __future__ import annotations

# macOS: torch and other packages (numpy/scipy/opencv) each ship their own
# libomp, and loading both aborts with "OMP: Error #15". This must be set
# BEFORE any of them are imported. Detection results are not affected in
# practice for inference; remove this if you fix the duplicate libomp in the env.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

# Reuse your existing preprocessing + dense filter so the ONLY difference
# between the LoG and Spotiflow outputs is the detector itself.
from preprocess_blob_det import (
    preprocess_stack,
    filter_dense_blobs,
    blobs_df_to_tyx,
    BIN_SIZE,
    BLOB_FILTER,
)


# ============================================================================
# CONFIG
# ============================================================================

INPUT_ZARR = Path(
    "/Users/kelpschdj/Documents/DataTecnica/TTU/Data/Sphere/"
    "220725_i11w-hT-M33-I76_sg1035_d10sphere/raw_data/"
    "sg100_Well5_1018.zarr"
)
ZARR_ARRAY_KEY = "s0"
OUTPUT_DIR = Path("/Users/kelpschdj/Documents/DataTecnica/TTU/TrainTracks/blobs")

BLOB_CHANNEL = 0

# Input to Spotiflow:
#   "raw"    -> raw channel (Spotiflow normalizes internally; its intended use)
#   "bgsub"  -> your median-filter background subtraction first (apples-to-apples
#               with the LoG input). Worth running BOTH and comparing.
INPUT_MODE = "raw"

MODEL_NAME = "general"          # pretrained: general | hybiss | synth_complex | smfish_3d (3D only)
DEVICE = "auto"                 # "auto" | "mps" | "cuda" | "cpu"

# Probability thresholds to emit. MIN_PROB is what the model is actually run
# with; it must be <= min(PROB_THRESHOLDS). The model's own auto-optimized
# threshold for 'general' is ~0.5-0.6; lower = more spots (like lower LoG thr).
PROB_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
MIN_PROB = min(PROB_THRESHOLDS)

# Same dense-bin filter as the LoG pipeline (5x5 px bin, >= 5 spots over movie).
APPLY_DENSE_FILTER = True
# Dense-bin cutoffs to emit. 5 is your LoG-calibrated value (original filenames);
# Spotiflow puts more spots per bin, so 10 recovers much more of the GT.
# Extra cutoffs are saved as ..._bf10.csv etc.
BLOB_FILTER_VALUES = [BLOB_FILTER, 10]


# ============================================================================

def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def detect_stack(frames: np.ndarray, model, min_prob: float) -> pd.DataFrame:
    """Run Spotiflow per frame. frames: (T, Y, X). Returns t,y,x,prob,intens."""
    from tqdm.auto import tqdm
    rows = []
    for t, frame in enumerate(tqdm(frames, desc="Spotiflow")):
        pts, details = model.predict(
            frame.astype(np.float32),
            prob_thresh=min_prob,
            subpix=True,
            verbose=False,
        )
        n = len(pts)
        prob = np.asarray(getattr(details, "prob", np.full(n, np.nan)))
        # details.intens may not be a flat per-spot array (shape varies by
        # version); reduce to one value per spot, else NaN. Not used downstream.
        intens = np.asarray(getattr(details, "intens", np.full(n, np.nan)), dtype=float)
        if intens.ndim == 0 or intens.shape[0] != n:
            intens = np.full(n, np.nan)
        else:
            intens = intens.reshape(n, -1).mean(axis=1)
        if t == 0:
            print(f"  details.prob shape={prob.shape}, intens raw shape="
                  f"{np.asarray(getattr(details, 'intens', [])).shape}, n={n}")
        for i in range(n):
            rows.append((t, float(pts[i][0]), float(pts[i][1]),
                         float(prob[i]), float(intens[i])))
    return pd.DataFrame(rows, columns=["t", "y", "x", "prob", "intens"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Only process the first N frames (smoke test)")
    args = ap.parse_args()

    from spotiflow.model import Spotiflow

    print(f"Loading {INPUT_ZARR} [{ZARR_ARRAY_KEY}]")
    root = zarr.open(str(INPUT_ZARR), mode="r")
    stack = np.array(root[ZARR_ARRAY_KEY])      # (T, C, Y, X)
    print(f"Image shape (T,C,Y,X): {stack.shape}")

    if INPUT_MODE == "bgsub":
        frames = preprocess_stack(stack, blob_channel=BLOB_CHANNEL)   # (T,Y,X) uint16
    elif INPUT_MODE == "raw":
        frames = stack[:, BLOB_CHANNEL]
    else:
        raise ValueError(f"INPUT_MODE must be 'raw' or 'bgsub', got {INPUT_MODE!r}")
    if args.max_frames:
        frames = frames[: args.max_frames]

    device = pick_device(DEVICE)
    print(f"Loading Spotiflow '{MODEL_NAME}' on device={device}")
    model = Spotiflow.from_pretrained(MODEL_NAME)
    try:
        model = model.to(device)
    except Exception as e:                      # fall back rather than crash
        print(f"  could not move model to {device} ({e}); using default device")

    df = detect_stack(frames, model, MIN_PROB)
    print(f"Spotiflow found {len(df)} spots at prob >= {MIN_PROB}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{MODEL_NAME}_{INPUT_MODE}"
    all_csv = OUTPUT_DIR / f"spotiflow_all_{tag}.csv"
    df.to_csv(all_csv, index=False)
    print(f"Wrote {all_csv}")

    for p in PROB_THRESHOLDS:
        sub = df[df["prob"] >= p]
        for bf in BLOB_FILTER_VALUES:
            # filter_dense_blobs / blobs_df_to_tyx expect ['frame','x','y'] columns
            tmp = pd.DataFrame({"frame": sub["t"], "x": sub["x"], "y": sub["y"]})
            if APPLY_DENSE_FILTER:
                tmp = filter_dense_blobs(tmp, BIN_SIZE, bf)
            tyx = blobs_df_to_tyx(tmp)
            # Default cutoff keeps the original filename; others get a _bf<N> suffix
            suffix = "" if bf == BLOB_FILTER else f"_bf{bf}"
            out = OUTPUT_DIR / (f"blobs_spotiflow_{tag}_p{str(p).replace('.', '_')}"
                                f"{suffix}.csv")
            tyx.to_csv(out, index=False)
            print(f"  prob>={p}, blob_filter={bf}: {len(sub)} spots -> "
                  f"{len(tyx)} after filter -> {out.name}")
            if not APPLY_DENSE_FILTER:
                break   # filter off: one file per threshold


if __name__ == "__main__":
    main()
