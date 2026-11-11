# spotiflow_recall_sweep.py
#
# Diagnostic: GT detection recall of Spotiflow as a function of probability
# threshold, WITH and WITHOUT your dense-bin filter, using the unfiltered
# spotiflow_all_<tag>.csv written by spotiflow_detect.py.
#
# Answers: are GT spots being lost because Spotiflow gives them low probability,
# or because filter_dense_blobs removes them (e.g. paused particles)?
#
# Usage: python scripts/spotiflow_recall_sweep.py [tag]     (default: general_raw)

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess_blob_det import filter_dense_blobs, BIN_SIZE, BLOB_FILTER

BASE = Path("/Users/kelpschdj/Documents/DataTecnica/TTU/TrainTracks")
GT_CSV = BASE / "sg100_Well5_1018_valid_tracks.csv"
TOL = 5.0
PROBS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def recall(det: pd.DataFrame, gt: pd.DataFrame) -> float:
    by_t = {t: g[["y", "x"]].to_numpy(float) for t, g in det.groupby("t")}
    hits = 0
    for _, r in gt.iterrows():
        d = by_t.get(r["frame"])
        if d is not None and np.hypot(d[:, 0] - r["y"], d[:, 1] - r["x"]).min() <= TOL:
            hits += 1
    return hits / len(gt)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "general_raw"
    allspots = pd.read_csv(BASE / "blobs" / f"spotiflow_all_{tag}.csv")
    gt = pd.read_csv(GT_CSV, encoding="utf-8-sig")
    print(f"min prob in file: {allspots['prob'].min():.3f}  "
          f"(thresholds below this are not meaningful)")
    rows = []
    for p in PROBS:
        sub = allspots[allspots["prob"] >= p]
        tmp = pd.DataFrame({"frame": sub["t"], "x": sub["x"], "y": sub["y"]})
        filt = filter_dense_blobs(tmp, BIN_SIZE, BLOB_FILTER).rename(columns={"frame": "t"})
        rows.append(dict(prob=p,
                         n_unfiltered=len(sub), recall_unfiltered=recall(sub, gt),
                         n_filtered=len(filt), recall_filtered=recall(filt, gt)))
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    # Second table: how does the dense-filter cutoff (BLOB_FILTER) trade recall
    # against spot count? 5 is the LoG-calibrated value; None = no filter.
    import contextlib, io
    rows2 = []
    for p in (0.3, 0.4):
        sub = allspots[allspots["prob"] >= p]
        tmp = pd.DataFrame({"frame": sub["t"], "x": sub["x"], "y": sub["y"]})
        for bf in (5, 10, 20, 50, None):
            if bf is None:
                filt = sub
            else:
                with contextlib.redirect_stdout(io.StringIO()):   # silence its print
                    filt = filter_dense_blobs(tmp, BIN_SIZE, bf).rename(columns={"frame": "t"})
            rows2.append(dict(prob=p, blob_filter=bf if bf else "none",
                              n_spots=len(filt), recall=recall(filt, gt)))
    print("\nDense-filter cutoff sweep (bin size %d px):" % BIN_SIZE)
    print(pd.DataFrame(rows2).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
