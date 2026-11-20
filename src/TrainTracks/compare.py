"""Compare two pipeline runs (an archived one and a fresh one) output by output.

Tracks are compared through ``track_uid`` (a hash of each track's frame/position sequence), never
through ``particle`` or node IDs, because those are not stable between runs. Flow is compared bit
for bit, one frame at a time (low memory, so it also works on a cluster node). Pure numpy/pandas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


def compare_blobs(a: pd.DataFrame, b: pd.DataFrame) -> Check:
    a = a.reset_index(drop=True).astype("int64")
    b = b.reset_index(drop=True).astype("int64")
    if a.equals(b):
        return Check("blobs.csv", True, f"identical ({len(a)} rows)")
    return Check("blobs.csv", False, f"different: {len(a)} vs {len(b)} rows")


def compare_flow(a, b) -> Check:
    """``a`` and ``b`` are arrays (numpy or zarr) of shape (T-1, Y, X, 2)."""
    if tuple(a.shape) != tuple(b.shape):
        return Check("flow_raw", False, f"shape {tuple(a.shape)} vs {tuple(b.shape)}")
    n_diff = 0
    max_abs = 0.0
    for t in range(a.shape[0]):
        fa = np.asarray(a[t]); fb = np.asarray(b[t])
        d = np.abs(fa.astype(np.float64) - fb.astype(np.float64))
        n_diff += int((d > 0).sum())
        max_abs = max(max_abs, float(d.max()))
    if n_diff == 0:
        return Check("flow_raw", True, f"bit-identical ({a.shape[0]} frames)")
    return Check("flow_raw", max_abs <= 1e-4, f"{n_diff} values differ, max |diff| {max_abs:.2e}")


def compare_tracks(name: str, a: pd.DataFrame, b: pd.DataFrame) -> Check:
    """Same set of tracks (track_uid) and same number of detections."""
    ua, ub = set(a["track_uid"]), set(b["track_uid"])
    if ua == ub and len(a) == len(b):
        return Check(name, True, f"identical ({len(ua)} tracks, {len(a)} detections)")
    return Check(name, False, f"tracks {len(ua)} vs {len(ub)}; only in old {len(ua - ub)}, only in new {len(ub - ua)}")


def compare_scores(a: pd.DataFrame, b: pd.DataFrame, atol: float = 1e-9) -> Check:
    """flow_scores.csv: match rows by track_uid, then compare every shared column except IDs."""
    if "track_uid" not in a.columns or "track_uid" not in b.columns:
        return Check("flow_scores.csv", False, "track_uid column missing")
    if set(a["track_uid"]) != set(b["track_uid"]) or a["track_uid"].duplicated().any() or b["track_uid"].duplicated().any():
        return Check("flow_scores.csv", False, "different set of track_uid values")
    a = a.set_index("track_uid").sort_index()
    b = b.set_index("track_uid").sort_index()
    bad = []
    for col in sorted((set(a.columns) & set(b.columns)) - {"particle"}):
        x, y = a[col], b[col]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(x):
            same = np.allclose(x.to_numpy(float), y.to_numpy(float), rtol=0, atol=atol, equal_nan=True)
        else:
            same = bool((x.astype(str) == y.astype(str)).all())
        if not same:
            bad.append(col)
    if bad:
        return Check("flow_scores.csv", False, f"columns differ: {bad}")
    return Check("flow_scores.csv", True, f"identical ({len(a)} tracks, {len(set(a.columns) & set(b.columns)) - 1} columns)")


def compare_metadata(a: dict, b: dict) -> Check:
    keys = ["n_blobs_raw", "n_blobs_input", "n_detections_tracked", "n_edges", "n_tracks", "n_skip_edges",
            "image_shape", "detection_params", "flow_params", "tracking_params"]
    bad = [k for k in keys if a.get(k) != b.get(k)]
    pa, pb = a.get("posthoc_results", {}), b.get("posthoc_results", {})
    # the old (Sept 30) posthoc_results came from the superseded pipeline scorer, so only report it
    note = "" if pa == pb else " (posthoc_results differ: old file predates the unified scorer)"
    return Check("metadata.json", not bad, ("identical counts and parameters" if not bad else f"differ: {bad}") + note)
