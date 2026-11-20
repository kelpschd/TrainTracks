"""Loading tracks from GEFF into a flat DataFrame.

Output columns: ``particle`` (int, renumbered by funtracks on every import, so only
valid within one import), ``frame`` (int), ``y``, ``x`` (float), ``track_uid`` (str,
stable across imports; use this for joins). Sorted by (particle, frame).

funtracks is imported lazily so this module (and the pure-pandas helpers below) can
be imported and tested without the full tracking stack installed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from TrainTracks.ids import add_track_uids

# How the pipeline's GEFFs map onto funtracks' node attributes.
NODE_NAME_MAP = {"time": "t", "pos": ["y", "x"]}

REQUIRED_NODE_ATTRS = ("tracklet_id", "t", "pos_0", "pos_1")


def track_df_from_node_attrs(attrs: pd.DataFrame, source: str = "<graph>") -> pd.DataFrame:
    """Build the track DataFrame from funtracks node attributes (pure pandas)."""
    missing = set(REQUIRED_NODE_ATTRS) - set(attrs.columns)
    if missing:
        raise ValueError(f"{source}: missing imported attributes {sorted(missing)}")
    df = pd.DataFrame(
        {
            "particle": attrs["tracklet_id"].astype(int),
            "frame": attrs["t"].astype(int),
            "y": attrs["pos_0"].astype(float),
            "x": attrs["pos_1"].astype(float),
        }
    )
    return add_track_uids(df).sort_values(["particle", "frame"]).reset_index(drop=True)


def solution_tracks_to_track_df(sg, source: str = "<SolutionTracks>") -> pd.DataFrame:
    """Convert a funtracks SolutionTracks object to the track DataFrame."""
    attrs = sg.graph.node_attrs(unpack=True).to_pandas()
    return track_df_from_node_attrs(attrs, source)


def load_solution_tracks(geff_path):
    """Import a GEFF as a funtracks SolutionTracks (needs funtracks installed)."""
    from funtracks.import_export import import_from_geff

    return import_from_geff(directory=str(geff_path), node_name_map=NODE_NAME_MAP)


def geff_to_track_df(geff_path) -> pd.DataFrame:
    """Read a GEFF directory into the track DataFrame."""
    return solution_tracks_to_track_df(load_solution_tracks(geff_path), source=str(Path(geff_path)))
