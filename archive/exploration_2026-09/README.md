# Exploration scripts, Jul-Sep 2026

Single-movie workflow on sg100_Well5_1018, before the batch pipeline existed.
`scripts/batch_pipeline.py` reimplements the detect -> flow -> candidate graph -> solve
steps for all movies; these scripts are what it grew out of.

| Script | Purpose |
|---|---|
| `candidate_graph.py` | Original single-run candidate graph + motile solve; writes run folders under `tracking_runs/runs/<run_id>/` and appends to `tracking_log.csv`. |
| `sweep_edge_constant.py`, `sweep_gap_penalty.py` | Parameter sweeps that call `candidate_graph`. |
| `traccuracy_compare.py`, `traccuracy_runs_EdgeConstantSweep.py`, `vs_gt_matplotlib.py` | Compare runs against the 11-track GT; plot the result. Outputs are in `../legacy_single_movie/`. (`traccuracy_runs_EdgeConstantSweep.py` labels its keys `pen*` for what is an edge-constant sweep.) |
| `track_vis.py`, `track_vis_FrameSweep.py`, `track_vis_thresholds.py`, `track_vis_extra.py` | Napari viewers for runs. `IMG_NUM = 9` selects the movie by glob order. |
| `flow_vis.py` | Renders flow fields to RGB frames in `flow.zarr/flow_frames_XY`. |

Known limitations (so nobody reuses them by accident): hard-coded `/Users/...` paths
(`track_vis_extra.py` still has an old `/Users/dankelpsch/` path), 63-bit track hashes
that collide when passed through float64 into napari, and the candidate graph merges
detections within about 10 px (see TODO.md Tier 2).

`spotiflow_experiment/scripts/` imports `traccuracy_compare` and `preprocess_blob_det`;
see that folder's README for the PYTHONPATH to use.
