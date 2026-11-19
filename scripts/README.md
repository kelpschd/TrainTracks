# scripts/

Current pipeline code (to be consolidated into `src/TrainTracks/` in cleanup Phase 3).
Older single-movie exploration scripts are in `../archive/`.

Pipeline: `import_export.py` (nd2 -> zarr) -> `batch_pipeline.py` (detect, flow, candidate
graph, solve, score, clean, all movies; the steps live in `src/TrainTracks/`:
`detection.py`, `flow.py`, `tracking.py`, `scoring.py`) -> `track_flow_score.py` (`--gt` calibration)
-> `batch_rescore_tracks.py`, `batch_traccuracy.py` -> `batch_lyso_interaction.py`,
`batch_puncta_colocalization.py`. Cotracking is in `../cotracking/`.

Review tools: `batch_track_review.py`, `review_lyso_interaction.py`,
`straightness_review.py`. Refactor safety net: `check_regression.py detection|flow|tracking
<movie_dir>` re-runs a stage and compares with the stored outputs.
