# scripts/

Current pipeline code (to be consolidated into `src/TrainTracks/` in cleanup Phase 3).
Older single-movie exploration scripts are in `../archive/`.

Pipeline: `import_export.py` (nd2 -> zarr) -> `preprocess_blob_det.py`,
`optical_flow.py` -> `batch_pipeline.py` (detect, flow, candidate graph, solve,
score, clean, all movies) -> `track_flow_score.py` (scorer; also `--gt` calibration)
-> `batch_rescore_tracks.py`, `batch_traccuracy.py` -> `batch_lyso_interaction.py`,
`batch_puncta_colocalization.py`. Cotracking is in `../cotracking/`.

Review tools: `batch_track_review.py`, `review_lyso_interaction.py`,
`straightness_review.py`.
