# Data from the single-movie exploration (sg100_Well5_1018)

- `blobs/`: LoG detections at thresholds 0.00015, 0.00017, 0.00019, 0.00021 (the
  Spotiflow detections live in `../spotiflow_experiment/blobs/`).
- `tracking_runs/`: 24 run folders (20260914-20260916 sweeps, and four 20261110 Spotiflow
  runs) with their `.geff` outputs and metadata, plus `tracking_log.csv` (27 rows
  from 2026-09-02 on; version controlled). `blobs_0_000*.geff` are the first four threshold runs.
- `traccuracy_vs_gt.csv`, `traccuracy_pairwise.csv`, `traccuracy_runs_vs_gt.csv`,
  `traccuracy_runs_pairwise.csv`, `vs_gt_metrics.png`: traccuracy results against the
  11-track GT (`sg100_Well5_1018_valid_tracks.csv`, repo root until Phase 3 moves it).
- `test_run_7.geff`: an early test tracking output from 2026-07-30; no metadata.

Mostly gitignored (`*.csv`, `*.geff`, `*.png`) so only `tracking_log.csv` is in git.
Back this folder up outside git if you want to keep the sweep results.
