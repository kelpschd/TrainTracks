# archive/

Frozen work that is no longer part of the pipeline but is kept for provenance.
Nothing here is imported by the current code. The scripts keep their original
hard-coded paths (`BASE = .../TrainTracks`, `tracking_runs/`, `blobs/`), which no
longer exist at those locations. To re-run any of it exactly as it was, check out the
tag `pre-cleanup-2026-11-19` (commit 7bb84e0), which has the original layout.

| Folder | What it is |
|---|---|
| `exploration_2026-09/` | Single-movie exploration scripts (Jul-Sep 2026): run viewers, parameter sweeps, traccuracy comparisons, the original candidate-graph step. |
| `legacy_single_movie/` | Data those scripts read and wrote: LoG blob CSVs, the 27-run tracking log and run folders, traccuracy result tables for sg100_Well5_1018. |
| `spotiflow_experiment/` | Spotiflow vs LoG detector comparison (archived 2026-11-10); conclusion in its README. |

The early optical flow for sg100_Well5_1018 that used to be `flow.zarr` here was moved
to `<results root>/sg100_Well5_1018/archived_flow/` (outside the repo).
