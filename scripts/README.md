# scripts/

Command-line entry points. The processing steps themselves live in `../src/TrainTracks/`; see the root
`README.md` for the run order, what each step writes and the settings used.

| Script | Purpose |
|---|---|
| `import_export.py` | convert `.nd2` movies to `.zarr` (next to the `.nd2`) |
| `batch_pipeline.py` | detect, flow, track, score, clean, for all movies; `--jobs`, `--profile`, `--movie`, `--skip-existing`, `--summarize-only`, SLURM sharding |
| `batch_traccuracy.py` | recall of raw and clean tracks against `../ground_truth/` |
| `batch_lyso_interaction.py` | lysosome / mitochondria segmentation and per-track contact features (`--jobs`, `--skip-existing`) |
| `batch_puncta_colocalization.py` | box-based puncta-organelle correlation per detection |
| `batch_rescore_tracks.py` | re-score existing tracks with the current scorer (add `--write-clean` to rebuild `tracks_clean.geff`; it rewrites `flow_scores.csv`) |
| `track_flow_score.py` | single-run scorer plus `--gt` calibration on one movie |
| `calibrate_cull_on_gt.py` | check the cull cutoff against the validated tracks of all movies |
| `check_regression.py` | re-run one stage (detection, flow, tracking) and compare with stored output |
| `compare_runs.py` | compare two complete runs movie by movie (blobs, flow, tracks, scores) |
| `batch_track_review.py`, `review_lyso_interaction.py`, `straightness_review.py` | napari review tools |

Cotracking scripts are in `../cotracking/`. Retired scripts are in `../archive/`.
