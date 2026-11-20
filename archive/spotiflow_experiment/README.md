# Spotiflow vs LoG experiment (archived 2026-11-10)

Question: does Spotiflow (pretrained `general`) beat the LoG detector for
tracking on sg100_Well5_1018? Conclusion: no. LoG matched or beat it on every
recall-side tracking metric with 3-5x fewer detections. The GT and the tracker
settings were built around LoG output, so this was not a neutral comparison.

Contents: scripts/ (detection, sweeps, comparison, napari viewers), blobs/
(Spotiflow detection CSVs), results/ (summary tables and plots).

Not moved here: two Spotiflow tracking runs are in
../legacy_single_movie/tracking_runs/runs/ (notes "SF raw p0.3 bf10", "SF bgsub p0.3 bf10")
plus their rows in ../legacy_single_movie/tracking_runs/tracking_log.csv.
(Moved from the repo root during cleanup, 2026-11-19.)

To re-run: these scripts import preprocess_blob_det and traccuracy_compare
(both now in archive/exploration_2026-09/scripts/), and their
hard-coded BASE / blobs / tracking_runs paths point at the old layout. Easiest is to
check out tag pre-cleanup-2026-11-19, where everything is in its original place:
  PYTHONPATH=scripts python spotiflow_experiment/scripts/spotiflow_detect.py
