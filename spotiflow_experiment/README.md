# Spotiflow vs LoG experiment (archived 2026-11-10)

Question: does Spotiflow (pretrained `general`) beat the LoG detector for
tracking on sg100_Well5_1018? Conclusion: no. LoG matched or beat it on every
recall-side tracking metric with 3-5x fewer detections. The GT and the tracker
settings were built around LoG output, so this was not a neutral comparison.

Contents: scripts/ (detection, sweeps, comparison, napari viewers), blobs/
(Spotiflow detection CSVs), results/ (summary tables and plots).

Not moved: two Spotiflow tracking runs remain in tracking_runs/runs/ (notes
"SF raw p0.3 bf10", "SF bgsub p0.3 bf10") plus their rows in tracking_log.csv.

To re-run: these scripts import preprocess_blob_det / traccuracy_compare from
the main scripts/ folder, so run with that folder on PYTHONPATH, e.g.
  PYTHONPATH=scripts python spotiflow_experiment/scripts/spotiflow_detect.py
