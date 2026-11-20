# TrainTracks

Track fast-moving puncta in live-cell movies, clean the tracks automatically, and analyse how they
interact with lysosomes and mitochondria. Raw data are 4-channel ND2 movies (channel 0 = puncta,
1 = lysosome, 2 = mitochondria, 3 = BFP, ignored). The pipeline detects blobs, estimates optical
flow, links detections into tracks with an integer program (motile), scores every track against the
flow, and removes short and erratic tracks. Downstream scripts measure organelle contact and test
whether puncta travel with organelles more often than chance (cotracking).

## Getting started

```bash
conda create -n TrainTracks python=3.11
conda activate TrainTracks
pip install -e ".[test]"                # the dependencies are declared in pyproject.toml
pytest tests                            # all tests should pass
```

To reproduce the exact tested versions instead of the loose ones in `pyproject.toml`, add
`pip install -r requirements-lock.txt` and then `pip install -e . --no-deps` (read the header of
`requirements-lock.txt` first: it is a full freeze of the working environment).

Python must be 3.11 (`requires-python = ">=3.11,<3.12"`); the base conda environment will not install it.

Tested with: numpy 2.4.6, pandas 3.0.5, zarr 3.1.6, scipy 1.17.1, scikit-image 0.26.0, opencv-python
5.0.0.93, networkx 3.6.1, motile 1.0.1, motile-tracker 5.0.1, funtracks 2.0.2, geff 1.3.0.1.2, nd2
0.11.3, napari 0.7.1. `pyproject.toml` keeps scikit-image below 0.28: two functions the pipeline uses are deprecated in 0.26
and the replacements are not drop-in (see `TODO.md`).

## Where the data live

Data stay outside the repo, in two folders side by side under the dataset root:

```
<dataset>/raw_data/   the .nd2 files and their .zarr conversions (nothing else)
<dataset>/results/    every pipeline output (zarr and csv), one folder per movie
```

Tell the code where they are, in order of priority: CLI flags, the environment variables
`TRAINTRACKS_RAW_ROOT` / `TRAINTRACKS_RESULTS_ROOT`, or `TrainTracks.toml` (copy
`TrainTracks.example.toml`; the local file is gitignored). The same file holds optional resource
profiles (movies in parallel, threads per movie). Paths stored in each movie's `metadata.json` are relative
to these roots, so moving or renaming a folder never leaves stale absolute paths.

## Running the pipeline

The commands below assume `TrainTracks.toml` is set up and `DS` points at the dataset root
(`export DS=/path/to/dataset`). Run them from the repo root.

| Step | Command | Writes |
|---|---|---|
| 0. Convert (once) | `python scripts/import_export.py "$DS/raw_data"` | `<movie>.zarr` next to each `.nd2` |
| 1. Track | `python scripts/batch_pipeline.py` | per movie: `blobs.csv`, `flow.zarr`, `tracks.geff`, `flow_scores.csv`, `tracks_clean.geff`, `metadata.json`; `batch_summary.csv` |
| 2. Validate | `python scripts/batch_traccuracy.py "$DS/results"` | `traccuracy/` (recall against the ground truth) |
| 3. Organelle contact | `python scripts/batch_lyso_interaction.py "$DS/results" --raw-root "$DS/raw_data"` | per movie `organelle/`; `organelle_interaction_{detections,tracks}.csv` |
| 4. Colocalization | `python scripts/batch_puncta_colocalization.py "$DS/results" --raw-root "$DS/raw_data"` | per movie `coloc/`; `organelle_coloc_{detections,tracks}.csv` |
| 5. Cotracking | `python cotracking/batch_cotracking.py "$DS/results" --raw-root "$DS/raw_data"` | `cotracking/<movie>/...` |
| 6. Pool cotracking | `python cotracking/summarize_cotracking_velocity.py "$DS/results/cotracking"` | `cotracking/summary/` tables and figure |

Refresh `batch_summary.csv` from the per-movie metadata at any time with
`python scripts/batch_pipeline.py --summarize-only`. Every step is resumable: `batch_pipeline.py` and
`batch_lyso_interaction.py` accept `--skip-existing`, and the pipeline writes each movie to
`<movie>.partial` and renames it only when complete, so a killed run never leaves a half-written movie.
A movie folder is complete if and only if it contains `metadata.json`.

Speed: step 1 sizes itself from the machine (movies at once, threads per movie; override with `--jobs`,
`--threads-per-job` or `--profile NAME`). The full 32-movie run took about 18 minutes with 4 movies at once on a
12-core Mac with 32 GB, against 98 minutes one movie at a time in the original September run (before the
candidate-graph speed-up). `batch_lyso_interaction.py` also takes `--jobs N`. Cluster (SLURM array) use and
measured memory and timings are in `docs/parallel_and_hpc.md`.

## What the pipeline does and the settings it uses

1. **Detection.** Background-subtract channel 0, Laplacian-of-Gaussian blobs (threshold 0.00017,
   sigma 1 to 3), then discard every blob in a 5 x 5 px bin that holds 5 or more detections anywhere in the
   movie (`filter_dense_blobs`, intentional). A real particle that pauses in one bin for 5 or more frames is lost.
2. **Optical flow.** CLAHE then Farneback flow between consecutive frames, smoothed with the previous
   frame's flow (weight 0.7); T-1 flow fields for T frames.
3. **Tracking.** One candidate node per merged blob region (disk dilation radius 5), candidate edges up to
   2 frames apart and 30 px per frame of gap, edge cost from the distance to the flow-predicted position
   (3 x offset - 72), at most one parent and one child, solved with motile.
4. **Cleaning.** A track is scored by `turn_incoherence`, the mean absolute turn between consecutive moving
   steps (steps at or below 0.5 px/frame count as pauses and are ignored). Tracks of at least 5 frames
   with `turn_incoherence >= 1.4` rad are flagged; flagged tracks and tracks shorter than 5 frames are
   removed, giving `tracks_clean.geff`. The cutoff was calibrated on the validated tracks; see
   `docs/methods_notes.md` for the numbers (about 1.7% of validated tracks fall above it).

All parameters are stored in each movie's `metadata.json`, together with counts, per-stage timings and
peak memory. A track's identity across files is `track_uid` (a hash of its frame and position sequence); the
integer `particle` is only meaningful within one file, so always join tables on `track_uid`.

## Checking your work

| What | Command |
|---|---|
| Unit and smoke tests | `pytest tests` |
| Re-run one stage and compare with stored output | `python scripts/check_regression.py detection\|flow\|tracking <results>/<movie>` |
| Compare two complete runs (archived vs fresh) | `python scripts/compare_runs.py <old_root> <new_root>` |
| Is the cull cutoff safe on the validated tracks? | `python scripts/calibrate_cull_on_gt.py "$DS/results"` |
| Recall against the validated tracks | `python scripts/batch_traccuracy.py "$DS/results"` (step 2) |

`ground_truth/` holds validated tracks for all 32 movies (positives only; read its README before using it).
A fresh run of the full pipeline reproduces the archived September run exactly (blobs, flow bit for bit, tracks,
clean tracks, flow scores) and the downstream tables and cotracking summaries.

## Repository layout

```
src/TrainTracks/   importable code: ids, tracks_io, detection, flow, tracking, scoring, calibration,
                   compare, config, resources, parallel
scripts/           command-line entry points (the table above) plus review and calibration tools
cotracking/        cotracking analysis: cotracking.py, organelle_tracks.py, batch_cotracking.py, summaries
analysis/          the R/Quarto analysis notebook
ground_truth/      validated tracks, one csv per movie
tests/             pytest suite
docs/              methods_notes.md (settings, calibration, findings), parallel_and_hpc.md
archive/           retired exploration code and data, each folder with a README
TODO.md            open items;  CLEANUP_PLAN.md  how the repo was reorganised
```

Review tools (napari): `scripts/batch_track_review.py`, `review_lyso_interaction.py`,
`straightness_review.py`.

## Notes

- Always use `track_uid`, not `particle`, to follow a track between files.
- `docs/methods_notes.md` is the place to look for what each number means and which analyses were retired.
- Open items and known limitations are in `TODO.md`.
