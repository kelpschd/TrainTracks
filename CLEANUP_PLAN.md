# TrainTracks directory cleanup plan (drafted 2026-11-19)

Nothing has been moved or deleted. This is a proposal to review. It is based on a
listing of the folder, the READMEs, TODO.md, git history and script headers.

## What I found

- Root is cluttered: loose result files (`vs_gt_metrics.png`, `traccuracy_*.csv` x4,
  `sg100_Well5_1018_valid_tracks.csv`), a stray `test_run_7.geff`, a legacy `blobs/`
  and `flow.zarr/` (1.2 GB, 99% of the folder), plus `.DS_Store` files.
- `scripts/` mixes four generations of work: early exploration (Jul-Sep: `track_vis*`,
  `sweep_*`, `traccuracy_compare`, `flow_vis`, `test_napari`), the batch pipeline
  (Sep-Oct), analysis (Oct: `batch_lyso_interaction`, `batch_puncta_colocalization`,
  `straightness_review`), and cotracking (Nov, in its own `cotracking/` folder).
- Duplicated code: `batch_pipeline.py` is a strict subset of
  `batch_pipeline_with_posthoc.py`; `geff_to_track_df` exists in 5 copies and the
  scorer in 2. `src/TrainTracks/` is empty (`__init__.py` only), so nothing is importable.
- `.gitignore` excludes `*.csv`, `*.png`, `*.geff` and `TODO.md`, so the ground truth,
  the run index and the TODO are not version-controlled. One tracked notebook is
  modified and uncommitted.
- `notebooks/` and `scripts/` READMEs are template placeholders; the root README has
  a typo (`conda activate 3.11`) and no run order.
- Two Tier 1 items in TODO.md are still open and change results (turn_incoherence
  definition, track-ID handling). Cleanup should not hide these; see Phase 0.

## Target layout

```
TrainTracks/
  README.md              run order, environment, data locations
  pyproject.toml, environment.yml (+ lock file)
  CLEANUP_PLAN.md -> delete when done
  src/TrainTracks/       shared, importable code (no hard-coded paths)
    io.py                geff_to_track_df, track_df_to_nx, uid helpers
    detection.py         preprocess_stack, blob detection
    flow.py              compute_optical_flow
    graph.py             build_candidate_graph
    scoring.py           score_track / score_all  (single turn definition)
    cotracking/          organelle_tracks, cotracking, summaries
  pipeline/              thin CLI entry points, in run order
    01_import_export.py  02_detect.py  03_flow.py  04_track.py
    05_score_clean.py    06_validate_traccuracy.py  07_organelle.py  08_cotracking.py
  analysis/              analysis_20261026.qmd -> analysis.qmd, cotracking_section.qmd
  review/                napari/diagnostic viewers (batch_track_review, review_lyso_interaction, ...)
  ground_truth/          sg100_Well5_1018_valid_tracks.csv + short README (11 tracks, hand-annotated)
  tests/                 one synthetic end-to-end smoke test
  results/               small, citable summary tables/figures only (tracked)
  archive/
    exploration_2026-09/ track_vis*, sweep_*, traccuracy_compare, flow_vis, vs_gt_matplotlib ...
    spotiflow_experiment/  (already self-contained; just move)
    legacy_single_movie/ blobs/, test_run_7.geff, root traccuracy CSVs, tracking_runs/
  (flow.zarr: goes to <results root>/sg100_Well5_1018/archived_flow/, outside the repo)
  TODO.md                tracked (remove from .gitignore)
```

## Status
Phase 0, 1, 2 done (2026-11-19). Phase 3 in progress:
- 3.1 done: `src/TrainTracks/ids.py` and `tracks_io.py` replace 5 copies of
  `geff_to_track_df` and 3 of the uid helpers (behaviour-identical; see tests/).
- 3.2 done: sequential `particle` IDs, `track_uid` joins (straightness_review fixed),
  `track_uid` carried through cotracking tables.
- 3.3 done: `src/TrainTracks/scoring.py` is the only scorer / turn_incoherence /
  clean-track rule; pipeline, rescore and track_flow_score import it.
- 3.4 done + verified on sg100_Well5_1018: detection (blobs.csv identical), flow (bit-identical),
  tracking (identical 3205 tracks); `src/TrainTracks/{detection,flow,tracking,scoring,ids,tracks_io}.py`;
  `scripts/check_regression.py detection|flow|tracking` is the safety net. 28 tests pass.
- 3.5a done in code (awaiting local check): `src/TrainTracks/config.py` (raw_root/results_root via CLI,
  env or `TrainTracks.toml`); results folder is named `results` (decided); batch_pipeline writes relative
  paths in metadata.json; consumers read relative or old absolute.
- 3.5a verified (38 tests pass). GT expansion done in code (awaiting local run): `ground_truth/` (32 movies),
  `scripts/calibrate_cull_on_gt.py`, stale GT-root defaults fixed.
- Next: `batch_root` defaults for the other batch scripts, then Phase 3b migration, then Phase 4/5.

## Phases (do in order; commit after each)

**Phase 0 - Freeze the science first (small, but do before moving anything).**
Tag the current commit (`git tag pre-cleanup-2026-11-19`) and commit the pending
notebook change. Record in a `ground_truth/README.md` or `docs/methods_notes.md` which
turn_incoherence definition produced the current `tracks_clean.geff` (scorer version,
pause-filtered, cutoff 1.4, span >= 5, per the 2026-10-26 batch notes) and that
`batch_rescore_tracks.py --write-clean` has not updated `metadata.json`. Decide whether
to fix the two open Tier 1 items now or log them as known limitations for the paper.

**Phase 1 - Safe housekeeping (no behavior change).**
Delete `.DS_Store`, `__pycache__`, `cotracking/.Rhistory`, `scripts/.claude` (stray
copy). Delete `scripts/test_napari.py`. Delete `batch_pipeline.py` and rename
`batch_pipeline_with_posthoc.py` to `batch_pipeline.py`. Fix `.gitignore`: add
`.DS_Store`, `.Rhistory`; stop ignoring `TODO.md`, the GT CSV and `tracking_log.csv`
(or `git add -f` them). Fix the README typo.

**Phase 2 - Archive, don't delete.**
Move the exploration scripts, legacy `blobs/`, `test_run_7.geff`, root
`traccuracy_*.csv`/`vs_gt_metrics.png` and `tracking_runs/` into `archive/` with a
one-paragraph README each (what question it answered, date, how to re-run), following
the style of `spotiflow_experiment/README.md`. Keep the two Spotiflow runs note from
that README in sync when `tracking_runs/` moves. `flow.zarr` is left in place for now
(`--gt` calibration still reads it) and moves to `archived_flow/` (see Decision 2).

**Phase 3 - Consolidate shared code into `src/TrainTracks/`.**
Extract in this order, running the existing batch on one movie after each step and
diffing outputs: `geff_to_track_df` and uid helpers, scoring (one turn definition),
`build_candidate_graph`/`preprocess_stack`/`compute_optical_flow`, then the cotracking
modules. Keep the cotracking segmentation, `filter_dense_blobs`, Kalman linking and
net-displacement filter byte-for-byte equivalent (your standing decision). Move the
numbered CLI wrappers into `pipeline/` and replace hard-coded paths and `IMG_NUM = 9`
with arguments or one config file. Write pure functions (DataFrame in/out) so the
napari plugin can wrap them later.

**Phase 3b - Fresh run into `DS/results/` (replaces "migrate the old outputs"; approved 2026-11-19).**
The old outputs are NOT moved. A move would not look like a fresh run (old `metadata.json` has
absolute, partly stale paths and stale `posthoc_results`; the geffs carry the old 63-bit IDs; files
come from different code versions and dates), so instead:
1. Rename `DS/tracking_batch_TrainTracks` to `..._archived_2026-11-19` (a rename: nothing copied or
   deleted; `raw_data/` untouched). Copy `analysis_20261026.qmd` into the repo's `analysis/`.
2. Create `TrainTracks.toml` (`raw_root`, `results_root = DS/results`, optional `[resources]`).
3. Dry run on one movie into a scratch folder (`batch_pipeline.py --movie X --output-dir /tmp/s`),
   then `compare_runs.py <archive> /tmp/s --movie X`. This is the first end-to-end run of
   `process_movie` since the refactor. Record `timings_seconds` and `peak_rss_gb` from its metadata.
4. Full run into `DS/results/` (`--jobs auto` or a profile; resumable with `--skip-existing`), then
   `compare_runs.py <archive> DS/results` over all 32 movies. A `--jobs 1` vs `--jobs N` comparison on a
   few movies confirms parallelism does not change results.
5. Downstream scripts into `DS/results/`, in this order: `batch_traccuracy.py`, `batch_lyso_interaction.py`,
   `batch_puncta_colocalization.py`, `batch_cotracking.py` (full sorted run: its permutation seeds are
   `1000 + movie index`), then `summarize_cotracking_velocity.py`, `length_matched_speed.py`,
   `leading_trailing.py`. Give the other batch scripts a `batch_root` default of `results_root` here.
6. Compare the headline numbers with the archive (2903 clean tracks; 41268 / 9425 / 6895 tracks /
   eligible / flagged; cotracking 7.6% vs 0.9%, p = 0.029), regenerate figures from `analysis/`.
`results/` is final only when 3 to 6 agree; the archive stays untouched as the record.

Speed: the September run took 98 min serial (83 s to 352 s per movie; time grows with blob count).
`batch_pipeline.py` now supports `--jobs N|auto`, `--profile`, `--threads-per-job`, `--movie`,
`--skip-existing`, SLURM job-array sharding and atomic per-movie output (see docs/parallel_and_hpc.md).
Candidate speed-ups to profile first: building the candidate graph makes a full-frame mask per blob
region (a bit-identical rewrite from region coordinates is likely much faster).

**Phase 4 - Reproducibility.**
`environment.yml` plus a lock file with the versions listed in TODO.md, add a smoke
test on a tiny synthetic movie, write the real README run order, and make the
parameters used for the paper (thresholds, bin size, gap, cutoffs) a single
`params.yaml` that the pipeline writes into every output's metadata.

**Phase 5 - Paper-ready results.** (Largely done by Phase 3b step 6: the fresh run IS the re-run.)
Tag the commit that produced `DS/results/` (`v1.0-paper-analysis`) and keep `TrainTracks.toml` params in the methods.

## Verification at each phase
Run one movie through the pipeline before and after and compare `tracks_clean`
node/edge counts and the cotracking summary; `git diff --stat` the move commits to
confirm renames only; grep for the old script names and `/Users/dankelpsch/` paths.

## Decisions (2026-11-19)
1. Batch outputs (`tracking_batch_TrainTracks`) and `analysis_20261026.qmd` live
   outside this repo, next to the raw data. They stay there; the repo gets a config
   entry (`data_root` / `results_root`) instead of hard-coded paths.
2. The repo's `flow.zarr` is one of the earliest flow runs for sg100_Well5_1018, kept
   as an archive, not deleted. It is the legacy (33, Y, X, 2) array (one frame per
   image frame, last frame zero) plus `flow_frames_XY` visualisation frames. The batch
   pipeline writes (T-1, Y, X, 2), one flow field per frame transition, for whatever
   T the movie has (33 frames gives 32). Destination:
   `<results root>/sg100_Well5_1018/archived_flow/flow.zarr`. Until it moves, the legacy
   scripts that read it (`track_flow_score.py` incl. `--gt`, `candidate_graph.py`,
   `flow_vis.py`, `track_vis*`) keep working; in Phase 3 `--gt` calibration is repointed
   at the batch flow. It is gitignored, so git has no copy.
3. Cleanup work happens on branch `cleanup/2026-11`; main keeps the pre-cleanup tag.
4. turn_incoherence and track-ID fixes are done in Phase 3 (see docs/methods_notes.md).
5. Data layout (decided 2026-11-19). Under the dataset root `DS`
   (`.../Data/Sphere/220725_i11w-hT-M33-I76_sg1035_d10sphere`) there are two sibling
   folders:
   - `DS/raw_data/`: the `.nd2` files AND their `.zarr` conversions (written by
     `import_export.py`, which already writes next to the `.nd2`). Nothing else.
   - `DS/results/` (new, clean; name decided 2026-11-19): EVERY pipeline output,
     both zarr and csv: per-movie blobs, flow.zarr, tracks / tracks_clean geff,
     flow_scores, organelle and coloc tables, cotracking tables, per-movie
     metadata.json, plus batch-level summaries, traccuracy outputs and figures.
   The repo reads and writes only through two config values, `raw_root` and
   `results_root`; no pipeline step may write into `raw_data/`, except the nd2->zarr
   conversion. Claude currently has read-only access to `DS`, so creating the folder
   and moving the existing `tracking_batch_TrainTracks` contents is a step you do (or
   explicitly grant write access for); see Phase 3b.
6. Ground truth can be expanded now. `DS/tracking_results_old/` has a
   `<movie>_valid_tracks.csv` for all 32 movies (previous pipeline + manual validation):
   294 validated tracks, 4040 rows, vs 11 tracks in the repo GT. The repo GT has the same
   row/track counts as the 1018 file (168 rows, 11 tracks) with columns trimmed to
   `frame,x,y,particle`, so it is a subset of this set. Caveats for the methods: these are
   validated positives picked from the old pipeline's output (not exhaustive annotation),
   so they support recall and "would the cull remove a known-good track" checks, not
   precision. Add a Phase 3 step: copy them to `ground_truth/` (trimmed columns, one
   file per movie, plus a README on provenance), then re-run the `--gt` calibration and
   traccuracy recall across all 32 movies. This also addresses TODO Tier 2 "expand
   ground truth".
