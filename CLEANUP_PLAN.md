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
    legacy_single_movie/ blobs/, flow.zarr, test_run_7.geff, root traccuracy CSVs, tracking_runs/
  TODO.md                tracked (remove from .gitignore)
```

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
Move the exploration scripts, legacy `blobs/`, `flow.zarr`, `test_run_7.geff`, root
`traccuracy_*.csv`/`vs_gt_metrics.png` and `tracking_runs/` into `archive/` with a
one-paragraph README each (what question it answered, date, how to re-run), following
the style of `spotiflow_experiment/README.md`. Keep the two Spotiflow runs note from
that README in sync when `tracking_runs/` moves. Check `flow.zarr` (1.2 GB) is not
needed by anything in the final pipeline before deciding whether to keep it locally,
move to external storage, or delete.

**Phase 3 - Consolidate shared code into `src/TrainTracks/`.**
Extract in this order, running the existing batch on one movie after each step and
diffing outputs: `geff_to_track_df` and uid helpers, scoring (one turn definition),
`build_candidate_graph`/`preprocess_stack`/`compute_optical_flow`, then the cotracking
modules. Keep the cotracking segmentation, `filter_dense_blobs`, Kalman linking and
net-displacement filter byte-for-byte equivalent (your standing decision). Move the
numbered CLI wrappers into `pipeline/` and replace hard-coded paths and `IMG_NUM = 9`
with arguments or one config file. Write pure functions (DataFrame in/out) so the
napari plugin can wrap them later.

**Phase 4 - Reproducibility.**
`environment.yml` plus a lock file with the versions listed in TODO.md, add a smoke
test on a tiny synthetic movie, write the real README run order, and make the
parameters used for the paper (thresholds, bin size, gap, cutoffs) a single
`params.yaml` that the pipeline writes into every output's metadata.

**Phase 5 - Paper-ready results.**
Re-run the full 32-movie batch from the cleaned code, regenerate the tracks_clean
files, the cotracking tables and `analysis.qmd` figures into `results/`, and confirm
the headline numbers match the current ones (lysosome cotracking 7.6% vs 0.9%,
p = 0.029; 2903 tracks). Tag `v1.0-paper-analysis`.

## Verification at each phase
Run one movie through the pipeline before and after and compare `tracks_clean`
node/edge counts and the cotracking summary; `git diff --stat` the move commits to
confirm renames only; grep for the old script names and `/Users/dankelpsch/` paths.

## Open questions for you
1. Where do `tracking_batch_TrainTracks` (the 32 movies) and `analysis_20261026.qmd`
   live? They are outside this folder, so the plan assumes they stay put and get a
   `data_paths` config entry.
2. Is `flow.zarr` (1.2 GB legacy single-movie flow) still needed?
3. Do you want the cleanup done on a branch (recommended) or directly on main?
