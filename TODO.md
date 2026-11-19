# TrainTracks priority list

Ordered by impact on scientific results first, then reproducibility.
Items in a tier are roughly ordered; tiers are not.

## Tier 1 - decide or fix before trusting any more batch results

- [x] **`filter_dense_blobs` is intentional and stays.** (Decided 2026-10-26.)
      Bins on x/y across all frames; any 5x5 px bin with >= 5 detections over
      the movie is cleared. Same paradigm used to generate the tracks that were
      hand-annotated into the GT. Already recorded in metadata as
      `bin_size` / `blob_filter`.
      Open question: raise the threshold of 5? GT track 1088 pauses 7 frames
      in one bin, so at 5 the pipeline cannot reproduce that track and it
      counts against recall. Add a one-line note on this in the methods.

- [x] **Unify the `turn_incoherence` definition.** (Code done 2026-11-19, cleanup
      Phase 3.3.) One implementation now: `src/TrainTracks/scoring.py` (pause-filtered
      mean |signed turn| between consecutive moving steps; the definition the 1.4 rad
      cutoff was calibrated on). `batch_pipeline.py`, `batch_rescore_tracks.py` and
      `track_flow_score.py` all import it; the pipeline's own all-steps scorer and the
      three copies of `write_clean_geff` are gone. Pinned by `tests/test_scoring.py`.
      Verified locally 2026-11-19: `pytest tests` 19 passed; `track_flow_score.py --gt`
      leaves all 11 GT tracks unflagged (max GT turn 0.69 rad vs cutoff 1.4); full
      `batch_rescore_tracks.py` on all 32 movies gives 41268 tracks / 9425 eligible /
      6895 flagged, identical to before the refactor, movie by movie.
      Existing `tracks_clean.geff` files already came from this definition, so they do
      not need regenerating.

- [x] **Fix track-ID handling.** (Done 2026-11-19, cleanup Phase 3.2.)
      `particle` is now sequential 1..N (ranked by `track_uid`) in newly generated
      tracks (`batch_pipeline.solution_to_track_df`); `track_uid` is the only key for
      joins. `straightness_review.py` now joins flow_scores.csv on `track_uid`
      (it used to join on the import-local int `particle`, which silently matched
      nothing). Cotracking tables (`particle_tracks.csv`, `episodes_*.csv`,
      `steps_*.csv`, `track_speed.csv`) carry `track_uid` next to `particle`.
      The 63-bit hash IDs (`canonical_track_id`) are removed. The track_vis* viewers
      that passed IDs through float64 are archived.
      Existing batch outputs keep their old IDs until the Phase 5 re-run; their
      `track_uid` is unchanged, so nothing downstream breaks.

## Tier 2 - method quality (affects accuracy, not correctness)

- [ ] **Stop merging detections in the candidate graph.**
      Integer rounding + disk(5) dilation + connected-component labelling
      merges puncta within ~10 px and throws away sub-pixel positions.
      Build one node per blob at its float (y, x) and average flow in a disk
      around it directly. Re-run traccuracy vs GT before/after.
      File: `scripts/batch_pipeline.py:412-468`.

- [ ] **Drop the inert `node_selected_weight` parameter** (motile ignores the
      weight when `attribute=None`). Record instead the effective link radius
      `-edge_selected_constant / edge_selected_weight` so sweeps are readable.
      Note -84 and -96 are identical runs because the radius exceeds
      `max_edge_distance=30`.

- [ ] **Investigate why edge recall is 0.69 when the max_gap=2 ceiling is 0.95.**
      Only 8 of 157 GT links span > 2 frames, so most misses are detection
      drop-outs or the density filter, not gap closing. Check after Tier 1.

- [x] **Expand ground truth beyond 11 tracks / 1 movie.** (Done 2026-11-19, cleanup Phase 3.)
      `ground_truth/` now holds the validated tracks of all 32 movies (294 tracks, 4040
      detections; old pipeline + manual validation, positives only; see its README).
      `batch_traccuracy.py`, `batch_track_review.py` and `straightness_review.py` default
      to it (their old default pointed at `tracking_results/`, since renamed).
      `scripts/calibrate_cull_on_gt.py` checks the turn cutoff on every movie.
- [ ] **Re-run `batch_traccuracy.py` on the CURRENT `tracks_clean.geff`.** The stored
      `traccuracy/` results are from 2026-09-30, i.e. the old pipeline's clean tracks (before
      the 10-26 rebuild with the calibrated scorer). They gave node recall 0.78 raw / 0.48
      clean over the 32 movies (precision not meaningful, GT is positives only), which no
      longer describes the final tracks.

## Tier 3 - reproducibility and structure

- [x] **Delete the old `scripts/batch_pipeline.py`** (was a strict subset of the
      posthoc version, zero unique lines); `batch_pipeline_with_posthoc.py` was
      renamed to `batch_pipeline.py`. (Done 2026-11-19, cleanup Phase 1.)

- [ ] **Move shared code into `src/TrainTracks/`** and import it everywhere
      (3.1 done 2026-11-19: `ids.py` = track_uid/canonical-ID helpers, `tracks_io.py` =
      `geff_to_track_df`; callers updated; tests in `tests/`. Still to move: scorer,
      3.4a done + verified 2026-11-19: `detection.py` (identical blobs.csv on 1018);
      3.4b done + verified 2026-11-19: `flow.py` (flow_raw bit-identical on 1018);
      3.4c done + verified 2026-11-19: `tracking.py` (build_candidate_graph, edges, solve,
      metrics, solution_to_track_df, run_tracking; re-solve on 1018 gives the identical
      3205 tracks); check with `scripts/check_regression.py
      detection|flow|tracking <movie_dir>`. `preprocess_blob_det.py` and `optical_flow.py`
      archived. Remaining shared code is config/paths, not functions):
      `geff_to_track_df` (5 copies), `score_track`/`score_all` (2),
      `track_uid_for_group`/`canonical_track_id` (3), `build_candidate_graph`,
      `preprocess_stack`, `compute_optical_flow`, `track_df_to_nx`.
      This is what makes the Tier 1 turn-definition drift impossible to repeat.

- [ ] **Version-control the ground truth and run index.**
      `.gitignore` excludes `*.csv` and `metadata.json`.
      `git add -f sg100_Well5_1018_valid_tracks.csv tracking_runs/tracking_log.csv`
      and consider un-ignoring `metadata.json`.

- [ ] **Pin the environment.** Only funtracks is pinned; working versions are
      motile 1.0.1, geff 1.3.0, traccuracy 0.4.3, scikit-image 0.26, numpy 2.4,
      zarr 3.1.6, opencv 5.0. Commit a `conda env export --from-history` plus a
      lock file.

- [ ] **Replace hard-coded paths/run IDs with CLI args or one config file.**
      (3.5a done 2026-11-19: `TrainTracks.config` (raw_root/results_root via CLI, env or
      `TrainTracks.toml`); `batch_pipeline.py` uses it for default input/output and writes
      relative paths in metadata.json; consumers (`resolve_raw_zarr`, `batch_track_review`,
      `straightness_review`) read relative or old absolute paths; `track_flow_score.py` no
      longer hard-codes the repo path. Still open: `batch_root` positional defaults for the
      other batch scripts, legacy single-run viewers are archived.)
      Worst offenders: `candidate_graph.py`, `track_flow_score.py`,
      `preprocess_blob_det.py`, `optical_flow.py`, all `track_vis*.py`
      (`IMG_NUM = 9` selects the movie by glob order), `flow_vis.py`,
      `track_vis_extra.py` (stale `/Users/dankelpsch/` paths).

- [ ] **Add one end-to-end smoke test** on a small synthetic movie with known
      tracks (detect -> flow -> solve -> score -> clean -> traccuracy).
      Would have caught every Tier 1 item.

## Batch state notes (2026-10-26)

- All 32 `tracks_clean.geff` in `tracking_batch_TrainTracks` were rewritten
  10:52-10:54 by `batch_rescore_tracks.py --write-clean` -> filters in effect
  are span >= 5 and scorer (pause-filtered) turn >= 1.4. Consistent across
  the batch.
- [ ] `batch_rescore_tracks.py --write-clean` should update `metadata.json`
      (`posthoc_results`, `output_clean_geff`) so `batch_traccuracy.py` and
      `batch_track_review.py` stop reading the stale Sep 30 counts/paths.
- [ ] Join `tracks_organelle.csv` <-> `flow_scores.csv` on `track_uid` only;
      `particle` is renumbered by funtracks on every import.
- Known / handled downstream (no action): `*_label_persistence` is
  meaningless (per-frame relabelling); `<org>_label` / `centroid_dist` come
  from nearest centroid, not the object the point is inside.
- `NEAR_EDGE_DIST_PX` changed 3 -> 9 between the 13:00 and 15:56 runs; the
  root-level organelle CSVs/params are overwritten by the later run.

## Tier 4 - cleanup

- [ ] `tracking.build_candidate_graph` calls `skimage.morphology.binary_dilation`, deprecated in
      scikit-image 0.26 and removed in 0.28 (FutureWarning seen 2026-11-19). `disk` is
      symmetric, so `skimage.morphology.dilation` should be equivalent; swap it and confirm
      with `check_regression.py tracking`. Until then keep scikit-image < 0.28 in the pin.
- [ ] `compute_optical_flow` hard-codes channel 0; use `blob_channel`.
- [ ] Legacy `flow.zarr/flow_raw` is (T, ...) with a zero last frame; new
      pipeline writes (T-1, ...). Regenerate the legacy one or note it.
- [ ] `batch_rescore_tracks.py` takes `min_span_keep` from the scorer module
      while `batch_track_review.py` reads it from metadata; use metadata.
- [x] (test_napari.py deleted 2026-11-19) Remaining: remove dead `assign_canonical_track_ids`
      in `batch_track_review.py` and dead `track_df_to_nx` in
      `track_vis_thresholds.py`.
- [ ] Fix stale labels: `traccuracy_runs_EdgeConstantSweep.py` keys say `pen*`
      for an edge-constant sweep; `sweep_edge_constant.py:22` comment.
- [ ] Fill in `README.md` / `scripts/README.md` with the actual run order:
      import_export -> preprocess_blob_det -> optical_flow -> candidate_graph
      (or batch_pipeline) -> track_flow_score -> batch_traccuracy ->
      batch_lyso_interaction.
