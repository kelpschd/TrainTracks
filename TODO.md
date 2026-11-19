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

- [ ] **Unify the `turn_incoherence` definition.**
      Batch pipeline = mean angle over all steps. `track_flow_score.py` =
      pause-filtered (speed > 0.5 px/frame only), signed arctan2.
      The 1.4 rad cutoff was calibrated with the scorer version, so the batch
      pipeline applies it to a different quantity.
      Pick one, make the other import it, then re-run `--gt` calibration and
      regenerate `tracks_clean.geff` for every movie.
      Files: `scripts/batch_pipeline.py:777`,
      `scripts/track_flow_score.py:316`.

- [ ] **Fix track-ID handling.**
      funtracks renumbers `tracklet_id` on import, and 63-bit hashes collide in
      float64 (napari). Use a <= 31-bit hash or sequential IDs plus a separate
      `track_uid` string column. Then:
      - `straightness_review.py:178` joins by int particle (currently an
        empty join in the batch layout, silent).
      - `track_vis.py`, `track_vis_FrameSweep.py`, `track_vis_thresholds.py`,
        `review_lyso_interaction.py` pass IDs through float64 into napari.
      - Keep `batch_track_review.py`'s uid-join pattern as the model.

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

- [ ] **Expand ground truth beyond 11 tracks / 1 movie** before relying on the
      cull threshold. Even 2-3 more validated movies changes the calibration
      from anecdote to estimate.

## Tier 3 - reproducibility and structure

- [x] **Delete the old `scripts/batch_pipeline.py`** (was a strict subset of the
      posthoc version, zero unique lines); `batch_pipeline_with_posthoc.py` was
      renamed to `batch_pipeline.py`. (Done 2026-11-19, cleanup Phase 1.)

- [ ] **Move shared code into `src/TrainTracks/`** and import it everywhere:
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
