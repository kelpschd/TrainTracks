# Methods notes: state of the pipeline at cleanup (2026-11-19)

Frozen reference: git tag `pre-cleanup-2026-11-19` (commit 7bb84e0).
Sources: `scripts/track_flow_score.py`, `scripts/batch_rescore_tracks.py`,
`scripts/batch_pipeline_with_posthoc.py`, TODO.md "Batch state notes (2026-10-26)",
and the cotracking notes. Items marked CONFIRM could not be checked from the repo
alone (they depend on files in `tracking_batch_TrainTracks`).

## Which definitions produced the current `tracks_clean.geff`

The 32 `tracks_clean.geff` files were rewritten on 2026-10-26 (10:52-10:54) by
`batch_rescore_tracks.py --write-clean`, which loads `track_flow_score.py` as the
scorer. So the filters in effect are the scorer's:

| Setting | Value | Where |
|---|---|---|
| Minimum span kept | >= 5 frames | `MIN_SPAN_KEEP` |
| Minimum span to be scored | >= 5 frames | `MIN_SPAN_FOR_SCORE` |
| Cull rule | `turn_incoherence >= 1.4` rad, turn-only (drift dropped) | `TURN_FLAG_RADIANS` |
| Pause definition | step speed <= 0.5 px/frame is neutral | `PAUSE_SPEED_PX_PER_FRAME` |
| Min edges for turn stats | 2 | `MIN_EDGES_FOR_STATS` |

Scorer `turn_incoherence` = mean absolute signed turn angle (arctan2 of cross/dot)
between consecutive MOVING steps, i.e. `turn_per_moving_step`. The 1.4 rad cutoff was
calibrated against this version with `track_flow_score.py --gt` (11 GT tracks, all
unflagged; random-walk reference about 1.57 rad).

Upstream detection filter (intentional, decided 2026-10-26): `filter_dense_blobs`
clears any 5x5 px x/y bin with >= 5 detections over the movie
(`bin_size` / `blob_filter` in metadata). Known consequence: GT track 1088 pauses
7 frames in one bin, so the pipeline cannot reproduce it (counts against recall).
State this in the methods.

## Turn-definition mismatch: what it does and does not affect

`batch_pipeline_with_posthoc.py:777` computes its own `turn_incoherence` as the mean
unsigned angle over ALL steps (pauses included). That is a different quantity from the
scorer's. Because `tracks_clean.geff` was regenerated from the scorer, the final
cleaned tracks use the calibrated definition. What is still affected:

- the `posthoc_results` counts and `output_clean_geff` path stored in each movie's
  `metadata.json` (stale, from 2026-09-30), read by `batch_traccuracy.py` and
  `batch_track_review.py`;
- anyone re-running the batch pipeline end to end, which would apply the 1.4 cutoff to
  the other quantity.

So this is a reproducibility problem more than a problem with the current results.

## Track-ID handling

funtracks renumbers `tracklet_id` / `particle` on every import, and 63-bit hashes
collide in float64 (napari). Any join across tables must use the string `track_uid`.
Known unsafe: `straightness_review.py:178` (silent empty join in the batch layout),
and the `track_vis*` / `review_lyso_interaction.py` napari viewers.
CONFIRM that the cotracking pipeline (`batch_cotracking.py`,
`summarize_cotracking_velocity.py`) joins particle tracks to organelle episodes on
`track_uid` and not on `particle`. This matters for the headline cotracking numbers.

## Other facts to state or fix

- `NEAR_EDGE_DIST_PX` changed 3 -> 9 between the 13:00 and 15:56 runs on 2026-10-26;
  root-level organelle CSVs/params are from the later run.
- `*_label_persistence` is meaningless (per-frame relabelling) and `<org>_label` /
  `centroid_dist` come from the nearest centroid, not the containing object.
- GT is one movie, 11 hand-annotated tracks
  (`sg100_Well5_1018_valid_tracks.csv`), not pipeline output.
- Spotiflow comparison (archived 2026-11-10): LoG matched or beat Spotiflow on recall
  metrics with 3-5x fewer detections; not a neutral comparison because the GT and tracker
  settings were built around LoG output.

## Decision needed before Phase 3

| Item | Option A (recommended) | Option B |
|---|---|---|
| turn_incoherence | Make the scorer the single definition; the pipeline imports it. Re-run `--gt` calibration once in Phase 3, then regenerate `tracks_clean`. Document as a code fix, no change to results expected. | Leave as is, document as a known limitation. |
| Track IDs | Fix in Phase 3 as part of moving shared uid helpers into `src/`: sequential ID plus string `track_uid`, join on `track_uid` only. | Document only. |
| metadata.json staleness | `--write-clean` updates `posthoc_results` and `output_clean_geff` (small fix). | Regenerate metadata in the Phase 5 re-run. |
