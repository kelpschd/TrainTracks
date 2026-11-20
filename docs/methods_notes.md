# Methods notes: state of the pipeline at cleanup (2026-11-19)

Frozen reference: git tag `pre-cleanup-2026-11-19` (commit 7bb84e0).
Sources: `scripts/track_flow_score.py`, `scripts/batch_rescore_tracks.py`,
`scripts/batch_pipeline.py`, TODO.md "Batch state notes (2026-10-26)",
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

## turn_incoherence (unified in cleanup Phase 3.3, 2026-11-19)

There is now a single implementation, `TrainTracks.scoring`. Before, `batch_pipeline.py`
computed its own `turn_incoherence` as the mean unsigned angle over ALL steps (pauses
included), while the scorer (`track_flow_score.py`, used by `batch_rescore_tracks.py`
to write the current `tracks_clean.geff`) used the pause-filtered version the 1.4 rad
cutoff was calibrated on. The current cleaned tracks were always from the scorer, so
results are unaffected; the old pipeline definition only affected the stale September
`posthoc_results` in `metadata.json` (for sg100_Well5_1018: 557 flagged / 788 eligible
then, versus 550 / 780 with the scorer) and would have affected a fresh end-to-end run.
New pipeline runs record `turn_definition` and `pause_speed_px_per_frame` in
`posthoc_params`.

## Track-ID handling (fixed in cleanup Phase 3.2, 2026-11-19)

funtracks renumbers `tracklet_id` / `particle` on every import, and 63-bit hashes
collide in float64 (napari). The stable key is `track_uid`, a SHA-1 of the track's
frame/position sequence (positions rounded to 1e-6 px); all joins across tables use it.
Newly generated tracks get a sequential integer `particle` ranked by `track_uid`.

Cotracking checked (code read, not re-run): the headline numbers were never affected.
`batch_cotracking.py` imports `tracks_clean.geff` once per movie, writes
`particle_tracks.csv` in that run, and every downstream script reads that CSV plus
episodes written alongside it, so `particle` is only matched within one import.
The tables now also carry `track_uid`, so results can be traced to a specific track in
`tracks_clean.geff`. Tables written before 2026-11-19 lack this column; regenerate
with `batch_cotracking.py` (the analysis itself is unchanged).

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
| turn_incoherence | DONE (Phase 3.3): the scorer is the single definition, imported by the pipeline. `--gt` calibration to be re-run locally; no change to results expected. | Leave as is, document as a known limitation. |
| Track IDs | DONE (Phase 3.2): sequential ID plus string `track_uid`, join on `track_uid` only. | Document only. |
| metadata.json staleness | `--write-clean` updates `posthoc_results` and `output_clean_geff` (small fix). | Regenerate metadata in the Phase 5 re-run. |

## Ground truth (expanded 2026-11-19)

`ground_truth/` holds validated tracks for all 32 movies (294 tracks, 4040 detections), copied
from the previous pipeline's manually validated output. Positives only and selected through
the old pipeline's filters, so use it for recall and for checking that the cull does not remove
known-good tracks; precision against it is not meaningful.

traccuracy against this GT (`batch_traccuracy.py`, match threshold 15 px, pooled over the 32
movies, re-run 2026-11-19 on the current `tracks_clean.geff`; identical to the 2026-09-30 run):
raw tracks node recall 0.779 (3147/4040), edge recall 0.644 (2412/3746), 41268 tracks; clean tracks
node recall 0.484 (1955/4040), edge recall 0.432 (1618/3746), 2903 tracks. Precision is not
meaningful (positives-only GT).

Cull calibration over all 32 movies (`calibrate_cull_on_gt.py`, 294 GT tracks, all eligible):
turn_incoherence of GT tracks has median 0.36, 90th percentile 0.93, 99th 1.41, maximum 1.71 rad.
At the 1.4 cutoff 5 GT tracks (1.7%) are flagged (turns 1.71, 1.53, 1.48, 1.40, 1.40); 2 at 1.5; 1 at
1.57 (the random-walk value is about pi/2 = 1.57). Of all 41268 tracks, 31470 have span < 5, 9425 are
eligible and 6895 flagged (6094 of those above 1.57), leaving 2903 clean tracks with median span 6
(2467 have span 5-9).

## Leading vs trailing (removed 2026-11-20)

The per-frame leading/trailing analysis (`leading_trailing.py` plus two functions in `cotracking.py`) was
archived in `archive/leading_trailing_2026-11/`. It gave only a small, inconclusive trailing lean and was
likely affected by Kalman lag and possible channel-timing offsets. It will be redone as a kymograph
analysis; do not cite the old numbers.
