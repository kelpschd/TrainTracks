# ground_truth/

Validated tracks for all 32 movies: `<movie>_valid_tracks.csv`, columns `frame, x, y, particle`
(pixels; `particle` is unique within a movie). 294 tracks, 4040 detections in total.

**Provenance.** Copied on 2026-11-19 from `tracking_results_old/<movie>_valid_tracks.csv` in the
dataset folder. Those files are the output of the previous (pre-TrainTracks) pipeline, Kalman
linking with displacement and self-intersection filters, after manual validation of each track.
The originals carry extra columns (the pipeline parameters, intensity and colocalisation
measures); only the four above are kept here. `sg100_Well5_1018_valid_tracks.csv` replaces the
earlier repo-root copy, which was the same 11 tracks with coordinates rounded to 6 decimals.

**What this set is and is not.**
- Positives only: tracks that survived the old pipeline AND were judged real. It is not an
  exhaustive annotation of every particle in the movie. Use it for recall (is a known-good track
  recovered?) and for "would the cull remove a known-good track?" checks.
- Precision against it is not meaningful: any real particle that is not in the set counts as a
  false positive (the 2026-09-30 traccuracy summary shows precision of about 0.02 to 0.1 for that
  reason).
- Selection bias: the old pipeline's filters, including the dense-blob filter, decided which
  tracks could be validated, so tracks that filter removes are absent from the GT by construction.
- Many tracks have frame gaps (a track is not detected in every frame between its first and last
  frame), which the matching and the scorer handle as skip edges.

**Used by:** `scripts/batch_traccuracy.py` (default `--gt-root`), `scripts/track_flow_score.py --gt`
(sg100_Well5_1018 only) and `scripts/calibrate_cull_on_gt.py` (all movies).
