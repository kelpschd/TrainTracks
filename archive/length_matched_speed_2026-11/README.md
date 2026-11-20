# Length-matched speed script (archived 2026-11-20)

`length_matched_speed.py` compared particle track speed, cotracking vs never-cotracking, within
track-length bins (raw; length-matched; movie x length-matched; regression with movie fixed effects;
outside-episode speed only) with cluster bootstrap CIs over movies, reading
`<cotracking_dir>/summary/track_speed.csv` and writing `speed_length_matched.csv`.

Why archived: the same analysis (raw, `length_matched`, `movie_and_length_matched`) is done directly in the
analysis `.qmd` (`analysis/analysis_20261026.qmd`, the "cotracking_length_matched" figure), which is where
the user keeps developing it. Nothing else read `speed_length_matched.csv`.
To re-run it as it was: check out tag `pre-cleanup-2026-11-19`.
