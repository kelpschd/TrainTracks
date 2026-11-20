# Leading vs trailing analysis (archived 2026-11-20)

Question: are particles consistently AHEAD of or TRAILING the organelle they cotrack with
(position of the particle projected on the organelle's heading during contact frames)?

Files: `leading_trailing.py` (the script) and `cotracking_offsets_section.py` (the two functions it
used, `episode_offsets` and `summarize_offsets`, removed from `cotracking/cotracking.py`).
To re-run exactly as it was, check out tag `pre-cleanup-2026-11-19`.

What it found (32 movies, pooled): only a small net trailing lean (about -0.85 px; 37 trailing vs 16 ahead
episodes, about half mixed). Not conclusive, and probably affected by two artifacts: the Kalman-linked
organelle tracks lag the true position, and the two channels may not be imaged at the same instant (a
0.2-frame lag made 22 of 24 synthetic zero-offset episodes read as "trailing"; per-channel ND2 timestamps
were never checked).

Why archived: to be replaced by a kymograph analysis along the organelle's path, which does not depend on
the Kalman positions or on a per-frame heading. The cotracking episodes themselves are unchanged.
