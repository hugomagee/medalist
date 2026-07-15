# DECISIONS.md — spec deviations and interpretations

- **2026-07-15 (M1)** SPEC §6.2 says only "status/score/notes transitions of a
  running experiment" may update a row, but §9 step 4 has the agent append
  error-analysis notes *after* an experiment completes. Interpretation: notes
  may also be set on completed/failed/timeout rows; all other columns remain
  immutable after insert.
- **2026-07-15 (M1)** SPEC §3 asks for uv; it was not installed, so it was
  installed via Homebrew (uv 0.11.28) rather than falling back to venv+pip,
  keeping the primary path. Python pinned to 3.12 as specified.
- **2026-07-15 (M1)** `competitions/playground-s3e14/private/` is empty —
  `solution.csv`/`leaderboard.csv` must be assembled by the human (SPEC §11)
  before that comp can be graded. M5 uses a synthetic competition, so this
  does not block M1–M5.
