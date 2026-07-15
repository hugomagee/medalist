# DECISIONS.md — spec deviations and interpretations

- **2026-07-15 (M1)** SPEC §6.2 says only "status/score/notes transitions of a
  running experiment" may update a row, but §9 step 4 has the agent append
  error-analysis notes *after* an experiment completes. Interpretation: notes
  may also be set on completed/failed/timeout rows; all other columns remain
  immutable after insert.
- **2026-07-15 (M1)** SPEC §3 asks for uv; it was not installed, so it was
  installed via Homebrew (uv 0.11.28) rather than falling back to venv+pip,
  keeping the primary path. Python pinned to 3.12 as specified.
- **2026-07-15 (M2)** SPEC §6.1 has no `queued -> failed` transition need, but
  the runner queues then immediately fails experiments that flunk pre-flight
  checks (dir-name mismatch, static private check). Implemented as
  `queued -> running -> failed` so the ledger's transition rules stay strict.
- **2026-07-15 (M4)** SPEC §4 layout has no grading module; grading
  (`grade_submission`, medal thresholds, RESULT.md) lives in
  `core/submission.py` rather than adding an unlisted module.
- **2026-07-15 (M5)** The spec-mandated static check (reject run.py containing
  the string "private") false-positives on macOS absolute temp paths
  (`/private/var/...`). Kept the check as specified; the dumb agent's ensemble
  reads sibling artifacts via relative paths instead. Documented in README
  limitations.
- **2026-07-15 (M5)** SPEC §10 doesn't say how total wall clock is measured
  across CLI invocations; implemented as the sum of recorded experiment
  `wall_seconds`, so agent thinking time between runs is uncounted.
- **2026-07-15 (M1)** `competitions/playground-s3e14/private/` is empty —
  `solution.csv`/`leaderboard.csv` must be assembled by the human (SPEC §11)
  before that comp can be graded. M5 uses a synthetic competition, so this
  does not block M1–M5.
