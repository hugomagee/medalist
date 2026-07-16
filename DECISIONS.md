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
- **2026-07-15 (M6)** Kaggle does not publish Playground private test labels,
  so `playground-s3e14` cannot be graded against the true private test set.
  Adaptation (`scripts/prepare_playground_s3e14.py`, seed 42, run BEFORE the
  agent loop saw any data): a 20% holdout (3,058 rows) was carved from the
  original train; the agent trains/CVs only on the remaining 80% (12,231
  rows). The holdout's features are the operative `data/test.csv`, its labels
  are `private/solution.csv`, and the originals were moved to
  `private/original_*.csv` (unreadable to experiments; untouched by the
  agent). The final submission is graded on holdout MAE and located in the
  real final private LB (`private/leaderboard.csv`, 1,877 teams).
  **The reported rank/percentile/medal is therefore an ESTIMATE**: the holdout
  is drawn from the train distribution, not the true private test set, and LB
  teams optimized against Kaggle's own test split. Holdout MAE is also noisier
  (3,058 rows vs Kaggle's ~6.9k private test rows).
- **2026-07-15 (M6)** The Kaggle leaderboard export had a UTF-8 BOM and a
  capital-S `Score` column; grading expects a lowercase `score` column
  (competitions/README.md), so the prep script normalized the file in place
  (BOM stripped, `Score` → `score`, other columns kept).
- **2026-07-16 (s6e3)** First maximize-direction competition (ROC AUC). Audited
  every direction-sensitive code path before relying on it: `grade_submission`
  rank comparison, percentile, medal bands, `ledger.best`, MEMORY.md top-5,
  RESULT.md top-3, and CLI best-experiment selection. All were already
  direction-correct, but only `ledger.best` had a maximize test; added maximize
  coverage (`TestGradingMaximize` in tests/test_submission.py incl. an
  asymmetric-leaderboard rank test and Kaggle-style tie handling,
  `test_top5_is_best_first_for_maximize` in tests/test_memory.py). No code
  changes needed; 122 tests green.
- **2026-07-16 (s6e3)** Same holdout adaptation as s3e14 (Kaggle publishes no
  Playground private labels): `scripts/prepare_playground_s6e3.py`, seed 42,
  run BEFORE the agent loop saw any data. 20% holdout (118,839 rows) carved
  from the original train (594,194 rows); agent trains/CVs on the remaining
  475,355. Holdout features are the operative `data/test.csv`, labels are
  `private/solution.csv`, originals moved to `private/original_*.csv`. Reported
  rank/percentile/medal against the real 4,143-team final LB is therefore an
  ESTIMATE (same caveats as s3e14). Two adaptations for classification: the
  holdout split is STRATIFIED on the target (matches the bundle's stratified
  CV policy; churn rate 22.52% preserved on both sides), and the target
  `Churn` was mapped Yes/No → 1/0 in every harness-visible file (the metric
  registry and AUC need numeric labels; the mapping is loss-free and the
  original untouched files retain the string labels).
