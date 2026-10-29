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
- **2026-10-29 (house-prices)** Kaggle "Getting Started" comp with a rolling
  leaderboard and no published test labels, so the bundle uses the REAL
  train/test (no holdout carve-out): all 1,460 labelled rows go to training,
  and grading is a real Kaggle submission located in the public leaderboard
  (4,639 teams on 2026-10-29; ~74 teams score ~0 by submitting the leaked
  public Ames labels and are not comparable). `leaderboard`/`private_labels`
  are null, so `medalist grade` does not apply. Reconstructing test labels
  from the public De Cock dataset was deliberately NOT done: it is the leak
  itself, and having it on disk would contaminate decisions.
- **2026-10-29 (house-prices)** CV is 10-fold KFold (seed 42) rather than 5:
  with 1,460 rows each fold-model sees 90% of the data, closer to the
  full-data refit used for test predictions.
- **2026-10-29 (house-prices)** Experiment ids are global across the ledger;
  the first three runs used dirs e0001-e0003 and were refused before
  executing (e0031-e0033, "dir != assigned id"), burning 3 of 25 slots.
- **2026-10-29 (house-prices)** Outliers: the two >4000 sq ft Edwards
  Partial sales (Ids 524, 1299; De Cock flags them as not reflecting market
  value) are excluded from every FIT set inside `strategies/ames.py`, never
  from scoring — they still get OOF predictions and count in every CV score.
  Test Id 2550 matches the same feature pattern (Edwards, Partial,
  GrLivArea > 4000, OverallQual 10). Rather than extrapolate a ~$500k+
  price, rows in that feature-defined segment are predicted with the mean log
  price of the segment's members in the fit set (a comparable-sales rule).
  The rule is validated out-of-fold: 524 and 1299 fall in different folds,
  and each is predicted from the other (OOF $160k / $185k vs actual
  $185k / $160k). The segment definition is features-only, though it was
  chosen after seeing the two outliers — documented here for that reason.
- **2026-10-29 (s6e10)** Live Playground competition (deadline 2026-10-31
  23:59 UTC), so like house-prices there is NO holdout carve-out: the agent
  trains on the full Kaggle train (699,635 rows) and predicts the real test
  (299,844); grading is real Kaggle submissions. `leaderboard` and
  `private_labels` are null. `scripts/prepare_playground_s6e10.py` maps the
  literal True/False target to 1/0 in the harness-visible `data/train.csv`;
  untouched Kaggle files and the public source dataset live in `raw/`
  (gitignored). The source dataset (arseniyshutko/binary-aviation-satisfaction-129k,
  129,880 rows, public, allowed by the rules) is aligned to train's column
  order with a 1/0 target in `data/original.csv`. Original rows may be used
  to TRAIN or to build features, but never count in any CV score: the harness
  scores only Kaggle-train OOF rows. CV: stratified 5-fold, seed 42.
- **2026-10-29 (s6e10)** Budget raised via repo-root `budget.yaml`
  (max_experiments 70, per_experiment_seconds 3600, total_wall_seconds
  129600, min_reserve_experiments 3). The defaults (25 x 900 s, 4 h) are
  sized for small data; at 700k rows a single tuned GBM takes 10-30 min and
  the run spans ~52 h. Wall clock is still the sum of experiment runtimes, so
  concurrent experiments each count their full runtime.
- **2026-10-29 (s6e10)** Experiments may run two at a time
  (`scripts/s6e10_launch.sh` waits for the ledger to assign each id before
  starting the next, so dir names match ids). Thread count is passed via the
  `S6E10_THREADS` env var. Caveat: `_PrivateGuard` restores private/ when
  the first of two overlapping runs ends; harmless here because private/ is
  empty for a live competition.
- **2026-10-29 (s6e10)** GBM early stopping uses the outer validation fold
  (standard Kaggle practice). At ~140k validation rows per fold the optimism
  this adds to OOF AUC is far below the measured noise floor; best-iteration
  counts are logged per fold in `result.json` extra.
- **2026-10-29 (s6e10)** PyTorch 2.15 added to the venv as dependency group
  `nn` (`uv add --group nn torch`) for the embedding-MLP members (MPS).
  On macOS LightGBM and torch each load their own libomp; once a LightGBM
  model had been fit in the same process, `torch.randperm` deadlocked in an
  OpenMP barrier (e0064). Fix in `strategies/s6e10.py::mlp_fit_predict`:
  permutations come from numpy and torch is pinned to one CPU thread (the
  heavy work is on MPS). e0060 ran before the fix; its frozen run.py would
  now draw different minibatch orders if re-run.
- **2026-10-29 (s6e10)** Policy for runaway experiments: when an experiment
  is provably going to hit the 3600 s cap (e0057 CatBoost at ~30-60 min per
  fold) or is deadlocked (e0064), the agent kills it early and records why
  in the ledger notes. The slot is spent either way; killing frees the lane.
- **2026-10-29 (s6e10)** The brief said 22 features / 14 ratings; the data
  has 21 features / 13 ratings (no "Inflight service" column in either the
  Kaggle files or the public source). Every column is used.
- **2026-10-29 (s6e10)** Memory incident: two concurrent fb_v3 experiments
  (~5-6 GB each in float64) plus offline analysis jobs and desktop apps
  exhausted the 16 GB machine; swap reached 16.4/17.4 GB and load ~250 for
  roughly two hours (~18:50-20:45 UTC). e0074 was killed as too slow and
  e0078/e0079 were killed mid-run and rerun as e0081/e0083. Fixes:
  `fb_v3_32` stores static features as float32; the dispatcher only launches
  when `memory_pressure` reports >= 45% free; launches set
  OMP_WAIT_POLICY=PASSIVE; no offline jobs while two lanes run.
