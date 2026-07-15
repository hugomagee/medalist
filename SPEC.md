# SPEC.md — `medalist`: An Autonomous ML Competition Agent Harness

## 0. One-line summary

A harness that lets an AI agent (Claude Code) autonomously solve tabular data-science competitions end-to-end — EDA → feature engineering → model training → error analysis → iteration → submission — with every experiment logged, budgeted, and scored against a ground-truth verifier, evaluated first on **finished Kaggle competitions** so results can be graded against real historical leaderboards with zero human involvement.

## 1. Goals and non-goals

### Goals
1. Given a competition bundle (data files + description + metric), the system autonomously produces a valid submission file and reports its expected score.
2. Every experiment is reproducible: config in, artifacts + score out, all recorded in an append-only ledger.
3. The system improves iteratively: later experiments should beat earlier ones by consuming structured feedback (CV scores, error analysis, feature importances) from the ledger.
4. Evaluation is fully hands-off: on historical competitions, the harness scores submissions locally against the private test labels and reports the equivalent leaderboard percentile.
5. Budget-bounded: hard caps on number of experiments, wall-clock per experiment, and total run time. The system must degrade gracefully (always keep a best-so-far submission) rather than crash at budget exhaustion.

### Non-goals (v1)
- Deep learning / GPU competitions (image, NLP, audio). Tabular and time-series only.
- Live Kaggle submission automation (stub the interface; wire it in v2 only after checking the specific competition's rules on automated/AI tooling).
- Multi-competition parallelism.
- A web UI. CLI + markdown reports only.

## 2. Why this architecture

The agent (Claude Code) is the *scientist*. The harness is the *lab*: it enforces discipline the agent can't be trusted to maintain on its own — data leakage prevention, honest cross-validation, budget limits, and a persistent memory of what has been tried. The harness must be useful even with a dumb agent (random search over a strategy library should still produce a baseline submission), and it must let a smart agent shine (free-form experiment code, only constrained by the sandbox contract).

## 3. Tech stack

- Python 3.12, managed with `uv` (fall back to venv + pip if uv unavailable).
- Core: pandas, numpy, scikit-learn, lightgbm, xgboost, catboost, optuna.
- Ledger: SQLite via `sqlite3` stdlib (single file `ledger.db`) + JSON artifact files. No ORM.
- CLI: `typer`. Reports: markdown files rendered from Jinja2 templates.
- Tests: pytest + hypothesis (property-based tests for the scoring and CV modules).
- Lint/format: ruff. Type checking: mypy (strict on `core/`, lenient on `experiments/`).

## 4. Repository layout

```
medalist/
├── SPEC.md                     # this file
├── README.md                   # generated last, includes results table
├── pyproject.toml
├── core/
│   ├── competition.py          # Competition bundle loading & validation
│   ├── ledger.py               # Experiment ledger (SQLite + artifacts)
│   ├── runner.py               # Sandboxed experiment executor
│   ├── budget.py               # Budget manager
│   ├── cv.py                   # Cross-validation policy engine
│   ├── scoring.py              # Metric implementations + verifier
│   ├── memory.py               # Agent-facing digest of ledger state
│   └── submission.py           # Submission builder + format validator
├── strategies/                 # Reusable building blocks the agent may import
│   ├── baselines.py            # mean/mode/median, simple GBM defaults
│   ├── features.py             # generic FE: target encoding, aggregations, datetime expansion
│   └── ensembling.py           # blending, stacking, rank averaging
├── agent/
│   ├── AGENT.md                # the agent's operating manual (see §9)
│   └── prompts/                # reusable prompt fragments for subagent calls, if any
├── competitions/
│   └── <comp-slug>/
│       ├── bundle.yaml         # metadata: metric, target col, id col, files, cv policy
│       ├── data/               # train/test files (gitignored)
│       └── private/            # held-out labels for historical comps (gitignored, runner CANNOT read)
├── experiments/
│   └── <comp-slug>/<exp-id>/   # one dir per experiment (code + artifacts)
├── reports/
│   └── <comp-slug>/            # run report, leaderboard percentile, plots
└── tests/
```

## 5. Competition bundle format

`bundle.yaml`:

```yaml
slug: playground-s3e14
title: "Prediction of Wild Blueberry Yield"
metric: mae                     # must exist in core/scoring.py registry
metric_direction: minimize
target_column: yield
id_column: id
files:
  train: data/train.csv
  test: data/test.csv
  sample_submission: data/sample_submission.csv
cv_policy: auto                 # auto | kfold | stratified | timeseries | group
cv_params: {n_splits: 5, seed: 42}
time_column: null               # required if cv_policy == timeseries
group_column: null              # required if cv_policy == group
leaderboard: private/leaderboard.csv   # historical comps only: public snapshot of final LB scores
private_labels: private/solution.csv   # historical comps only
```

`core/competition.py` validates the bundle on load: files exist, target/id columns present, metric registered, sample submission parses. Fail fast with actionable errors.

## 6. The experiment ledger (the heart of the system)

### 6.1 Schema (SQLite, table `experiments`)

| column | type | notes |
|---|---|---|
| exp_id | TEXT PK | `e0001`, `e0002`, ... monotonically increasing |
| comp_slug | TEXT | |
| parent_id | TEXT NULL | experiment this one mutates/extends |
| hypothesis | TEXT | one sentence: what this experiment tests and why |
| approach | JSON | structured: {model_family, features_added, features_removed, hyperparams, ensemble_of} |
| status | TEXT | queued / running / completed / failed / timeout |
| cv_score_mean | REAL NULL | |
| cv_score_std | REAL NULL | |
| cv_fold_scores | JSON NULL | |
| wall_seconds | REAL | |
| created_at / finished_at | TEXT | ISO 8601 UTC |
| artifact_dir | TEXT | path to experiments/<slug>/<exp_id>/ |
| error | TEXT NULL | traceback tail on failure |
| notes | TEXT NULL | agent's post-hoc analysis (error analysis findings, importances summary) |

### 6.2 Ledger rules (enforce in code, test with pytest)
- Append-only: no row is ever updated except status/score/notes transitions of a running experiment. No deletes.
- An experiment without a recorded `hypothesis` (min 20 chars) is rejected at queue time. This forces the agent to think.
- `core/memory.py` produces `MEMORY.md` per competition after every experiment: top-5 experiments by score, last-5 attempted, a "graveyard" list of hypothesis families that failed ≥2 times (so the agent stops retrying them), current best submission path, and remaining budget. This file is the agent's working memory between sessions — it must stay under 300 lines.

## 7. Sandboxed experiment runner

`core/runner.py` executes one experiment:

1. Input: a single Python file `experiments/<slug>/<exp_id>/run.py` written by the agent, which must define `def run(train_df, test_df, cv_splitter, metric) -> ExperimentResult` where `ExperimentResult = {oof_predictions, test_predictions, fold_scores, feature_importances (optional), extra (optional dict)}`.
2. The runner — not the experiment — loads data, constructs the CV splitter from the bundle policy, computes fold scores from OOF predictions with the registered metric, and writes artifacts. **Experiments never see the private/ directory and never compute their own headline score.** This is the anti-cheating boundary; test it explicitly (an experiment attempting to read `private/` must hard-fail — enforce by running `run.py` in a subprocess with cwd set to the experiment dir and the private path made unreadable, and by static-checking `run.py` for the string `private`).
3. Timeout: SIGKILL the subprocess at `budget.per_experiment_seconds` (default 900s). Record status=timeout.
4. Determinism: runner sets global seeds (numpy, python, lightgbm/xgboost seeds via convention documented in AGENT.md). Same experiment re-run must reproduce CV mean within 1e-9 for deterministic models.
5. Artifacts written per experiment: `run.py` (frozen copy), `oof.parquet`, `test_pred.parquet`, `result.json`, `importances.json` if provided.

## 8. Cross-validation policy engine (`core/cv.py`)

- `auto` policy inspects the bundle: if `time_column` set → TimeSeriesSplit; elif `group_column` → GroupKFold; elif target is classification with <20 classes → StratifiedKFold; else KFold. Log the decision.
- OOF/test prediction alignment is validated (row counts, id ordering) before scoring — a misaligned submission is the classic silent killer.
- Property tests (hypothesis): fold indices partition the training set exactly; no train/valid overlap; group splits never split a group; time splits never let future rows into training folds.

## 9. AGENT.md — the agent's operating manual

Written for Claude Code, lives in `agent/AGENT.md`, and is the file a `CLAUDE.md` at repo root points to. It must specify the loop:

1. **Orient**: read `bundle.yaml`, `MEMORY.md`, and (first session only) run the built-in EDA script `python -m core.eda <slug>` which writes `reports/<slug>/eda.md` (dtypes, missingness, cardinality, target distribution, basic correlations, train/test drift check via adversarial validation AUC).
2. **Hypothesize**: state one falsifiable hypothesis. Check the graveyard first.
3. **Experiment**: write `run.py`, queue it via `medalist run <slug> --exp <id>`.
4. **Analyze**: after completion, read `result.json`; if score improved, run error analysis (worst-predicted rows, residual patterns, importances) and append findings to `notes`.
5. **Iterate or ensemble**: when single-model gains flatten (< 0.2% relative improvement over 3 experiments), switch to ensembling OOF predictions of the top-k diverse experiments.
6. **Finalize**: `medalist submit <slug>` builds the submission from the best experiment (or ensemble), validates format against sample_submission, and — for historical comps — scores it against private labels and computes the leaderboard percentile.

Mandated discipline items in AGENT.md: never fit anything on the full train set before CV is finalized for that pipeline; target encoding and imputation must be fit inside folds (use `strategies/features.py` helpers which are fold-safe by construction); first three experiments are always (a) naive baseline, (b) LightGBM on raw features, (c) LightGBM + generic FE — these anchor the ledger.

## 10. Budget manager (`core/budget.py`)

Config per run: `max_experiments` (default 25), `per_experiment_seconds` (900), `total_wall_seconds` (default 4 hours), `min_reserve_experiments` (2 — reserved for final ensemble + submission build). The CLI refuses to queue experiments past the caps and prints remaining budget in every command's output so the agent always sees it.

## 11. Evaluation protocol (the headline feature)

For historical competitions:

1. `medalist grade <slug>` scores the final submission against `private/solution.csv` with the bundle metric.
2. It then locates the score in `private/leaderboard.csv` (final private LB export) and reports: raw score, rank-equivalent, percentile, and medal band (Kaggle rules: bronze/silver/gold thresholds depend on team count — implement the standard thresholds).
3. Output: `reports/<slug>/RESULT.md` with the score, percentile, budget consumed, experiment count, a score-vs-experiment-number progression chart (matplotlib, saved PNG), and the top-3 experiments' hypotheses.

Target first benchmarks (small, tabular, laptop-friendly — acquire via Kaggle API, checking each comp's data remains downloadable):
- 2–3 Kaggle Playground Series episodes (season 3/4, regression + classification mix)
- One classic: e.g. *Titanic*-class comps are too degenerate; prefer something like *Santander Customer Transaction* scale-down or a Playground time-series episode for the timeseries CV path.

Note: `private/leaderboard.csv` and `private/solution.csv` must be assembled once per comp by the human (downloading via Kaggle API where available); the harness treats them as read-only inputs. Document the assembly steps in `competitions/README.md`.

## 12. CLI surface

```
medalist init <slug>            # scaffold competition dir + bundle template
medalist validate <slug>        # bundle + data sanity checks
medalist eda <slug>             # writes reports/<slug>/eda.md
medalist run <slug> --exp e0007 # execute one experiment
medalist status <slug>          # ledger summary + budget remaining
medalist memory <slug>          # regenerate MEMORY.md
medalist submit <slug>          # build + validate submission from best/ensemble
medalist grade <slug>           # historical comps: score vs private LB
```

## 13. Testing & definition of done

Milestones (each must end with all tests green before the next starts):

- **M1 — Core skeleton**: bundle loading, ledger CRUD, CLI stubs. Unit tests for ledger append-only rules and bundle validation.
- **M2 — Runner + CV + scoring**: sandboxed execution, timeout kill, determinism test, CV property tests, metric registry (mae, rmse, rmsle, auc, logloss, accuracy, f1) each tested against sklearn reference implementations to 1e-12.
- **M3 — Strategies + EDA + memory**: fold-safe feature helpers (leakage test: target encoding fit on full data vs fold-safe must differ; fold-safe OOF must not degrade under label shuffling beyond chance), EDA report, MEMORY.md generation.
- **M4 — Submission + grading**: format validator, percentile/medal computation with unit tests against hand-computed fixtures.
- **M5 — Dry run**: a scripted "dumb agent" (fixed sequence: baseline → LGBM → LGBM+FE → ensemble) runs end-to-end on a synthetic competition generated by `tests/make_synthetic_comp.py`, produces a graded RESULT.md. This proves the harness with zero LLM involvement.
- **M6 — Live agent run**: Claude Code, following AGENT.md, runs a real historical Playground comp within budget and produces RESULT.md with a leaderboard percentile.

**Definition of done for v1**: M1–M5 complete with pytest fully green and mypy clean on `core/`; M6 executed at least once with percentile reported; README.md written with the results table and a honest limitations section.

## 14. Guardrails & conventions for the building agent (Claude Code)

- TDD: write tests first for every `core/` module.
- Commit per milestone minimum; conventional commit messages.
- No silent exception swallowing anywhere in `core/`.
- If a design decision in this spec proves wrong during implementation, do not silently deviate: record the deviation and rationale in `DECISIONS.md` and proceed.
- Keep `core/` under ~2,500 LOC total. Complexity belongs in experiments, not the harness.
