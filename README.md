# medalist

[![CI](https://github.com/hugomagee/medalist/actions/workflows/ci.yml/badge.svg)](https://github.com/hugomagee/medalist/actions/workflows/ci.yml)

A harness that lets an AI agent autonomously solve tabular data-science
competitions end-to-end — EDA → feature engineering → training → error
analysis → iteration → submission — with every experiment logged, budgeted,
and scored against a ground-truth verifier. Evaluated on finished Kaggle
competitions so results grade against real historical leaderboards with zero
human involvement. See [SPEC.md](SPEC.md) for the full design.

## Status: M1–M6 complete

```
uv sync
uv run pytest          # 122 tests
uv run medalist --help
```

The agent-facing loop is documented in [agent/AGENT.md](agent/AGENT.md);
assembling historical competition bundles is documented in
[competitions/README.md](competitions/README.md).

## Results

| competition | type | metric | experiments | best CV | private score | rank | percentile | medal |
|---|---|---|---|---|---|---|---|---|
| synthetic-mae (M5 dry run, scripted agent) | regression | mae | 4 | 1.118 | 1.144 | 5/100 | top 5% | gold* |
| playground-s3e14 (M6, live agent) | regression | mae | 13 | 342.49 | 335.77 | 601/1877 | top 32.0% | none† |
| playground-series-s6e3 (live agent) | binary classification | auc | 16 | 0.91643 | 0.91709 | 655/4143 | top 15.8% | none† |

\* Synthetic leaderboard — proves the plumbing, not competitive skill. The
scripted "dumb agent" ran baseline → LightGBM → LightGBM+FE → blend with no
LLM involvement: baseline MAE 5.60 → best 1.12, CV within 2.3% of the
private score.

† Playground ranks are ESTIMATES: Kaggle does not publish Playground private
test labels, so submissions are scored on a seeded 20% holdout carved from
the original train before the agent loop starts, then located in the real
final private leaderboard. See `reports/<slug>/RESULT.md` for the full
caveats.

## Limitations (honest)

- **Tabular/time-series only.** No deep learning, images, NLP, or GPU comps.
- **Grading needs human-assembled ground truth.** Private labels for
  Playground-style comps must be reconstructed from source datasets; where
  that's impossible, percentile grading is unavailable.
- **Synthetic leaderboard percentiles are not comparable** to real ones; the
  M5 medal is a plumbing check.
- **The sandbox is a discipline boundary, not a security boundary.** The
  subprocess shares the interpreter and filesystem; chmod + static checks
  stop honest-but-curious experiment code, not a motivated adversary.
- **Static `private` check can false-positive** (e.g. macOS `/private/var`
  temp paths); use relative paths inside `run.py`.
- **Wall-clock budget** is tracked as the sum of experiment runtimes, not
  session elapsed time; agent thinking time between experiments is uncounted.
- **Ensemble/stacking assumes shared folds.** All members must come from the
  same bundle CV policy (the harness fixes folds per competition, so this
  holds unless bundle CV params change mid-run).
- Timeseries CV maps unsorted input correctly, but duplicate timestamps at a
  fold boundary can place equal-time rows in both train and valid.
