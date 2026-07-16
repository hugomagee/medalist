# MEMORY — `playground-series-s6e3`

Metric: auc (maximize)

## Top 5 experiments by CV score

| exp | cv mean | cv std | family | hypothesis |
|---|---|---|---|---|
| e0023 | 0.916430 | 0.000778 | ensemble | LightGBM, XGBoost and CatBoost make partially decorrelated ranking errors,... |
| e0027 | 0.916430 | 0.000778 | ensemble | Exhaustive equal-weight subset search plus an integer-weight hill climb... |
| e0030 | 0.916430 | 0.000778 | ensemble | Adding the DART member to the exhaustive subset + integer-weight search... |
| e0026 | 0.916424 | 0.000781 | ensemble | Adding the seed-bagged LGBM and the structurally different logistic member... |
| e0024 | 0.916117 | 0.000798 | gbm-bagged | Averaging the best LGBM config (e0019) over five seeds removes bagging... |


## Last 5 attempted

| exp | status | cv mean | hypothesis |
|---|---|---|---|
| e0030 | completed | 0.916430 | Adding the DART member to the exhaustive subset + integer-weight search... |
| e0029 | completed | 0.915232 | DART's dropout de-correlates tree contributions, producing a LightGBM whose... |
| e0028 | completed | 0.913948 | A fold-safe linear stack on member logits assigns continuous weights (incl.... |
| e0027 | completed | 0.916430 | Exhaustive equal-weight subset search plus an integer-weight hill climb... |
| e0026 | completed | 0.916424 | Adding the seed-bagged LGBM and the structurally different logistic member... |


## Graveyard — do not retry these families

(empty)


## Current best

- experiment: e0023 (cv 0.916430)
- submission source: /Users/hugomagee/medalist/experiments/playground-series-s6e3/e0023/test_pred.parquet
- notes: Greedy rank-blend .916430 (+.0004 over best single). Picked XGB+LGBM-FE+CatBoost+LGBM-highcap equally; skipped the best single e0019 — diversity beats individual strength. Next: add decorrelated members (seed-bag, linear).


## Budget

budget: 8 of 25 experiments remaining (2 reserved), 12314s of 14400s wall clock left
