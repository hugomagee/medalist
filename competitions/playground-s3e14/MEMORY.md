# MEMORY — `playground-s3e14`

Metric: mae (minimize)

## Top 5 experiments by CV score

| exp | cv mean | cv std | family | hypothesis |
|---|---|---|---|---|
| e0013 | 342.493905 | 4.277520 | ensemble | The OOF-selected equal blend of XGB, CatBoost, tuned LGBM and its seed-bag,... |
| e0010 | 343.360407 | 4.341046 | ensemble | An equal-weight blend of the three L1 GBMs preserves the MAE geometry better... |
| e0012 | 343.631958 | 4.047983 | gbm-l1-tuned | Averaging the tuned L1 LightGBM over five seeds reduces bagging variance and... |
| e0011 | 343.784368 | 3.971236 | gbm-l1-tuned | A seeded 40-trial Optuna search over L1 LightGBM regularization and capacity... |
| e0005 | 345.251783 | 4.548689 | gbm-l1 | Training LightGBM with the L1 objective aligns the loss with the MAE metric... |


## Last 5 attempted

| exp | status | cv mean | hypothesis |
|---|---|---|---|
| e0013 | completed | 342.493905 | The OOF-selected equal blend of XGB, CatBoost, tuned LGBM and its seed-bag,... |
| e0012 | completed | 343.631958 | Averaging the tuned L1 LightGBM over five seeds reduces bagging variance and... |
| e0011 | completed | 343.784368 | A seeded 40-trial Optuna search over L1 LightGBM regularization and capacity... |
| e0010 | completed | 343.360407 | An equal-weight blend of the three L1 GBMs preserves the MAE geometry better... |
| e0009 | completed | 346.014943 | A fold-safe linear stack of the three L1 GBMs plus snapping predictions to... |


## Graveyard — do not retry these families

(empty)


## Current best

- experiment: e0013 (cv 342.493905)
- submission source: /Users/hugomagee/medalist/experiments/playground-s3e14/e0013/test_pred.parquet
- notes: —


## Budget

budget: 12 of 25 experiments remaining (2 reserved), 14075s of 14400s wall clock left
