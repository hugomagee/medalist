# RESULT — `house-prices` (Kaggle: House Prices - Advanced Regression Techniques)

**Submission:** `reports/house-prices/submission.csv`, built from **e0050** (final convex blend).
**10-fold CV RMSLE:** 0.10333 (fold mean ± 0.0150), pooled 0.10442.
**Kaggle public score: 0.11588** (submitted 2026-10-29, ref 57775555).
**Rank:** ~189 / 4,639 teams (top 4.1%); ~115 / 4,565 (top 2.5%) excluding
the 74 leaked-label scores (< 0.05). Ranks are located in the public
leaderboard snapshot downloaded 2026-10-29.

## Leaderboard context (pre-submission estimate)

Public leaderboard on 2026-10-29: 4,639 teams. About 74 score ~0 by
submitting the leaked public Ames labels; those are not comparable.

| public LB score | rank | percentile |
|---|---|---|
| 0.110 | ~98 | top 2.1% |
| 0.115 | ~164 | top 3.5% |
| 0.118 | ~270 | top 5.8% |
| 0.120 | ~455 | top 9.8% |

On this competition, honest public LB scores usually run ~0.005-0.015 above
a well-built 10-fold CV with the two outliers handled, so a public score
around 0.110-0.118 is the realistic range. The public LB is only 50% of the
test set (~730 rows), so ±0.005 of luck either way is normal.

Actual: CV 0.1033 -> public 0.1159 (gap 0.0126), inside the predicted range.

## Progression (17 scored experiments, 3 refused on a dir/id mismatch)

| exp | model | CV RMSLE |
|---|---|---|
| e0034 | constant median (floor) | 0.3988 |
| e0035 | LightGBM, raw features | 0.1237 |
| e0036 | LightGBM + Ames domain FE + outlier-free fits | 0.1202 |
| e0037 | Lasso, nested alpha, skew-corrected one-hot design | 0.1073 |
| e0038 | Ridge | 0.1117 |
| e0039 | RBF SVR (nested grid) | 0.1106 |
| e0040 | **FGLS Lasso** (weights = 1 / residual variance by SaleCondition) | 0.1061 |
| e0041 | Huber GBR | 0.1120 |
| e0042 | LightGBM, tuned for small data | 0.1132 |
| e0043 | XGBoost depth 3 | 0.1114 |
| e0044 | CatBoost, native categoricals | 0.1127 |
| e0045 | convex blend of e0037-e0044 | 0.1049 |
| e0046 | **hybrid**: FGLS Lasso + GBR on its inner-CV residuals | 0.1035 |
| e0047 | blend incl. e0046 | 0.1035 |
| e0048 | hybrid, bagged residual booster (3 GBR seeds + XGB) | 0.1033 |
| e0049 | hybrid with FGLS SVR base | 0.1084 |
| e0050 | **final convex blend** (e0048 0.79, GBR 0.18, SVR 0.03, XGB 0.01) | **0.1033** |

## What mattered

1. **Log target + domain cleaning** (NA = "none", ordinal quality scales,
   totals, ages): the basis for everything.
2. **Linear models beat GBMs on 1.4k rows.** Lasso 0.107 vs best GBM 0.111.
3. **Heteroscedasticity.** Abnormal/Family/Alloca sales have ~4x the residual
   variance of Normal sales. Feasible GLS weights improved Lasso 0.1073 ->
   0.1061.
4. **Residual boosting.** Shallow trees fit to the Lasso's *inner-CV*
   residuals (not in-sample ones) picked up interactions: 0.1061 -> 0.1035.
5. **Blending gave little beyond the hybrid.** The hybrid already contains
   the tree signal; final blend vs e0048 paired fold diff 0.00000 ± 0.00044.
   The blend was kept for diversity at equal CV.

## Honesty notes

- Every hyperparameter (Lasso/Ridge alpha, SVR grid) was picked by inner CV
  on each fold-train; blend weights are nested (fold k scored with weights
  fit on the other folds). GBM settings were fixed a priori, not searched.
- Outliers 524/1299 are excluded from fit sets only, never from scoring.
  Test Id 2550 sits in the same feature-defined segment and is predicted
  with the segment's comparable-sales mean ($171,930). See DECISIONS.md.
- No leaked test labels were downloaded or used.
- Remaining error is dominated by non-Normal sales (RMSE 0.16 on 262 rows
  vs 0.087 on Normal sales) and the model is well calibrated (slope 1.004),
  so further CV tuning would mostly fit noise.

## Reproduce

```
uv run medalist validate house-prices
uv run medalist run house-prices --exp <id>   # experiments/house-prices/<id>/run.py
uv run medalist submit house-prices --exp e0050
```

Shared logic lives in `strategies/ames.py` (features, fold-safe prep, CV
driver) and `strategies/ames_blend.py` (nested convex blend).
