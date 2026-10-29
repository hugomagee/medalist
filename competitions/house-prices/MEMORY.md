# MEMORY — `house-prices`

Metric: rmsle (minimize)

## Top 5 experiments by CV score

| exp | cv mean | cv std | family | hypothesis |
|---|---|---|---|---|
| e0050 | 0.103331 | 0.015038 | ensemble | Final convex blend around the bagged hybrid e0048 plus diverse GBM/kernel... |
| e0048 | 0.103334 | 0.014214 | hybrid | Seed/model-averaging the residual booster and inner residuals lowers hybrid... |
| e0047 | 0.103486 | 0.015093 | ensemble | Adding the hybrid e0046 (0.1035) to the convex blend pushes nested blend CV... |
| e0046 | 0.103526 | 0.014230 | hybrid | Boosting shallow trees on the FGLS Lasso's inner-CV residuals captures... |
| e0045 | 0.104917 | 0.015097 | ensemble | A nested convex blend of linear, kernel and GBM members beats the best... |


## Last 5 attempted

| exp | status | cv mean | hypothesis |
|---|---|---|---|
| e0050 | completed | 0.103331 | Final convex blend around the bagged hybrid e0048 plus diverse GBM/kernel... |
| e0049 | completed | 0.108447 | An FGLS SVR base + residual GBR is a strong (<0.106) hybrid whose errors... |
| e0048 | completed | 0.103334 | Seed/model-averaging the residual booster and inner residuals lowers hybrid... |
| e0047 | completed | 0.103486 | Adding the hybrid e0046 (0.1035) to the convex blend pushes nested blend CV... |
| e0046 | completed | 0.103526 | Boosting shallow trees on the FGLS Lasso's inner-CV residuals captures... |


## Graveyard — do not retry these families

(empty)


## Current best

- experiment: e0050 (cv 0.103331)
- submission source: /Users/hugomagee/medalist/experiments/house-prices/e0050/test_pred.parquet
- notes: —


## Budget

budget: 5 of 25 experiments remaining (2 reserved), 13965s of 14400s wall clock left
