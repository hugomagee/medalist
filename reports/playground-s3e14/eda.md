# EDA — Prediction of Wild Blueberry Yield (Playground S3E14) (`playground-s3e14`)

Train: 12231 rows · Test: 3058 rows · Metric: mae (minimize)

## Columns: dtype, missingness, cardinality

| column | dtype | missing % (train) | missing % (test) | cardinality |
|---|---|---|---|---|
| id | int64 | 0.00 | 0.00 | 12231 |
| clonesize | float64 | 0.00 | 0.00 | 6 |
| honeybee | float64 | 0.00 | 0.00 | 7 |
| bumbles | float64 | 0.00 | 0.00 | 10 |
| andrena | float64 | 0.00 | 0.00 | 15 |
| osmia | float64 | 0.00 | 0.00 | 11 |
| MaxOfUpperTRange | float64 | 0.00 | 0.00 | 6 |
| MinOfUpperTRange | float64 | 0.00 | 0.00 | 5 |
| AverageOfUpperTRange | float64 | 0.00 | 0.00 | 5 |
| MaxOfLowerTRange | float64 | 0.00 | 0.00 | 6 |
| MinOfLowerTRange | float64 | 0.00 | 0.00 | 6 |
| AverageOfLowerTRange | float64 | 0.00 | 0.00 | 5 |
| RainingDays | float64 | 0.00 | 0.00 | 6 |
| AverageRainingDays | float64 | 0.00 | 0.00 | 6 |
| fruitset | float64 | 0.00 | 0.00 | 1399 |
| fruitmass | float64 | 0.00 | 0.00 | 1387 |
| seeds | float64 | 0.00 | 0.00 | 1861 |
| yield | float64 | 0.00 | n/a | 776 |


## Target: `yield`

- mean 6033.1800, std 1340.9389
- min 1945.5306, max 8969.4018
- distinct values: 776

## Correlation with target (numeric features)

| feature | pearson r |
|---|---|
| fruitset | 0.8858 |
| seeds | 0.8681 |
| fruitmass | 0.8267 |
| AverageRainingDays | -0.4749 |
| RainingDays | -0.4675 |
| clonesize | -0.3822 |
| osmia | 0.1973 |
| bumbles | 0.1699 |
| honeybee | -0.1254 |
| andrena | 0.0688 |
| MaxOfUpperTRange | -0.0196 |
| MinOfLowerTRange | -0.0194 |
| MinOfUpperTRange | -0.0194 |
| MaxOfLowerTRange | -0.0192 |
| AverageOfLowerTRange | -0.0191 |


## Train/test drift

Adversarial validation AUC (0.5 = indistinguishable, 1.0 = fully separable): **0.5041**

