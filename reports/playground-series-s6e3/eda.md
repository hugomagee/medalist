# EDA — Telco Customer Churn (Playground S6E3) (`playground-series-s6e3`)

Train: 475355 rows · Test: 118839 rows · Metric: auc (maximize)

## Columns: dtype, missingness, cardinality

| column | dtype | missing % (train) | missing % (test) | cardinality |
|---|---|---|---|---|
| id | int64 | 0.00 | 0.00 | 475355 |
| gender | str | 0.00 | 0.00 | 2 |
| SeniorCitizen | int64 | 0.00 | 0.00 | 2 |
| Partner | str | 0.00 | 0.00 | 2 |
| Dependents | str | 0.00 | 0.00 | 2 |
| tenure | int64 | 0.00 | 0.00 | 72 |
| PhoneService | str | 0.00 | 0.00 | 2 |
| MultipleLines | str | 0.00 | 0.00 | 3 |
| InternetService | str | 0.00 | 0.00 | 3 |
| OnlineSecurity | str | 0.00 | 0.00 | 3 |
| OnlineBackup | str | 0.00 | 0.00 | 3 |
| DeviceProtection | str | 0.00 | 0.00 | 3 |
| TechSupport | str | 0.00 | 0.00 | 3 |
| StreamingTV | str | 0.00 | 0.00 | 3 |
| StreamingMovies | str | 0.00 | 0.00 | 3 |
| Contract | str | 0.00 | 0.00 | 3 |
| PaperlessBilling | str | 0.00 | 0.00 | 2 |
| PaymentMethod | str | 0.00 | 0.00 | 4 |
| MonthlyCharges | float64 | 0.00 | 0.00 | 1911 |
| TotalCharges | float64 | 0.00 | 0.00 | 29941 |
| Churn | int64 | 0.00 | n/a | 2 |


## Target: `Churn`

- mean 0.2252, std 0.4177
- min 0.0000, max 1.0000
- distinct values: 2 (looks like classification)

## Correlation with target (numeric features)

| feature | pearson r |
|---|---|
| tenure | -0.4182 |
| MonthlyCharges | 0.2722 |
| SeniorCitizen | 0.2364 |
| TotalCharges | -0.2182 |


## Train/test drift

Adversarial validation AUC (0.5 = indistinguishable, 1.0 = fully separable): **0.5009**

