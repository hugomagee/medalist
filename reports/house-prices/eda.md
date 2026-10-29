# EDA — House Prices - Advanced Regression Techniques (Ames, Iowa) (`house-prices`)

Train: 1460 rows · Test: 1459 rows · Metric: rmsle (minimize)

## Columns: dtype, missingness, cardinality

| column | dtype | missing % (train) | missing % (test) | cardinality |
|---|---|---|---|---|
| Id | int64 | 0.00 | 0.00 | 1460 |
| MSSubClass | int64 | 0.00 | 0.00 | 15 |
| MSZoning | str | 0.00 | 0.27 | 5 |
| LotFrontage | float64 | 17.74 | 15.56 | 110 |
| LotArea | int64 | 0.00 | 0.00 | 1073 |
| Street | str | 0.00 | 0.00 | 2 |
| Alley | str | 93.77 | 92.67 | 2 |
| LotShape | str | 0.00 | 0.00 | 4 |
| LandContour | str | 0.00 | 0.00 | 4 |
| Utilities | str | 0.00 | 0.14 | 2 |
| LotConfig | str | 0.00 | 0.00 | 5 |
| LandSlope | str | 0.00 | 0.00 | 3 |
| Neighborhood | str | 0.00 | 0.00 | 25 |
| Condition1 | str | 0.00 | 0.00 | 9 |
| Condition2 | str | 0.00 | 0.00 | 8 |
| BldgType | str | 0.00 | 0.00 | 5 |
| HouseStyle | str | 0.00 | 0.00 | 8 |
| OverallQual | int64 | 0.00 | 0.00 | 10 |
| OverallCond | int64 | 0.00 | 0.00 | 9 |
| YearBuilt | int64 | 0.00 | 0.00 | 112 |
| YearRemodAdd | int64 | 0.00 | 0.00 | 61 |
| RoofStyle | str | 0.00 | 0.00 | 6 |
| RoofMatl | str | 0.00 | 0.00 | 8 |
| Exterior1st | str | 0.00 | 0.07 | 15 |
| Exterior2nd | str | 0.00 | 0.07 | 16 |
| MasVnrType | str | 59.73 | 61.27 | 3 |
| MasVnrArea | float64 | 0.55 | 1.03 | 327 |
| ExterQual | str | 0.00 | 0.00 | 4 |
| ExterCond | str | 0.00 | 0.00 | 5 |
| Foundation | str | 0.00 | 0.00 | 6 |
| BsmtQual | str | 2.53 | 3.02 | 4 |
| BsmtCond | str | 2.53 | 3.08 | 4 |
| BsmtExposure | str | 2.60 | 3.02 | 4 |
| BsmtFinType1 | str | 2.53 | 2.88 | 6 |
| BsmtFinSF1 | int64 | 0.00 | 0.07 | 637 |
| BsmtFinType2 | str | 2.60 | 2.88 | 6 |
| BsmtFinSF2 | int64 | 0.00 | 0.07 | 144 |
| BsmtUnfSF | int64 | 0.00 | 0.07 | 780 |
| TotalBsmtSF | int64 | 0.00 | 0.07 | 721 |
| Heating | str | 0.00 | 0.00 | 6 |
| HeatingQC | str | 0.00 | 0.00 | 5 |
| CentralAir | str | 0.00 | 0.00 | 2 |
| Electrical | str | 0.07 | 0.00 | 5 |
| 1stFlrSF | int64 | 0.00 | 0.00 | 753 |
| 2ndFlrSF | int64 | 0.00 | 0.00 | 417 |
| LowQualFinSF | int64 | 0.00 | 0.00 | 24 |
| GrLivArea | int64 | 0.00 | 0.00 | 861 |
| BsmtFullBath | int64 | 0.00 | 0.14 | 4 |
| BsmtHalfBath | int64 | 0.00 | 0.14 | 3 |
| FullBath | int64 | 0.00 | 0.00 | 4 |
| HalfBath | int64 | 0.00 | 0.00 | 3 |
| BedroomAbvGr | int64 | 0.00 | 0.00 | 8 |
| KitchenAbvGr | int64 | 0.00 | 0.00 | 4 |
| KitchenQual | str | 0.00 | 0.07 | 4 |
| TotRmsAbvGrd | int64 | 0.00 | 0.00 | 12 |
| Functional | str | 0.00 | 0.14 | 7 |
| Fireplaces | int64 | 0.00 | 0.00 | 4 |
| FireplaceQu | str | 47.26 | 50.03 | 5 |
| GarageType | str | 5.55 | 5.21 | 6 |
| GarageYrBlt | float64 | 5.55 | 5.35 | 97 |
| GarageFinish | str | 5.55 | 5.35 | 3 |
| GarageCars | int64 | 0.00 | 0.07 | 5 |
| GarageArea | int64 | 0.00 | 0.07 | 441 |
| GarageQual | str | 5.55 | 5.35 | 5 |
| GarageCond | str | 5.55 | 5.35 | 5 |
| PavedDrive | str | 0.00 | 0.00 | 3 |
| WoodDeckSF | int64 | 0.00 | 0.00 | 274 |
| OpenPorchSF | int64 | 0.00 | 0.00 | 202 |
| EnclosedPorch | int64 | 0.00 | 0.00 | 120 |
| 3SsnPorch | int64 | 0.00 | 0.00 | 20 |
| ScreenPorch | int64 | 0.00 | 0.00 | 76 |
| PoolArea | int64 | 0.00 | 0.00 | 8 |
| PoolQC | str | 99.52 | 99.79 | 3 |
| Fence | str | 80.75 | 80.12 | 4 |
| MiscFeature | str | 96.30 | 96.50 | 4 |
| MiscVal | int64 | 0.00 | 0.00 | 21 |
| MoSold | int64 | 0.00 | 0.00 | 12 |
| YrSold | int64 | 0.00 | 0.00 | 5 |
| SaleType | str | 0.00 | 0.07 | 9 |
| SaleCondition | str | 0.00 | 0.00 | 6 |
| SalePrice | int64 | 0.00 | n/a | 663 |


## Target: `SalePrice`

- mean 180921.1959, std 79442.5029
- min 34900.0000, max 755000.0000
- distinct values: 663

## Correlation with target (numeric features)

| feature | pearson r |
|---|---|
| OverallQual | 0.7910 |
| GrLivArea | 0.7086 |
| GarageCars | 0.6404 |
| GarageArea | 0.6234 |
| TotalBsmtSF | 0.6136 |
| 1stFlrSF | 0.6059 |
| FullBath | 0.5607 |
| TotRmsAbvGrd | 0.5337 |
| YearBuilt | 0.5229 |
| YearRemodAdd | 0.5071 |
| MasVnrArea | 0.4752 |
| GarageYrBlt | 0.4702 |
| Fireplaces | 0.4669 |
| BsmtFinSF1 | 0.3864 |
| LotFrontage | 0.3349 |


## Train/test drift

Adversarial validation AUC (0.5 = indistinguishable, 1.0 = fully separable): **0.1905**

