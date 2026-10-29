"""Ames housing (Kaggle House Prices) domain features + fold-safe CV driver.

Three layers:

1. `base_frame` — deterministic, row-wise cleaning and feature construction
   driven by data_description.txt (NA = "feature absent", ordinal quality
   scales, totals, ages). Nothing is fit, so it cannot leak.
2. `LinearPrep` / `tree_matrix` — the parts that ARE fit (LotFrontage
   imputation, modes, skew correction, one-hot vocabulary, scaling). These
   are fit on the fold-train rows only.
3. `cv_log_target` — runs any `fit_predict(train_X, y_log, pred_X)` over the
   harness folds on log1p(SalePrice), drops the documented training outliers
   from every fit set, applies the feature-defined anomaly rule, refits on all
   (clean) rows for the test predictions, and returns dollar-scale arrays.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Fold = tuple[NDArray[np.int64], NDArray[np.int64]]
FitPredict = Callable[[pd.DataFrame, FloatArray, pd.DataFrame], FloatArray]

TARGET = "SalePrice"
ID = "Id"

# NA in these columns means "the house has none" (data_description.txt).
NONE_CATS = [
    "Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
    "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "PoolQC", "Fence", "MiscFeature", "MasVnrType",
]
ZERO_NUMS = [
    "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
    "BsmtFullBath", "BsmtHalfBath", "GarageCars", "GarageArea",
]
# Sparse NA in genuine categoricals: imputed with the fold-train mode.
MODE_CATS = [
    "MSZoning", "Exterior1st", "Exterior2nd", "Electrical", "KitchenQual",
    "Functional", "SaleType",
]

_QUAL = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ORDINAL_MAPS: dict[str, dict[str, int]] = {
    **{c: _QUAL for c in (
        "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
        "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC",
    )},
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7},
    "LotShape": {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "LandSlope": {"Sev": 0, "Mod": 1, "Gtl": 2},
    "PavedDrive": {"N": 0, "P": 1, "Y": 2},
    "CentralAir": {"N": 0, "Y": 1},
    "Street": {"Grvl": 0, "Pave": 1},
    "Fence": {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
}
# Near-constant / redundant columns that only add noise.
DROP = ["Utilities", "Street", "PoolQC"]


def training_outliers(train_df: pd.DataFrame) -> NDArray[np.bool_]:
    """De Cock's documented >4000 sq ft partial sales far below market value.

    Uses the target, so it is only ever applied to fit sets (never to rows
    being predicted)."""
    return ((train_df["GrLivArea"] > 4000) & (train_df[TARGET] < 300_000)).to_numpy()


def anomaly_segment(df: pd.DataFrame) -> NDArray[np.bool_]:
    """Feature-only definition of the same segment: very large new builds in
    Edwards sold as Partial (pre-completion) sales. Train has two (both
    ~$160-185k); test has one (Id 2550)."""
    return (
        (df["Neighborhood"] == "Edwards")
        & (df["SaleCondition"] == "Partial")
        & (df["GrLivArea"] > 4000)
    ).to_numpy()


def base_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic row-wise cleaning + domain features. No fitting."""
    X = df.drop(columns=[c for c in (ID, TARGET) if c in df.columns]).copy()

    for c in NONE_CATS:
        X[c] = X[c].fillna("None")
    for c in ZERO_NUMS:
        X[c] = X[c].fillna(0)
    X["GarageYrBlt"] = X["GarageYrBlt"].fillna(X["YearBuilt"])
    # One test row has a typo'd 2207 garage year.
    X.loc[X["GarageYrBlt"] > 2010, "GarageYrBlt"] = X["YearBuilt"]

    X["MSSubClass"] = X["MSSubClass"].astype(str)
    X["YrSoldCat"] = X["YrSold"].astype(str)
    X["MoSoldCat"] = X["MoSold"].astype(str)

    for col, mapping in ORDINAL_MAPS.items():
        X[col + "_o"] = X[col].map(mapping)

    # size
    X["TotalSF"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
    X["TotalFinSF"] = X["BsmtFinSF1"] + X["BsmtFinSF2"] + X["1stFlrSF"] + X["2ndFlrSF"]
    X["TotalBath"] = (
        X["FullBath"] + 0.5 * X["HalfBath"] + X["BsmtFullBath"] + 0.5 * X["BsmtHalfBath"]
    )
    X["TotalPorchSF"] = (
        X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"] + X["WoodDeckSF"]
    )
    # age / timing (sales span the 2006-2010 housing crash)
    X["Age"] = (X["YrSold"] - X["YearBuilt"]).clip(lower=0)
    X["RemodAge"] = (X["YrSold"] - X["YearRemodAdd"]).clip(lower=0)
    X["GarageAge"] = (X["YrSold"] - X["GarageYrBlt"]).clip(lower=0)
    X["IsRemodeled"] = (X["YearRemodAdd"] != X["YearBuilt"]).astype(int)
    X["IsNew"] = (X["YrSold"] == X["YearBuilt"]).astype(int)
    X["SaleTime"] = X["YrSold"] + (X["MoSold"] - 1) / 12.0
    # quality x quantity
    X["QualSF"] = X["OverallQual"] * X["GrLivArea"]
    X["QualTotalSF"] = X["OverallQual"] * X["TotalSF"]
    X["OverallGrade"] = X["OverallQual"] * X["OverallCond"]
    X["ExterGrade"] = X["ExterQual_o"] * X["ExterCond_o"]
    X["KitchenScore"] = X["KitchenAbvGr"] * X["KitchenQual_o"].fillna(3)
    X["GarageScore"] = X["GarageArea"] * X["GarageQual_o"]
    X["BsmtScore"] = X["TotalBsmtSF"] * X["BsmtQual_o"]
    # presence flags
    X["HasPool"] = (X["PoolArea"] > 0).astype(int)
    X["Has2ndFlr"] = (X["2ndFlrSF"] > 0).astype(int)
    X["HasGarage"] = (X["GarageArea"] > 0).astype(int)
    X["HasBsmt"] = (X["TotalBsmtSF"] > 0).astype(int)
    X["HasFireplace"] = (X["Fireplaces"] > 0).astype(int)
    X["HasPorch"] = (X["TotalPorchSF"] > 0).astype(int)

    return X.drop(columns=[c for c in DROP if c in X.columns])


def categorical_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) in ("string", "str")]


class LinearPrep:
    """Fold-fit preprocessing for linear / kernel models.

    LotFrontage <- fold-train neighbourhood median; sparse categorical NA <-
    fold-train mode; log1p on numeric columns whose fold-train skew > 0.75;
    one-hot with the fold-train vocabulary (rare levels < min_count dropped);
    robust scaling with fold-train median/IQR.
    """

    def __init__(self, skew_threshold: float = 0.75, min_count: int = 3) -> None:
        self.skew_threshold = skew_threshold
        self.min_count = min_count

    def fit(self, X: pd.DataFrame) -> LinearPrep:
        self.lot_by_nbhd_ = X.groupby("Neighborhood")["LotFrontage"].median()
        self.lot_global_ = float(X["LotFrontage"].median())
        self.modes_ = {c: X[c].mode().iloc[0] for c in MODE_CATS}
        Z = self._impute(X)
        self.cats_ = categorical_columns(Z)
        self.nums_ = [c for c in Z.columns if c not in self.cats_]
        num = Z[self.nums_].astype(float)
        num_med = num.median()
        num = num.fillna(num_med)
        self.num_fill_ = num_med
        skew = num.skew()
        self.skewed_ = [c for c in self.nums_ if abs(skew[c]) > self.skew_threshold and num[c].min() >= 0]
        D = self._design(Z, fit=True)
        self.center_ = D.median()
        iqr = D.quantile(0.75) - D.quantile(0.25)
        std = D.std(ddof=0)
        self.scale_ = iqr.where(iqr > 0, std).replace(0, 1.0)
        return self

    def _impute(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = X.copy()
        lot = Z["Neighborhood"].map(self.lot_by_nbhd_)
        Z["LotFrontage"] = Z["LotFrontage"].fillna(lot).fillna(self.lot_global_)
        for c, m in self.modes_.items():
            Z[c] = Z[c].fillna(m)
        return Z

    def _design(self, Z: pd.DataFrame, fit: bool) -> pd.DataFrame:
        num = Z[self.nums_].astype(float).fillna(self.num_fill_)
        num[self.skewed_] = np.log1p(num[self.skewed_])
        dummies = pd.get_dummies(Z[self.cats_].astype(str), dtype=float)
        if fit:
            counts = dummies.sum()
            self.dummy_cols_ = list(counts[counts >= self.min_count].index)
        dummies = dummies.reindex(columns=self.dummy_cols_, fill_value=0.0)
        return pd.concat([num.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)

    def transform(self, X: pd.DataFrame) -> NDArray[np.float64]:
        D = self._design(self._impute(X), fit=False)
        return ((D - self.center_) / self.scale_).to_numpy(dtype=np.float64)

    @property
    def feature_names(self) -> list[str]:
        return list(self.center_.index)


def tree_matrix(X_fit: pd.DataFrame, X_pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GBM view: categoricals as integer codes over the fold-train vocabulary
    (unseen levels -> -1), numerics untouched (GBMs handle NaN)."""
    A, B = X_fit.copy(), X_pred.copy()
    for c in categorical_columns(A):
        vocab = {v: i for i, v in enumerate(sorted(A[c].dropna().astype(str).unique()))}
        A[c] = A[c].astype(str).map(vocab).fillna(-1).astype(int)
        B[c] = B[c].astype(str).map(vocab).fillna(-1).astype(int)
    return A.astype(float), B.astype(float)


def cv_log_target(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    folds: list[Fold],
    fit_predict: FitPredict,
    *,
    drop_outliers: bool = True,
    anomaly_rule: bool = True,
) -> dict[str, Any]:
    """Fold loop on log1p(SalePrice); returns harness result dict pieces.

    `fit_predict(X_fit, y_fit_log, X_pred) -> log predictions` receives
    `base_frame` outputs, so any further fitting it does is fold-local.
    """
    X_all = base_frame(train_df)
    X_test = base_frame(test_df)
    y_log = np.log1p(train_df[TARGET].to_numpy(dtype=np.float64))
    outlier = training_outliers(train_df) if drop_outliers else np.zeros(len(train_df), bool)
    seg_train = anomaly_segment(train_df)
    seg_test = anomaly_segment(test_df)

    def _segment_value(fit_idx: NDArray[np.int64]) -> float | None:
        members = fit_idx[seg_train[fit_idx]]
        return float(np.mean(y_log[members])) if len(members) else None

    oof_log = np.empty(len(train_df), dtype=np.float64)
    for tr, va in folds:
        fit_idx = tr[~outlier[tr]]
        oof_log[va] = fit_predict(
            X_all.iloc[fit_idx].reset_index(drop=True), y_log[fit_idx],
            X_all.iloc[va].reset_index(drop=True),
        )
        seg_val = _segment_value(tr)
        if anomaly_rule and seg_val is not None:
            hit = va[seg_train[va]]
            oof_log[hit] = seg_val

    all_idx = np.arange(len(train_df), dtype=np.int64)
    fit_idx = all_idx[~outlier]
    test_log = fit_predict(
        X_all.iloc[fit_idx].reset_index(drop=True), y_log[fit_idx], X_test.reset_index(drop=True)
    )
    seg_val = _segment_value(all_idx)
    if anomaly_rule and seg_val is not None:
        test_log[seg_test] = seg_val

    clean = ~seg_train
    fold_scores = [float(np.sqrt(np.mean((oof_log[va] - y_log[va]) ** 2))) for _, va in folds]
    return {
        "oof_predictions": np.expm1(oof_log),
        "test_predictions": np.expm1(test_log),
        "fold_scores": fold_scores,
        "extra": {
            "rmse_log_all": float(np.sqrt(np.mean((oof_log - y_log) ** 2))),
            "rmse_log_excl_segment": float(np.sqrt(np.mean((oof_log[clean] - y_log[clean]) ** 2))),
            "segment_oof": {
                str(int(i)): float(np.expm1(oof_log[k]))
                for k, i in zip(np.where(seg_train)[0], train_df[ID].to_numpy()[seg_train])
            },
            "segment_test": {
                str(int(i)): float(np.expm1(test_log[k]))
                for k, i in zip(np.where(seg_test)[0], test_df[ID].to_numpy()[seg_test])
            },
        },
    }
