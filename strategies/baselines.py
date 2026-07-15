"""Baseline building blocks: fold-safe constant predictors and LightGBM defaults."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Fold = tuple[NDArray[np.int64], NDArray[np.int64]]

SEED = 42


def _constant(values: FloatArray, strategy: str) -> float:
    if strategy == "mean":
        return float(np.mean(values))
    if strategy == "median":
        return float(np.median(values))
    if strategy == "mode":
        uniques, counts = np.unique(values, return_counts=True)
        return float(uniques[np.argmax(counts)])
    raise ValueError(f"unknown strategy '{strategy}'")


def constant_baseline(
    y: pd.Series, folds: list[Fold], n_test: int, strategy: str = "mean"
) -> tuple[FloatArray, FloatArray]:
    """OOF/test predictions of a per-fold constant. The honest floor score."""
    values = y.to_numpy(dtype=np.float64)
    oof = np.empty(len(values), dtype=np.float64)
    for train_idx, valid_idx in folds:
        oof[valid_idx] = _constant(values[train_idx], strategy)
    test_pred = np.full(n_test, _constant(values, strategy), dtype=np.float64)
    return oof, test_pred


def lgbm_default_params(task: str) -> dict[str, Any]:
    params: dict[str, Any] = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": SEED,
        "deterministic": True,
        "force_row_wise": True,
        "n_jobs": 1,
        "verbose": -1,
    }
    if task == "classification":
        params["objective"] = "binary"
    else:
        params["objective"] = "regression"
    return params


def fit_predict_lgbm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    target: str,
    folds: list[Fold],
    task: str = "regression",
    params: dict[str, Any] | None = None,
) -> tuple[FloatArray, FloatArray, dict[str, float]]:
    """Per-fold LightGBM fit; returns (oof, mean test prediction, gain importances)."""
    merged = lgbm_default_params(task)
    merged.update(params or {})
    X = train_df[features]
    y = train_df[target].to_numpy()
    X_test = test_df[features]

    oof = np.empty(len(X), dtype=np.float64)
    test_preds = np.zeros((len(folds), len(X_test)), dtype=np.float64)
    importance_sum = np.zeros(len(features), dtype=np.float64)

    for i, (train_idx, valid_idx) in enumerate(folds):
        if task == "classification":
            clf = lgb.LGBMClassifier(**merged)
            clf.fit(X.iloc[train_idx], y[train_idx])
            oof[valid_idx] = clf.predict_proba(X.iloc[valid_idx])[:, 1]
            test_preds[i] = clf.predict_proba(X_test)[:, 1]
            importance_sum += clf.booster_.feature_importance(importance_type="gain")
        else:
            reg = lgb.LGBMRegressor(**merged)
            reg.fit(X.iloc[train_idx], y[train_idx])
            oof[valid_idx] = reg.predict(X.iloc[valid_idx])
            test_preds[i] = reg.predict(X_test)
            importance_sum += reg.booster_.feature_importance(importance_type="gain")

    importances = {f: float(v) for f, v in zip(features, importance_sum / len(folds), strict=True)}
    return oof, test_preds.mean(axis=0), importances
