"""Generic feature engineering helpers, fold-safe by construction (SPEC §9).

Anything that looks at the target (target encoding) or at data statistics
that would leak (imputation means) is fit inside folds for the OOF view and
on the full train only for the test view.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
Fold = tuple[NDArray[np.int64], NDArray[np.int64]]

DEFAULT_SMOOTHING = 10.0


def _smoothed_means(
    cat: pd.Series, y: pd.Series, smoothing: float
) -> tuple[pd.Series, float]:
    global_mean = float(y.mean())
    stats = y.groupby(cat, observed=True).agg(["mean", "count"])
    encoded = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
        stats["count"] + smoothing
    )
    return encoded, global_mean


def target_encode_full_fit(
    train_cat: pd.Series,
    y: pd.Series,
    apply_cat: pd.Series,
    smoothing: float = DEFAULT_SMOOTHING,
) -> FloatArray:
    """Encoding fit on ALL of train — leaky for train rows; exists so tests can
    demonstrate the leak and so test-set encoding has a full-fit view."""
    mapping, global_mean = _smoothed_means(train_cat, y, smoothing)
    return apply_cat.map(mapping).fillna(global_mean).to_numpy(dtype=np.float64)


def target_encode_oof(
    train_cat: pd.Series,
    y: pd.Series,
    test_cat: pd.Series,
    folds: list[Fold],
    smoothing: float = DEFAULT_SMOOTHING,
) -> tuple[FloatArray, FloatArray]:
    """Fold-safe target encoding: each train row is encoded with a mapping fit
    only on its fold's training split; test rows use the full-train mapping."""
    train_cat = train_cat.reset_index(drop=True)
    y = y.reset_index(drop=True)
    oof = np.empty(len(train_cat), dtype=np.float64)
    for train_idx, valid_idx in folds:
        mapping, global_mean = _smoothed_means(
            train_cat.iloc[train_idx], y.iloc[train_idx], smoothing
        )
        oof[valid_idx] = (
            train_cat.iloc[valid_idx].map(mapping).fillna(global_mean).to_numpy(dtype=np.float64)
        )
    test_enc = target_encode_full_fit(train_cat, y, test_cat, smoothing)
    return oof, test_enc


def impute_numeric_oof(
    train_col: pd.Series, test_col: pd.Series, folds: list[Fold]
) -> tuple[FloatArray, FloatArray]:
    """Fold-safe mean imputation: valid rows are filled with the fold-train mean."""
    values = train_col.to_numpy(dtype=np.float64)
    oof = values.copy()
    for train_idx, valid_idx in folds:
        fold_mean = float(np.nanmean(values[train_idx]))
        valid_values = values[valid_idx]
        oof[valid_idx] = np.where(np.isnan(valid_values), fold_mean, valid_values)
    full_mean = float(np.nanmean(values))
    test_values = test_col.to_numpy(dtype=np.float64)
    test_out = np.where(np.isnan(test_values), full_mean, test_values)
    return oof, test_out


def add_group_aggregates(
    train: pd.DataFrame, test: pd.DataFrame, by: str, num_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group-statistic features (mean/std by category + category count).

    These use only feature columns — no target — so a full-train fit is safe.
    """
    train = train.copy()
    test = test.copy()
    counts = train[by].value_counts()
    for frame in (train, test):
        frame[f"{by}_count"] = frame[by].map(counts).fillna(0).astype(np.float64)
    for col in num_cols:
        stats = train.groupby(by, observed=True)[col].agg(["mean", "std"])
        for frame in (train, test):
            frame[f"{col}_mean_by_{by}"] = frame[by].map(stats["mean"])
            frame[f"{col}_std_by_{by}"] = frame[by].map(stats["std"])
    return train, test


def expand_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Expand a datetime column into year/month/day/dayofweek/hour features."""
    df = df.copy()
    ts = pd.to_datetime(df[col])
    df[f"{col}_year"] = ts.dt.year
    df[f"{col}_month"] = ts.dt.month
    df[f"{col}_day"] = ts.dt.day
    df[f"{col}_dayofweek"] = ts.dt.dayofweek
    df[f"{col}_hour"] = ts.dt.hour
    return df
