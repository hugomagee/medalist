"""Ensembling: blending, rank averaging, OOF-safe linear stacking."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression

FloatArray = NDArray[np.float64]
Fold = tuple[NDArray[np.int64], NDArray[np.int64]]


def _stacked(preds: list[FloatArray]) -> FloatArray:
    arrays = [np.asarray(p, dtype=np.float64).ravel() for p in preds]
    lengths = {len(a) for a in arrays}
    if len(lengths) != 1:
        raise ValueError(f"prediction arrays have differing lengths: {sorted(lengths)}")
    return np.column_stack(arrays)


def blend(preds: list[FloatArray], weights: list[float] | None = None) -> FloatArray:
    """Weighted mean of prediction arrays; weights normalised to sum 1."""
    matrix = _stacked(preds)
    if weights is None:
        w = np.full(matrix.shape[1], 1.0 / matrix.shape[1])
    else:
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
    return matrix @ w


def rank_average(preds: list[FloatArray]) -> FloatArray:
    """Average of per-model normalised ranks — scale-invariant blending."""
    matrix = _stacked(preds)
    n = matrix.shape[0]
    ranks = np.empty_like(matrix)
    for j in range(matrix.shape[1]):
        order = np.argsort(matrix[:, j], kind="mergesort")
        r = np.empty(n, dtype=np.float64)
        r[order] = np.arange(n, dtype=np.float64)
        ranks[:, j] = r / (n - 1) if n > 1 else 0.0
    return ranks.mean(axis=1)


def stack(
    oof_preds: list[FloatArray],
    test_preds: list[FloatArray],
    y: FloatArray,
    folds: list[Fold],
) -> tuple[FloatArray, FloatArray]:
    """Linear stacker fit OOF-safely: the meta-model's own OOF view is produced
    with the same folds; the test view uses a meta-model fit on all OOF rows."""
    X_oof = _stacked(oof_preds)
    X_test = _stacked(test_preds)
    y = np.asarray(y, dtype=np.float64).ravel()

    meta_oof = np.empty(len(y), dtype=np.float64)
    for train_idx, valid_idx in folds:
        model = LinearRegression()
        model.fit(X_oof[train_idx], y[train_idx])
        meta_oof[valid_idx] = model.predict(X_oof[valid_idx])
    full = LinearRegression()
    full.fit(X_oof, y)
    return meta_oof, full.predict(X_test)
