"""Metric implementations + verifier registry.

Independent numpy implementations, each pinned to the sklearn reference
to 1e-12 by tests. auc/logloss expect probabilities; accuracy/f1 accept
labels or probabilities (thresholded at 0.5).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
MetricFn = Callable[[FloatArray, FloatArray], float]


def _as_arrays(y_true: object, y_pred: object) -> tuple[FloatArray, FloatArray]:
    t = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_pred, dtype=np.float64).ravel()
    if t.shape != p.shape:
        raise ValueError(f"length mismatch: y_true has {t.shape[0]} rows, y_pred {p.shape[0]}")
    return t, p


def _threshold_labels(p: FloatArray) -> FloatArray:
    unique = np.unique(p)
    if np.all(np.isin(unique, (0.0, 1.0))):
        return p
    return (p >= 0.5).astype(np.float64)


def mae(y_true: FloatArray, y_pred: FloatArray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: FloatArray, y_pred: FloatArray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def rmsle(y_true: FloatArray, y_pred: FloatArray) -> float:
    if np.any(y_true < 0) or np.any(y_pred < 0):
        raise ValueError("rmsle requires non-negative targets and predictions")
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def auc(y_true: FloatArray, y_pred: FloatArray) -> float:
    """ROC AUC via the rank-statistic (Mann-Whitney) formulation with average ranks."""
    pos = y_true == 1
    n_pos = int(pos.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("auc requires both classes present")
    order = np.argsort(y_pred, kind="mergesort")
    ranks = np.empty(len(y_pred), dtype=np.float64)
    sorted_pred = y_pred[order]
    i = 0
    while i < len(sorted_pred):
        j = i
        while j + 1 < len(sorted_pred) and sorted_pred[j + 1] == sorted_pred[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0  # average rank, 1-based
        i = j + 1
    rank_sum_pos = float(ranks[pos].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def logloss(y_true: FloatArray, y_pred: FloatArray) -> float:
    eps = np.finfo(np.float64).eps
    p = np.clip(y_pred, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def accuracy(y_true: FloatArray, y_pred: FloatArray) -> float:
    return float(np.mean(y_true == _threshold_labels(y_pred)))


def f1(y_true: FloatArray, y_pred: FloatArray) -> float:
    labels = _threshold_labels(y_pred)
    tp = float(np.sum((y_true == 1) & (labels == 1)))
    fp = float(np.sum((y_true == 0) & (labels == 1)))
    fn = float(np.sum((y_true == 1) & (labels == 0)))
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else 0.0


@dataclass(frozen=True)
class Metric:
    name: str
    fn: MetricFn
    direction: str  # conventional direction, informational


METRICS: dict[str, Metric] = {
    m.name: m
    for m in (
        Metric("mae", mae, "minimize"),
        Metric("rmse", rmse, "minimize"),
        Metric("rmsle", rmsle, "minimize"),
        Metric("auc", auc, "maximize"),
        Metric("logloss", logloss, "minimize"),
        Metric("accuracy", accuracy, "maximize"),
        Metric("f1", f1, "maximize"),
    )
}

METRIC_NAMES: frozenset[str] = frozenset(METRICS)


def score(metric_name: str, y_true: object, y_pred: object) -> float:
    """Compute a registered metric; the single scoring entrypoint for the harness."""
    metric = METRICS[metric_name]
    t, p = _as_arrays(y_true, y_pred)
    return metric.fn(t, p)
