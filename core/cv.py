"""Cross-validation policy engine (SPEC §8).

`make_folds` resolves the bundle policy to concrete fold indices so the
runner (parent) and the experiment (child) share the exact same splits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

logger = logging.getLogger(__name__)

IndexArray = NDArray[np.int64]
Fold = tuple[IndexArray, IndexArray]

MAX_CLASSIFICATION_CLASSES = 20


@dataclass(frozen=True)
class CVPlan:
    resolved_policy: str
    n_splits: int
    folds: list[Fold]


class PrecomputedSplitter:
    """Passed into experiment run(): yields the harness-fixed folds."""

    def __init__(self, folds: list[Fold]) -> None:
        self._folds = folds
        self.n_splits = len(folds)

    def split(self, X: object = None, y: object = None, groups: object = None) -> list[Fold]:
        return list(self._folds)

    def get_n_splits(self, X: object = None, y: object = None, groups: object = None) -> int:
        return self.n_splits


def _looks_like_classification(y: pd.Series) -> bool:
    if y.dtype.kind == "f" and not np.allclose(y.dropna() % 1, 0):
        return False
    return y.nunique() < MAX_CLASSIFICATION_CLASSES


def make_folds(
    train_df: pd.DataFrame,
    target_column: str,
    policy: str,
    cv_params: dict[str, Any],
    time_column: str | None = None,
    group_column: str | None = None,
) -> CVPlan:
    n_splits = int(cv_params.get("n_splits", 5))
    seed = int(cv_params.get("seed", 42))
    y = train_df[target_column]

    if policy == "auto":
        if time_column is not None:
            policy = "timeseries"
        elif group_column is not None:
            policy = "group"
        elif _looks_like_classification(y):
            policy = "stratified"
        else:
            policy = "kfold"
        logger.info("cv_policy auto resolved to '%s'", policy)

    folds: list[Fold]
    if policy == "kfold":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = [(t.astype(np.int64), v.astype(np.int64)) for t, v in splitter.split(train_df)]
    elif policy == "stratified":
        strat = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = [(t.astype(np.int64), v.astype(np.int64)) for t, v in strat.split(train_df, y)]
    elif policy == "group":
        if group_column is None:
            raise ValueError("group policy requires group_column")
        groups = train_df[group_column]
        gkf = GroupKFold(n_splits=n_splits)
        folds = [
            (t.astype(np.int64), v.astype(np.int64))
            for t, v in gkf.split(train_df, y, groups=groups)
        ]
    elif policy == "timeseries":
        if time_column is None:
            raise ValueError("timeseries policy requires time_column")
        # TimeSeriesSplit assumes row order == time order; sort, split, map back.
        time_order = np.argsort(train_df[time_column].to_numpy(), kind="mergesort")
        tss = TimeSeriesSplit(n_splits=n_splits)
        folds = [
            (
                np.sort(time_order[t]).astype(np.int64),
                np.sort(time_order[v]).astype(np.int64),
            )
            for t, v in tss.split(time_order)
        ]
    else:
        raise ValueError(f"unknown cv policy '{policy}'")

    return CVPlan(resolved_policy=policy, n_splits=len(folds), folds=folds)


def validate_alignment(
    oof: NDArray[np.float64],
    test_pred: NDArray[np.float64],
    n_train: int,
    n_test: int,
) -> None:
    """Row-count alignment checks before scoring (SPEC §8) — the classic silent killer."""
    if len(oof) != n_train:
        raise ValueError(
            f"misaligned OOF predictions: got {len(oof)} rows, train has {n_train}"
        )
    if len(test_pred) != n_test:
        raise ValueError(
            f"misaligned test predictions: got {len(test_pred)} rows, test has {n_test}"
        )
    if not np.all(np.isfinite(oof)) or not np.all(np.isfinite(test_pred)):
        raise ValueError("predictions contain NaN or inf")
