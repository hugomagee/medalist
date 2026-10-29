"""Constrained blending on log predictions with nested-CV-honest scoring.

Weights are non-negative and sum to 1 (a convex combination, 1 dof per
member, little room to overfit). The blend's OOF row for fold k uses weights
fit on the OOF rows of the OTHER folds only, so the reported score is not
optimised on the rows it is measured on. Test weights use all OOF rows.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize  # type: ignore[import-untyped]

FloatArray = NDArray[np.float64]
Fold = tuple[NDArray[np.int64], NDArray[np.int64]]


def load_member(member: str, name: str, ids: NDArray[np.int64], id_col: str = "Id") -> FloatArray:
    frame = pd.read_parquet(f"../{member}/{name}").set_index(id_col).loc[ids]
    return np.asarray(np.log1p(frame["prediction"].to_numpy(dtype=np.float64)), dtype=np.float64)


def convex_weights(P: FloatArray, y: FloatArray) -> FloatArray:
    k = P.shape[1]

    def loss(w: FloatArray) -> float:
        return float(np.mean((P @ w - y) ** 2))

    res = minimize(
        loss, np.full(k, 1.0 / k), method="SLSQP", bounds=[(0.0, 1.0)] * k,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    w = np.clip(np.asarray(res.x, dtype=np.float64), 0, None)
    return np.asarray(w / w.sum(), dtype=np.float64)


def nested_blend(
    oof: FloatArray, test: FloatArray, y: FloatArray, folds: Sequence[Fold]
) -> tuple[FloatArray, FloatArray, FloatArray, list[FloatArray]]:
    out = np.empty(len(y))
    per_fold: list[FloatArray] = []
    for tr, va in folds:
        w = convex_weights(oof[tr], y[tr])
        out[va] = oof[va] @ w
        per_fold.append(w)
    w_all = convex_weights(oof, y)
    return out, test @ w_all, w_all, per_fold
