"""Metric registry and verifier.

M1 exposes only the registered metric names; implementations land in M2.
"""

METRIC_NAMES: frozenset[str] = frozenset(
    {"mae", "rmse", "rmsle", "auc", "logloss", "accuracy", "f1"}
)
