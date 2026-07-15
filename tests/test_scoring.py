import numpy as np
import pytest
import sklearn.metrics as skm

from core.scoring import METRIC_NAMES, METRICS, score

rng = np.random.default_rng(42)

N = 500
Y_REG = rng.uniform(1.0, 100.0, N)
PRED_REG = Y_REG + rng.normal(0, 5.0, N)
PRED_REG_POS = np.clip(PRED_REG, 0.1, None)
Y_BIN = rng.integers(0, 2, N)
PROB_BIN = np.clip(rng.beta(2, 2, N) * 0.5 + Y_BIN * 0.3, 0.01, 0.99)
LABEL_BIN = (PROB_BIN >= 0.5).astype(int)

TOL = 1e-12

REFERENCE_CASES = [
    ("mae", Y_REG, PRED_REG, skm.mean_absolute_error(Y_REG, PRED_REG)),
    ("rmse", Y_REG, PRED_REG, skm.root_mean_squared_error(Y_REG, PRED_REG)),
    ("rmsle", Y_REG, PRED_REG_POS, skm.root_mean_squared_log_error(Y_REG, PRED_REG_POS)),
    ("auc", Y_BIN, PROB_BIN, skm.roc_auc_score(Y_BIN, PROB_BIN)),
    ("logloss", Y_BIN, PROB_BIN, skm.log_loss(Y_BIN, PROB_BIN)),
    ("accuracy", Y_BIN, LABEL_BIN, skm.accuracy_score(Y_BIN, LABEL_BIN)),
    ("f1", Y_BIN, LABEL_BIN, skm.f1_score(Y_BIN, LABEL_BIN)),
]


@pytest.mark.parametrize("name,y_true,y_pred,expected", REFERENCE_CASES, ids=lambda v: str(v)[:12])
def test_metric_matches_sklearn_reference(
    name: str, y_true: np.ndarray, y_pred: np.ndarray, expected: float
) -> None:
    assert abs(score(name, y_true, y_pred) - expected) < TOL


def test_registry_contains_all_spec_metrics() -> None:
    assert set(METRICS) == {"mae", "rmse", "rmsle", "auc", "logloss", "accuracy", "f1"}
    assert METRIC_NAMES == frozenset(METRICS)


def test_unknown_metric_raises() -> None:
    with pytest.raises(KeyError):
        score("nope", Y_REG, PRED_REG)


def test_probability_predictions_thresholded_for_label_metrics() -> None:
    # accuracy/f1 on probabilities: harness thresholds at 0.5
    expected = skm.accuracy_score(Y_BIN, (PROB_BIN >= 0.5).astype(int))
    assert abs(score("accuracy", Y_BIN, PROB_BIN) - expected) < TOL


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        score("mae", Y_REG, PRED_REG[:-1])


def test_auc_handles_ties_like_sklearn() -> None:
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.5, 0.5, 0.5, 0.8, 0.2, 0.8])
    assert abs(score("auc", y, p) - skm.roc_auc_score(y, p)) < TOL
