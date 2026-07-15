import numpy as np
import pandas as pd

from core.cv import make_folds
from core.scoring import score
from strategies.baselines import constant_baseline, fit_predict_lgbm


def _signal_frame(n: int = 300, seed: int = 3) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    x1 = r.normal(size=n)
    x2 = r.normal(size=n)
    y = 3.0 * x1 - 2.0 * x2 + r.normal(scale=0.1, size=n)
    return pd.DataFrame({"id": range(n), "x1": x1, "x2": x2, "target": y})


class TestConstantBaseline:
    def test_mean_strategy_fold_safe(self) -> None:
        y = pd.Series([1.0, 2.0, 3.0, 4.0])
        folds = [
            (np.array([0, 1]), np.array([2, 3])),
            (np.array([2, 3]), np.array([0, 1])),
        ]
        oof, test_pred = constant_baseline(y, folds, n_test=2, strategy="mean")
        assert oof[2] == 1.5 and oof[3] == 1.5  # mean of fold-train rows 0,1
        assert oof[0] == 3.5 and oof[1] == 3.5
        assert np.all(test_pred == 2.5)  # full-train mean

    def test_median_strategy(self) -> None:
        y = pd.Series([1.0, 1.0, 10.0, 1.0])
        folds = [(np.array([0, 1, 2]), np.array([3]))]
        oof, test_pred = constant_baseline(y, folds, n_test=1, strategy="median")
        assert oof[3] == 1.0
        assert test_pred[0] == 1.0

    def test_mode_strategy(self) -> None:
        y = pd.Series([0.0, 0.0, 1.0, 0.0])
        folds = [(np.array([0, 1, 2]), np.array([3]))]
        oof, test_pred = constant_baseline(y, folds, n_test=1, strategy="mode")
        assert oof[3] == 0.0


class TestLgbm:
    def test_lgbm_beats_constant_baseline_on_signal(self) -> None:
        df = _signal_frame()
        test_df = _signal_frame(seed=4).drop(columns=["target"])
        folds = make_folds(df, "target", "kfold", {"n_splits": 4, "seed": 1}).folds
        oof_lgbm, test_pred, importances = fit_predict_lgbm(
            df, test_df, features=["x1", "x2"], target="target", folds=folds, task="regression"
        )
        oof_mean, _ = constant_baseline(df["target"], folds, n_test=len(test_df))
        y = df["target"].to_numpy()
        assert score("mae", y, oof_lgbm) < 0.5 * score("mae", y, oof_mean)
        assert len(test_pred) == len(test_df)
        assert set(importances) == {"x1", "x2"}

    def test_lgbm_deterministic(self) -> None:
        df = _signal_frame()
        test_df = _signal_frame(seed=4).drop(columns=["target"])
        folds = make_folds(df, "target", "kfold", {"n_splits": 3, "seed": 1}).folds
        a, _, _ = fit_predict_lgbm(df, test_df, ["x1", "x2"], "target", folds, task="regression")
        b, _, _ = fit_predict_lgbm(df, test_df, ["x1", "x2"], "target", folds, task="regression")
        assert np.allclose(a, b, atol=1e-9)

    def test_lgbm_classification_outputs_probabilities(self) -> None:
        df = _signal_frame()
        df["target"] = (df["target"] > 0).astype(int)
        test_df = _signal_frame(seed=4).drop(columns=["target"])
        folds = make_folds(df, "target", "stratified", {"n_splits": 3, "seed": 1}).folds
        oof, test_pred, _ = fit_predict_lgbm(
            df, test_df, ["x1", "x2"], "target", folds, task="classification"
        )
        assert np.all((oof >= 0) & (oof <= 1))
        assert score("auc", df["target"].to_numpy(), oof) > 0.9
