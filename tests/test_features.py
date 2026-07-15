import numpy as np
import pandas as pd

from core.cv import make_folds
from strategies.features import (
    add_group_aggregates,
    expand_datetime,
    impute_numeric_oof,
    target_encode_full_fit,
    target_encode_oof,
)


def _high_cardinality_frame(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    r = np.random.default_rng(seed)
    cat = pd.Series([f"c{i % 200}" for i in range(n)])  # ~2 rows per category
    y = pd.Series(r.normal(size=n))
    df = pd.DataFrame({"id": range(n), "cat": cat, "target": y})
    return df, y


def _folds(df: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    return make_folds(df, "target", "kfold", {"n_splits": 4, "seed": 1}).folds


class TestTargetEncoding:
    def test_fold_safe_differs_from_full_fit(self) -> None:
        df, y = _high_cardinality_frame()
        folds = _folds(df)
        oof_enc, _ = target_encode_oof(df["cat"], y, df["cat"], folds)
        full_enc = target_encode_full_fit(df["cat"], y, df["cat"])
        assert not np.allclose(oof_enc, full_enc)

    def test_full_fit_leaks_under_label_shuffle_but_fold_safe_does_not(self) -> None:
        # With shuffled labels there is NO real signal. A leaky encoding still
        # correlates with y (each row's own label contaminates its encoding);
        # the fold-safe encoding must not, beyond chance.
        df, _ = _high_cardinality_frame()
        r = np.random.default_rng(1)
        y_shuffled = pd.Series(r.permutation(df["target"].to_numpy()))
        df = df.assign(target=y_shuffled)
        folds = _folds(df)

        full_enc = target_encode_full_fit(df["cat"], y_shuffled, df["cat"])
        oof_enc, _ = target_encode_oof(df["cat"], y_shuffled, df["cat"], folds)

        leak_corr = abs(np.corrcoef(full_enc, y_shuffled)[0, 1])
        safe_corr = abs(np.corrcoef(oof_enc, y_shuffled)[0, 1])
        assert leak_corr > 0.3  # own-label contamination is visible
        assert safe_corr < 0.15  # fold-safe stays at chance level

    def test_unseen_category_gets_global_mean(self) -> None:
        train_cat = pd.Series(["a", "a", "b", "b"])
        y = pd.Series([1.0, 1.0, 3.0, 3.0])
        test_cat = pd.Series(["zzz"])
        _, test_enc = target_encode_oof(
            train_cat, y, test_cat, [(np.array([0, 1]), np.array([2, 3]))]
        )
        assert test_enc[0] == 2.0  # global mean

    def test_output_lengths(self) -> None:
        df, y = _high_cardinality_frame(100)
        folds = _folds(df)
        oof_enc, test_enc = target_encode_oof(df["cat"], y, df["cat"].iloc[:37], folds)
        assert len(oof_enc) == 100
        assert len(test_enc) == 37


class TestImputation:
    def test_fold_safe_imputation_uses_fold_means(self) -> None:
        col = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0, 7.0])
        folds = [
            (np.array([0, 1, 2]), np.array([3, 4, 5])),
            (np.array([3, 4, 5]), np.array([0, 1, 2])),
        ]
        oof_imputed, test_imputed = impute_numeric_oof(col, pd.Series([np.nan]), folds)
        # row 3 (valid in fold 0) filled with mean of rows 0..2 -> mean(1, 3) = 2
        assert oof_imputed[3] == 2.0
        # row 1 (valid in fold 1) filled with mean of rows 3..5 -> mean(5, 7) = 6
        assert oof_imputed[1] == 6.0
        # test filled with full-train mean: mean(1, 3, 5, 7) = 4
        assert test_imputed[0] == 4.0


class TestAggregates:
    def test_group_stats_added(self) -> None:
        train = pd.DataFrame({"g": ["a", "a", "b"], "x": [1.0, 3.0, 10.0]})
        test = pd.DataFrame({"g": ["a", "b"], "x": [0.0, 0.0]})
        train_out, test_out = add_group_aggregates(train, test, by="g", num_cols=["x"])
        assert train_out["x_mean_by_g"].tolist() == [2.0, 2.0, 10.0]
        assert test_out["x_mean_by_g"].tolist() == [2.0, 10.0]
        assert "x_std_by_g" in train_out.columns
        assert "g_count" in train_out.columns


class TestDatetime:
    def test_expand_datetime_columns(self) -> None:
        df = pd.DataFrame({"ts": pd.to_datetime(["2024-03-05 14:30", "2024-12-31 23:59"])})
        out = expand_datetime(df, "ts")
        assert out["ts_year"].tolist() == [2024, 2024]
        assert out["ts_month"].tolist() == [3, 12]
        assert out["ts_day"].tolist() == [5, 31]
        assert out["ts_dayofweek"].tolist() == [1, 1]  # both Tuesdays
        assert out["ts_hour"].tolist() == [14, 23]
