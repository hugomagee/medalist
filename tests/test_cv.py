import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from core.cv import make_folds, validate_alignment

SEED_PARAMS = {"n_splits": 4, "seed": 42}


def _regression_df(n: int) -> pd.DataFrame:
    r = np.random.default_rng(0)
    return pd.DataFrame({"id": range(n), "x": r.normal(size=n), "target": r.normal(size=n)})


def _classification_df(n: int, n_classes: int = 2) -> pd.DataFrame:
    r = np.random.default_rng(0)
    # ensure every class has >= n_splits members for stratification
    target = np.array([i % n_classes for i in range(n)])
    return pd.DataFrame({"id": range(n), "x": r.normal(size=n), "target": target})


class TestPolicyResolution:
    def test_auto_regression_is_kfold(self) -> None:
        plan = make_folds(_regression_df(40), "target", "auto", SEED_PARAMS)
        assert plan.resolved_policy == "kfold"

    def test_auto_few_class_target_is_stratified(self) -> None:
        plan = make_folds(_classification_df(40), "target", "auto", SEED_PARAMS)
        assert plan.resolved_policy == "stratified"

    def test_auto_many_class_int_target_is_kfold(self) -> None:
        df = _regression_df(100)
        df["target"] = range(100)  # 100 distinct values -> regression-like
        plan = make_folds(df, "target", "auto", SEED_PARAMS)
        assert plan.resolved_policy == "kfold"

    def test_auto_with_time_column_is_timeseries(self) -> None:
        df = _regression_df(40)
        df["ts"] = range(40)
        plan = make_folds(df, "target", "auto", SEED_PARAMS, time_column="ts")
        assert plan.resolved_policy == "timeseries"

    def test_auto_with_group_column_is_group(self) -> None:
        df = _regression_df(40)
        df["grp"] = [i // 5 for i in range(40)]
        plan = make_folds(df, "target", "auto", SEED_PARAMS, group_column="grp")
        assert plan.resolved_policy == "group"

    def test_explicit_policy_respected(self) -> None:
        plan = make_folds(_classification_df(40), "target", "kfold", SEED_PARAMS)
        assert plan.resolved_policy == "kfold"

    def test_deterministic_across_calls(self) -> None:
        a = make_folds(_regression_df(50), "target", "kfold", SEED_PARAMS)
        b = make_folds(_regression_df(50), "target", "kfold", SEED_PARAMS)
        for (ta, va), (tb, vb) in zip(a.folds, b.folds, strict=True):
            assert np.array_equal(ta, tb) and np.array_equal(va, vb)


# --- property tests (SPEC §8) ---------------------------------------------

n_samples = st.integers(min_value=20, max_value=120)
n_splits = st.integers(min_value=2, max_value=5)


@settings(max_examples=30, deadline=None)
@given(n=n_samples, k=n_splits)
def test_kfold_valid_indices_partition_exactly(n: int, k: int) -> None:
    plan = make_folds(_regression_df(n), "target", "kfold", {"n_splits": k, "seed": 1})
    all_valid = np.sort(np.concatenate([v for _, v in plan.folds]))
    assert np.array_equal(all_valid, np.arange(n))


@settings(max_examples=30, deadline=None)
@given(n=n_samples, k=n_splits)
def test_kfold_no_train_valid_overlap(n: int, k: int) -> None:
    plan = make_folds(_regression_df(n), "target", "kfold", {"n_splits": k, "seed": 1})
    for train_idx, valid_idx in plan.folds:
        assert len(np.intersect1d(train_idx, valid_idx)) == 0
        assert len(train_idx) + len(valid_idx) == n


@settings(max_examples=30, deadline=None)
@given(n=n_samples, k=st.integers(min_value=2, max_value=4))
def test_stratified_partitions_exactly(n: int, k: int) -> None:
    plan = make_folds(_classification_df(n), "target", "stratified", {"n_splits": k, "seed": 1})
    all_valid = np.sort(np.concatenate([v for _, v in plan.folds]))
    assert np.array_equal(all_valid, np.arange(n))


@settings(max_examples=30, deadline=None)
@given(n=n_samples, k=st.integers(min_value=2, max_value=4), group_size=st.integers(2, 8))
def test_group_folds_never_split_a_group(n: int, k: int, group_size: int) -> None:
    # GroupKFold requires n_groups >= n_splits; fewer groups is a caller error
    assume(-(-n // group_size) >= k)
    df = _regression_df(n)
    df["grp"] = [i // group_size for i in range(n)]
    plan = make_folds(df, "target", "group", {"n_splits": k, "seed": 1}, group_column="grp")
    groups = df["grp"].to_numpy()
    for train_idx, valid_idx in plan.folds:
        assert len(np.intersect1d(groups[train_idx], groups[valid_idx])) == 0


@settings(max_examples=30, deadline=None)
@given(n=n_samples, k=st.integers(min_value=2, max_value=4))
def test_time_folds_never_leak_future_into_train(n: int, k: int) -> None:
    df = _regression_df(n)
    # shuffled time column: make_folds must handle unsorted input
    r = np.random.default_rng(7)
    df["ts"] = r.permutation(n)
    plan = make_folds(df, "target", "timeseries", {"n_splits": k, "seed": 1}, time_column="ts")
    ts = df["ts"].to_numpy()
    for train_idx, valid_idx in plan.folds:
        assert ts[train_idx].max() <= ts[valid_idx].min()


class TestValidateAlignment:
    def test_accepts_aligned_finite_predictions(self) -> None:
        validate_alignment(np.zeros(10), np.zeros(5), n_train=10, n_test=5)

    def test_rejects_wrong_oof_length(self) -> None:
        with pytest.raises(ValueError, match="misaligned OOF"):
            validate_alignment(np.zeros(9), np.zeros(5), n_train=10, n_test=5)

    def test_rejects_wrong_test_length(self) -> None:
        with pytest.raises(ValueError, match="misaligned test"):
            validate_alignment(np.zeros(10), np.zeros(6), n_train=10, n_test=5)

    def test_rejects_nan_and_inf(self) -> None:
        bad = np.array([0.0, np.nan, 1.0])
        with pytest.raises(ValueError, match="NaN or inf"):
            validate_alignment(bad, np.zeros(5), n_train=3, n_test=5)
        bad_inf = np.array([0.0, np.inf, 1.0])
        with pytest.raises(ValueError, match="NaN or inf"):
            validate_alignment(np.zeros(3), bad_inf, n_train=3, n_test=3)
