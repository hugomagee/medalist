import numpy as np
import pandas as pd
import pytest

from strategies.ames import (
    LinearPrep,
    anomaly_segment,
    base_frame,
    cv_log_target,
    training_outliers,
    tree_matrix,
)


def _frame(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = {
        "Id": np.arange(1, n + 1),
        "MSSubClass": rng.choice([20, 60], n),
        "LotFrontage": np.where(rng.random(n) < 0.2, np.nan, rng.integers(40, 100, n)),
        "GrLivArea": rng.integers(800, 3000, n),
        "Neighborhood": rng.choice(["Edwards", "NAmes", "OldTown"], n),
        "SaleCondition": rng.choice(["Normal", "Partial"], n),
        "OverallQual": rng.integers(3, 10, n),
        "OverallCond": rng.integers(3, 9, n),
        "YearBuilt": rng.integers(1950, 2008, n),
        "YrSold": rng.integers(2008, 2011, n),
        "MoSold": rng.integers(1, 13, n),
    }
    df = pd.DataFrame(cols)
    df["YearRemodAdd"] = df["YearBuilt"]
    df["GarageYrBlt"] = df["YearBuilt"].astype(float)
    for c in ["1stFlrSF", "2ndFlrSF", "TotalBsmtSF", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",
              "MasVnrArea", "GarageArea", "GarageCars", "PoolArea", "OpenPorchSF",
              "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF", "FullBath",
              "HalfBath", "BsmtFullBath", "BsmtHalfBath", "KitchenAbvGr", "Fireplaces"]:
        df[c] = rng.integers(0, 3, n)
    for c in ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
              "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC",
              "Fence", "MiscFeature", "MasVnrType"]:
        df[c] = np.nan
    for c, v in [("ExterQual", "TA"), ("ExterCond", "TA"), ("HeatingQC", "Gd"),
                 ("KitchenQual", "TA"), ("Functional", "Typ"), ("LotShape", "Reg"),
                 ("LandSlope", "Gtl"), ("PavedDrive", "Y"), ("CentralAir", "Y"),
                 ("Street", "Pave"), ("Utilities", "AllPub"), ("MSZoning", "RL"),
                 ("Exterior1st", "VinylSd"), ("Exterior2nd", "VinylSd"),
                 ("Electrical", "SBrkr"), ("SaleType", "WD")]:
        df[c] = v
    df["SalePrice"] = np.exp(11 + 0.0004 * df["GrLivArea"] + rng.normal(0, 0.1, n)).round()
    return df


def test_base_frame_fills_absent_features_and_drops_target():
    X = base_frame(_frame(50))
    assert "SalePrice" not in X.columns and "Id" not in X.columns
    assert (X["GarageQual"] == "None").all()
    assert X["TotalSF"].notna().all()


def test_linear_prep_handles_unseen_levels_and_missing_lotfrontage():
    train, test = base_frame(_frame(80, 1)), base_frame(_frame(20, 2))
    test.loc[0, "Neighborhood"] = "NeverSeen"
    test.loc[1, "MSZoning"] = np.nan
    prep = LinearPrep().fit(train)
    out = prep.transform(test)
    assert out.shape == (20, len(prep.feature_names))
    assert not np.isnan(out).any()


def test_tree_matrix_maps_unseen_to_minus_one():
    a, b = base_frame(_frame(30, 3)), base_frame(_frame(5, 4))
    b.loc[0, "Neighborhood"] = "NeverSeen"
    A, B = tree_matrix(a, b)
    assert B.loc[0, "Neighborhood"] == -1
    assert A.dtypes.eq(float).all()


def test_outlier_and_segment_masks():
    df = _frame(10)
    df.loc[0, ["GrLivArea", "SalePrice", "Neighborhood", "SaleCondition"]] = [5000, 150000, "Edwards", "Partial"]
    assert training_outliers(df)[0] and training_outliers(df).sum() == 1
    assert anomaly_segment(df)[0] and anomaly_segment(df).sum() == 1


def test_cv_log_target_never_fits_on_predicted_rows_or_outliers():
    df = _frame(60, 5)
    df.loc[3, ["GrLivArea", "SalePrice"]] = [5000, 100000]
    test = _frame(15, 6).drop(columns="SalePrice")
    idx = np.arange(60)
    folds = [(np.setdiff1d(idx, idx[k::3]), idx[k::3]) for k in range(3)]
    seen: list[int] = []

    def fit_predict(X_fit, y_fit, X_pred):
        seen.append(len(X_fit))
        return np.full(len(X_pred), y_fit.mean())

    res = cv_log_target(df, test, folds, fit_predict)
    assert len(res["oof_predictions"]) == 60 and len(res["test_predictions"]) == 15
    # fold containing row 3 as valid fits on 40 rows; the other two drop the outlier
    assert sorted(seen) == [39, 39, 40, 59]
    assert np.all(res["oof_predictions"] > 0)


@pytest.mark.parametrize("rule", [True, False])
def test_segment_rule_uses_fit_set_members(rule):
    df = _frame(30, 7)
    for i, price in [(0, 160000), (1, 185000)]:
        df.loc[i, ["GrLivArea", "SalePrice", "Neighborhood", "SaleCondition"]] = [4600, price, "Edwards", "Partial"]
    test = _frame(5, 8).drop(columns="SalePrice")
    test.loc[0, ["GrLivArea", "Neighborhood", "SaleCondition"]] = [5000, "Edwards", "Partial"]
    folds = [(np.arange(1, 30), np.array([0])), (np.r_[0, 2:30], np.array([1]))]
    res = cv_log_target(df, test, folds, lambda a, y, b: np.full(len(b), 13.0), anomaly_rule=rule)
    if rule:
        assert res["oof_predictions"][0] == pytest.approx(185000, rel=1e-9)
        assert res["test_predictions"][0] == pytest.approx(np.sqrt(160000 * 185000), rel=1e-3)
    else:
        assert res["test_predictions"][0] == pytest.approx(np.expm1(13.0))
