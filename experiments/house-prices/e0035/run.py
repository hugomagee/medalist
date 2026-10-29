"""LightGBM on raw columns (categoricals as codes), log1p target, nothing else."""
import lightgbm as lgb
import numpy as np
import pandas as pd

from strategies.baselines import lgbm_default_params


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    feats = [c for c in train_df.columns if c not in ("Id", "SalePrice")]
    both = pd.concat([train_df[feats], test_df[feats]])
    for c in feats:
        if both[c].dtype == object or str(both[c].dtype) in ("str", "string"):
            both[c] = both[c].astype("category")
    X, Xt = both.iloc[: len(train_df)], both.iloc[len(train_df):]
    y = np.log1p(train_df["SalePrice"].to_numpy())
    params = lgbm_default_params("regression")
    oof = np.empty(len(X))
    tp = np.zeros(len(Xt))
    for tr, va in folds:
        m = lgb.LGBMRegressor(**params).fit(X.iloc[tr], y[tr])
        oof[va] = m.predict(X.iloc[va])
        tp += m.predict(Xt) / len(folds)
    return {"oof_predictions": np.expm1(oof), "test_predictions": np.expm1(tp), "fold_scores": []}
