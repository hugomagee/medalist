"""Hybrid v2 (variance reduction): FGLS Lasso base as e0046; the residual
booster is an average of 3 GBR seeds + 1 XGBoost, and the inner residuals are
averaged over two different inner KFold shuffles."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold, cross_val_predict

from strategies.ames import LinearPrep, cv_log_target, tree_matrix

ALPHAS = np.logspace(-4.5, -2.0, 30)
INNER = KFold(5, shuffle=True, random_state=42)
GBR = dict(n_estimators=600, learning_rate=0.01, max_depth=3, max_features="sqrt",
           min_samples_leaf=20, subsample=0.7, loss="huber")
XGB = dict(n_estimators=700, learning_rate=0.01, max_depth=3, subsample=0.7, colsample_bytree=0.4,
           min_child_weight=5, reg_lambda=5.0, random_state=42, n_jobs=4, tree_method="hist")


def _weights(A, y, X_fit):
    a1 = LassoCV(alphas=ALPHAS, cv=INNER, max_iter=50000, n_jobs=-1).fit(A, y).alpha_
    res = y - cross_val_predict(Lasso(alpha=a1, max_iter=50000), A, y, cv=INNER)
    grp = X_fit["SaleCondition"].where(X_fit["SaleCondition"].isin(["Normal", "Partial"]), "Other")
    var = pd.Series(res**2).groupby(grp.to_numpy()).mean()
    w = 1.0 / grp.map(var).to_numpy()
    return w / w.mean()


def fit_predict(X_fit, y_fit, X_pred):
    prep = LinearPrep().fit(X_fit)
    A, B = prep.transform(X_fit), prep.transform(X_pred)
    w = _weights(A, y_fit, X_fit)
    alpha = LassoCV(alphas=ALPHAS, cv=INNER, max_iter=50000, n_jobs=-1).fit(A, y_fit, sample_weight=w).alpha_
    lasso = Lasso(alpha=alpha, max_iter=50000)
    inner_res = np.mean([
        y_fit - cross_val_predict(lasso, A, y_fit, cv=KFold(5, shuffle=True, random_state=s),
                                  params={"sample_weight": w})
        for s in (42, 7)
    ], axis=0)
    lasso.fit(A, y_fit, sample_weight=w)
    Ta, Tb = tree_matrix(X_fit, X_pred)
    Ta, Tb = Ta.fillna(-1), Tb.fillna(-1)
    boosts = [GradientBoostingRegressor(**GBR, random_state=s).fit(Ta, inner_res, sample_weight=w).predict(Tb)
              for s in (42, 1, 2)]
    boosts.append(xgb.XGBRegressor(**XGB).fit(Ta, inner_res, sample_weight=w).predict(Tb))
    return lasso.predict(B) + np.mean(boosts, axis=0)


def run(train_df, test_df, cv_splitter, metric):
    return cv_log_target(train_df, test_df, cv_splitter.split(train_df), fit_predict)
