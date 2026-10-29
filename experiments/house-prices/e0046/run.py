"""Hybrid: FGLS Lasso (as e0040) + shallow GBR fitted on the Lasso's
inner-CV residuals, so the trees only learn interactions/nonlinearity the
linear model systematically misses (not its in-sample-shrunk residuals)."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold, cross_val_predict

from strategies.ames import LinearPrep, cv_log_target, tree_matrix

ALPHAS = np.logspace(-4.5, -2.0, 30)
INNER = KFold(5, shuffle=True, random_state=42)
GBR = dict(n_estimators=600, learning_rate=0.01, max_depth=3, max_features="sqrt",
           min_samples_leaf=20, subsample=0.7, loss="huber", random_state=42)


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
    inner_res = y_fit - cross_val_predict(lasso, A, y_fit, cv=INNER, params={"sample_weight": w})
    lasso.fit(A, y_fit, sample_weight=w)
    Ta, Tb = tree_matrix(X_fit, X_pred)
    gbr = GradientBoostingRegressor(**GBR).fit(Ta.fillna(-1), inner_res, sample_weight=w)
    return lasso.predict(B) + gbr.predict(Tb.fillna(-1))


def run(train_df, test_df, cv_splitter, metric):
    return cv_log_target(train_df, test_df, cv_splitter.split(train_df), fit_predict)
