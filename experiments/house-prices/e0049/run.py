"""Second hybrid with a different base: FGLS-weighted RBF SVR (nested grid),
plus a shallow GBR on its inner-CV residuals. Aimed at blend diversity."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.svm import SVR

from strategies.ames import LinearPrep, cv_log_target, tree_matrix

ALPHAS = np.logspace(-4.5, -2.0, 30)
INNER = KFold(5, shuffle=True, random_state=42)
GRID = {"C": [3, 10, 30], "gamma": [3e-5, 1e-4, 3e-4], "epsilon": [0.01, 0.03]}
GBR = dict(n_estimators=600, learning_rate=0.01, max_depth=3, max_features="sqrt",
           min_samples_leaf=20, subsample=0.7, loss="huber", random_state=42)
log = []


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
    gs = GridSearchCV(SVR(kernel="rbf"), GRID, cv=KFold(3, shuffle=True, random_state=42),
                      scoring="neg_root_mean_squared_error", n_jobs=-1).fit(A, y_fit, sample_weight=w)
    log.append(gs.best_params_)
    svr = SVR(kernel="rbf", **gs.best_params_)
    inner_res = y_fit - cross_val_predict(svr, A, y_fit, cv=INNER, params={"sample_weight": w})
    svr.fit(A, y_fit, sample_weight=w)
    Ta, Tb = tree_matrix(X_fit, X_pred)
    gbr = GradientBoostingRegressor(**GBR).fit(Ta.fillna(-1), inner_res, sample_weight=w)
    return svr.predict(B) + gbr.predict(Tb.fillna(-1))


def run(train_df, test_df, cv_splitter, metric):
    out = cv_log_target(train_df, test_df, cv_splitter.split(train_df), fit_predict)
    out["extra"]["svr_params"] = log
    return out
