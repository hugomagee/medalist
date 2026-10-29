"""Feasible GLS Lasso. Residual variance differs ~4x between Normal sales and
Abnorml/Family/Alloca/AdjLand sales, so OLS-style equal weighting lets noisy
rows pull the coefficients. Per fold-train: pass 1 = nested LassoCV, estimate
residual variance per SaleCondition group from its inner-CV residuals, pass 2 =
LassoCV with weights 1/var_group (normalised). Predictions are unaffected by
the weights except through better coefficients."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold, cross_val_predict

from strategies.ames import LinearPrep, cv_log_target

ALPHAS = np.logspace(-4.5, -2.0, 30)
INNER = KFold(5, shuffle=True, random_state=42)
log = []


def fit_predict(X_fit, y_fit, X_pred):
    prep = LinearPrep().fit(X_fit)
    A, B = prep.transform(X_fit), prep.transform(X_pred)
    a1 = LassoCV(alphas=ALPHAS, cv=INNER, max_iter=50000, n_jobs=-1).fit(A, y_fit).alpha_
    res = y_fit - cross_val_predict(Lasso(alpha=a1, max_iter=50000), A, y_fit, cv=INNER)
    grp = X_fit["SaleCondition"].where(X_fit["SaleCondition"].isin(["Normal", "Partial"]), "Other")
    var = pd.Series(res**2).groupby(grp.to_numpy()).mean()
    w = 1.0 / grp.map(var).to_numpy()
    w = w / w.mean()
    a2 = LassoCV(alphas=ALPHAS, cv=INNER, max_iter=50000, n_jobs=-1).fit(A, y_fit, sample_weight=w).alpha_
    log.append({"alpha": float(a2), "var": {k: round(float(v), 5) for k, v in var.items()}})
    return Lasso(alpha=a2, max_iter=50000).fit(A, y_fit, sample_weight=w).predict(B)


def run(train_df, test_df, cv_splitter, metric):
    out = cv_log_target(train_df, test_df, cv_splitter.split(train_df), fit_predict)
    out["extra"]["fgls"] = log
    return out
