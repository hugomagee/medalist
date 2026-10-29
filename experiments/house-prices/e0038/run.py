"""Ridge on the same design; alpha by inner 5-fold CV (nested)."""
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV

from strategies.ames import LinearPrep, cv_log_target

ALPHAS = np.logspace(-1, 2.5, 40)
chosen = []


def fit_predict(X_fit, y_fit, X_pred):
    prep = LinearPrep().fit(X_fit)
    A, B = prep.transform(X_fit), prep.transform(X_pred)
    alpha = RidgeCV(alphas=ALPHAS, cv=5).fit(A, y_fit).alpha_
    chosen.append(float(alpha))
    return Ridge(alpha=alpha).fit(A, y_fit).predict(B)


def run(train_df, test_df, cv_splitter, metric):
    out = cv_log_target(train_df, test_df, cv_splitter.split(train_df), fit_predict)
    out["extra"]["alphas"] = chosen
    return out
