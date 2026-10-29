"""Lasso on fold-fit LinearPrep design; alpha picked by inner 5-fold CV on each
fold-train (nested, so the OOF score is not tuned on its own rows)."""
import numpy as np
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold

from strategies.ames import LinearPrep, cv_log_target

ALPHAS = np.logspace(-4.5, -2.0, 30)
chosen = []


def fit_predict(X_fit, y_fit, X_pred):
    prep = LinearPrep().fit(X_fit)
    A, B = prep.transform(X_fit), prep.transform(X_pred)
    cv = LassoCV(alphas=ALPHAS, cv=KFold(5, shuffle=True, random_state=42), max_iter=50000, n_jobs=-1).fit(A, y_fit)
    chosen.append(float(cv.alpha_))
    return Lasso(alpha=cv.alpha_, max_iter=50000).fit(A, y_fit).predict(B)


def run(train_df, test_df, cv_splitter, metric):
    out = cv_log_target(train_df, test_df, cv_splitter.split(train_df), fit_predict)
    out["extra"]["alphas"] = chosen
    return out
