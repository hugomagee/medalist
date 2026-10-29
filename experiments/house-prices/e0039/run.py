"""RBF SVR on the robust-scaled design. Small grid on C/gamma/epsilon chosen by
inner 3-fold CV on each fold-train (nested)."""
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVR

from strategies.ames import LinearPrep, cv_log_target

GRID = {"C": [5, 20, 50], "gamma": [1e-4, 3e-4, 1e-3], "epsilon": [0.008, 0.02]}
chosen = []


def fit_predict(X_fit, y_fit, X_pred):
    prep = LinearPrep().fit(X_fit)
    A, B = prep.transform(X_fit), prep.transform(X_pred)
    gs = GridSearchCV(SVR(kernel="rbf"), GRID, cv=KFold(3, shuffle=True, random_state=42),
                      scoring="neg_root_mean_squared_error", n_jobs=-1).fit(A, y_fit)
    chosen.append(gs.best_params_)
    return gs.best_estimator_.predict(B)


def run(train_df, test_df, cv_splitter, metric):
    out = cv_log_target(train_df, test_df, cv_splitter.split(train_df), fit_predict)
    out["extra"]["params"] = chosen
    return out
