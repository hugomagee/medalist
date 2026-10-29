"""Huber-loss gradient boosting: shallow trees, sqrt feature sampling."""
from sklearn.ensemble import GradientBoostingRegressor

from strategies.ames import cv_log_target, tree_matrix

PARAMS = dict(n_estimators=3000, learning_rate=0.02, max_depth=4, max_features="sqrt",
              min_samples_leaf=15, min_samples_split=10, loss="huber", subsample=0.8,
              random_state=42)


def fit_predict(X_fit, y_fit, X_pred):
    A, B = tree_matrix(X_fit, X_pred)
    return GradientBoostingRegressor(**PARAMS).fit(A.fillna(-1), y_fit).predict(B.fillna(-1))


def run(train_df, test_df, cv_splitter, metric):
    return cv_log_target(train_df, test_df, cv_splitter.split(train_df), fit_predict)
