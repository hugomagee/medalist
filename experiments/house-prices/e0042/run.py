"""LightGBM re-shaped for 1.4k rows: tiny trees, slow learning rate, heavy
column subsampling (most of the 120 features are weak)."""
import lightgbm as lgb

from strategies.ames import cv_log_target, tree_matrix

PARAMS = dict(objective="regression", num_leaves=6, learning_rate=0.01, n_estimators=5000,
              max_bin=200, subsample=0.75, subsample_freq=1, colsample_bytree=0.25,
              min_child_samples=8, reg_lambda=1.0, random_state=42, deterministic=True,
              force_row_wise=True, n_jobs=4, verbose=-1)


def fit_predict(X_fit, y_fit, X_pred):
    A, B = tree_matrix(X_fit, X_pred)
    return lgb.LGBMRegressor(**PARAMS).fit(A, y_fit).predict(B)


def run(train_df, test_df, cv_splitter, metric):
    return cv_log_target(train_df, test_df, cv_splitter.split(train_df), fit_predict)
