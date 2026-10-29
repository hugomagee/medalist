"""CatBoost with native categorical handling (ordered target statistics are
computed inside CatBoost on the fit set only)."""
import numpy as np
from catboost import CatBoostRegressor

from strategies.ames import categorical_columns, cv_log_target

PARAMS = dict(iterations=3000, learning_rate=0.03, depth=5, l2_leaf_reg=3, rsm=0.5,
              loss_function="RMSE", random_seed=42, verbose=0, thread_count=4)


def fit_predict(X_fit, y_fit, X_pred):
    cats = categorical_columns(X_fit)
    A, B = X_fit.copy(), X_pred.copy()
    A[cats] = A[cats].fillna("NA").astype(str)
    B[cats] = B[cats].fillna("NA").astype(str)
    return CatBoostRegressor(**PARAMS, cat_features=cats).fit(A, y_fit).predict(B)


def run(train_df, test_df, cv_splitter, metric):
    return cv_log_target(train_df, test_df, cv_splitter.split(train_df), fit_predict)
