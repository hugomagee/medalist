"""LightGBM + Ames domain FE, outlier-free fit sets, Edwards partial-sale rule."""
import lightgbm as lgb

from strategies.ames import cv_log_target, tree_matrix
from strategies.baselines import lgbm_default_params

PARAMS = {**lgbm_default_params("regression")}


def fit_predict(X_fit, y_fit, X_pred):
    A, B = tree_matrix(X_fit, X_pred)
    return lgb.LGBMRegressor(**PARAMS).fit(A, y_fit).predict(B)


def run(train_df, test_df, cv_splitter, metric):
    return cv_log_target(train_df, test_df, cv_splitter.split(train_df), fit_predict)
