"""XGBoost depth-3 slow learner with row/column subsampling + L1/L2."""
import xgboost as xgb

from strategies.ames import cv_log_target, tree_matrix

PARAMS = dict(n_estimators=4000, learning_rate=0.01, max_depth=3, subsample=0.6,
              colsample_bytree=0.4, min_child_weight=2, gamma=0.0, reg_lambda=1.0,
              reg_alpha=0.005, random_state=42, n_jobs=4, tree_method="hist")


def fit_predict(X_fit, y_fit, X_pred):
    A, B = tree_matrix(X_fit, X_pred)
    return xgb.XGBRegressor(**PARAMS).fit(A, y_fit).predict(B)


def run(train_df, test_df, cv_splitter, metric):
    return cv_log_target(train_df, test_df, cv_splitter.split(train_df), fit_predict)
