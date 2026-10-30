"""LightGBM + generic row-wise FE (rating sum/dispersion, delay diff, log delays)."""
from strategies.s6e10 import TARGET, base_frame, cv_loop, generic_fe, lgb_fit_predict

PARAMS = {"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0}


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    X, Xt = generic_fe(base_frame(train_df)), generic_fe(base_frame(test_df))
    y = train_df[TARGET].to_numpy()

    def fold_fn(k, tr, va):
        pv, pt, it, imp = lgb_fit_predict(X.iloc[tr], y[tr], X.iloc[va], y[va], Xt, PARAMS,
                                          early_stopping=200)
        return pv, pt, {"iters": it, "importances": imp}

    return cv_loop(y, folds, len(Xt), fold_fn, "lgb-fe")
