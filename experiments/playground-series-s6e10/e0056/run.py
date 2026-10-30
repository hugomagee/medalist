"""B1: same LightGBM as e0052 but ratings, the 4 categoricals and Age are declared
categorical (native LightGBM categorical splits) instead of ordered numeric."""
from strategies.s6e10 import CATS, RATINGS, TARGET, base_frame, cv_loop, lgb_fit_predict

PARAMS = {"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
          "max_cat_to_onehot": 8, "cat_smooth": 10, "cat_l2": 10, "min_data_per_group": 50}
CAT_COLS = RATINGS + CATS + ["Age"]


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    X, Xt = base_frame(train_df), base_frame(test_df)
    for c in CAT_COLS:
        X[c] = X[c].astype(int)
        Xt[c] = Xt[c].astype(int)
    y = train_df[TARGET].to_numpy()

    def fold_fn(k, tr, va):
        pv, pt, it, imp = lgb_fit_predict(X.iloc[tr], y[tr], X.iloc[va], y[va], Xt, PARAMS,
                                          early_stopping=200, categorical=CAT_COLS)
        return pv, pt, {"iters": it, "importances": imp}

    return cv_loop(y, folds, len(Xt), fold_fn, "lgb-allcat")
