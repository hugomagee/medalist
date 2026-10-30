"""C6: LightGBM tuned for 700k rows (lr 0.02, 127 leaves, ff 0.5, L2 5) on the e0068
feature set (fb_v3)."""
from strategies.s6e10 import cv_loop, fb_v3, lgb_fit_predict

PARAMS = {"learning_rate": 0.02, "num_leaves": 127, "min_data_in_leaf": 100,
          "feature_fraction": 0.5, "bagging_fraction": 0.7, "bagging_freq": 1,
          "lambda_l2": 5.0, "max_bin": 255}


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    fb = fb_v3(train_df, test_df)
    y = fb.y

    def fold_fn(k, tr, va):
        Xtr, Xva, Xte = fb.fold(tr, va)
        pv, pt, it, imp = lgb_fit_predict(Xtr, y[tr], Xva, y[va], Xte, PARAMS, early_stopping=400)
        return pv, pt, {"iters": it, "importances": imp}

    return cv_loop(y, folds, len(fb.Xt), fold_fn, "lgb-v3-slow")
