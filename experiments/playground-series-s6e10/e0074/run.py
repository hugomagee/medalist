"""C8: LightGBM, higher capacity and slower (lr 0.01, 255 leaves, min_data 200, ff 0.4, L2 10) on the e0068
feature set (fb_v3)."""
from strategies.s6e10 import cv_loop, fb_v3, lgb_fit_predict

PARAMS = {"learning_rate": 0.01, "num_leaves": 255, "min_data_in_leaf": 200,
          "feature_fraction": 0.4, "bagging_fraction": 0.7, "bagging_freq": 1,
          "lambda_l2": 10.0, "max_bin": 255, "min_sum_hessian_in_leaf": 5.0}


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    fb = fb_v3(train_df, test_df)
    y = fb.y

    def fold_fn(k, tr, va):
        Xtr, Xva, Xte = fb.fold(tr, va)
        pv, pt, it, imp = lgb_fit_predict(Xtr, y[tr], Xva, y[va], Xte, PARAMS, early_stopping=400)
        return pv, pt, {"iters": it, "importances": imp}

    return cv_loop(y, folds, len(fb.Xt), fold_fn, "lgb-v3-xslow")
