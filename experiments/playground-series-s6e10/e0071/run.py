"""C7: XGBoost (hist, lossguide 127 leaves, eta 0.03) on the e0068 feature set (fb_v3)."""
from strategies.s6e10 import cv_loop, fb_v3, xgb_fit_predict

PARAMS = {"eta": 0.03, "grow_policy": "lossguide", "max_leaves": 127, "max_depth": 0,
          "min_child_weight": 10, "subsample": 0.75, "colsample_bytree": 0.5, "lambda": 5.0,
          "max_bin": 256}


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    fb = fb_v3(train_df, test_df)
    y = fb.y

    def fold_fn(k, tr, va):
        Xtr, Xva, Xte = fb.fold(tr, va)
        pv, pt, it, imp = xgb_fit_predict(Xtr, y[tr], Xva, y[va], Xte, PARAMS, early_stopping=300)
        return pv, pt, {"iters": it, "importances": imp}

    return cv_loop(y, folds, len(fb.Xt), fold_fn, "xgb-v3")
