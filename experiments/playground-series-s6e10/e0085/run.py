"""C11: XGBoost (e0071 config: lossguide 127 leaves, eta 0.03) on fb_v3 WITH the original
rows appended to every fold's training set (as e0080 does for LightGBM)."""
from strategies.s6e10 import cv_loop, fb_v3_32, fold_with_original, original_rows_v3, xgb_fit_predict

PARAMS = {"eta": 0.03, "grow_policy": "lossguide", "max_leaves": 127, "max_depth": 0,
          "min_child_weight": 10, "subsample": 0.75, "colsample_bytree": 0.5, "lambda": 5.0,
          "max_bin": 256}


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    fb = fb_v3_32(train_df, test_df)
    O_static, k_o, yo = original_rows_v3(fb)
    y = fb.y

    def fold_fn(k, tr, va):
        Xtr, ytr, Xva, Xte = fold_with_original(fb, O_static, k_o, yo, tr, va)
        pv, pt, it, imp = xgb_fit_predict(Xtr, ytr, Xva, y[va], Xte, PARAMS, early_stopping=300)
        return pv, pt, {"iters": it, "importances": imp}

    return cv_loop(y, folds, len(fb.Xt), fold_fn, "xgb-v3-orig")
