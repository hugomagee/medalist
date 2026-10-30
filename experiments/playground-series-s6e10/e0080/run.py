"""A1c: e0070 (LightGBM lr 0.02 on fb_v3) with all 129,880 ORIGINAL rows appended to every
fold's training set (is_original=1). Original rows' original-label features and teacher
logit are out-of-fold within the original; their Kaggle TE uses the Kaggle fold-train map.
Validation/test rows are Kaggle rows only, so the score is Kaggle-only OOF."""
from strategies.s6e10 import cv_loop, fb_v3_32, fold_with_original, lgb_fit_predict, original_rows_v3

PARAMS = {"learning_rate": 0.02, "num_leaves": 127, "min_data_in_leaf": 100,
          "feature_fraction": 0.5, "bagging_fraction": 0.7, "bagging_freq": 1,
          "lambda_l2": 5.0, "max_bin": 255}


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    fb = fb_v3_32(train_df, test_df)
    O_static, k_o, yo = original_rows_v3(fb)
    y = fb.y

    def fold_fn(k, tr, va):
        Xtr, ytr, Xva, Xte = fold_with_original(fb, O_static, k_o, yo, tr, va)
        pv, pt, it, imp = lgb_fit_predict(Xtr, ytr, Xva, y[va], Xte, PARAMS, early_stopping=400)
        return pv, pt, {"iters": it, "importances": imp}

    return cv_loop(y, folds, len(fb.Xt), fold_fn, "lgb-v3-orig")
