"""A1: append all 129,880 original rows to every fold's TRAINING set (flag is_original=1).
Validation and test rows are Kaggle rows (flag 0); original rows never get scored.
Extra arm: same with original rows down-weighted to 0.5."""
import numpy as np
import pandas as pd

from strategies.s6e10 import TARGET, base_frame, cv_loop, lgb_fit_predict, load_original

PARAMS = {"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0}


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    X, Xt = base_frame(train_df), base_frame(test_df)
    og = load_original()
    Xo = base_frame(og)
    yo = og[TARGET].to_numpy()
    y = train_df[TARGET].to_numpy()
    X["is_original"] = 0
    Xt["is_original"] = 0
    Xo["is_original"] = 1
    out = {}
    for w_orig in (1.0, 0.5):
        def fold_fn(k, tr, va, w_orig=w_orig):
            Xtr = pd.concat([X.iloc[tr], Xo], ignore_index=True)
            ytr = np.concatenate([y[tr], yo])
            w = np.concatenate([np.ones(len(tr)), np.full(len(yo), w_orig)])
            pv, pt, it, imp = lgb_fit_predict(Xtr, ytr, X.iloc[va], y[va], Xt, PARAMS,
                                              early_stopping=200, weight=w)
            return pv, pt, {"iters": it, "importances": imp}
        out[w_orig] = cv_loop(y, folds, len(Xt), fold_fn, f"lgb-orig-w{w_orig}")
    res = out[1.0]
    res["extra"]["w0.5_cv"] = float(np.mean(out[0.5]["fold_scores"]))
    res["extra"]["w0.5_fold_scores"] = out[0.5]["fold_scores"]
    np.save("oof_w05.npy", out[0.5]["oof_predictions"])
    np.save("test_w05.npy", out[0.5]["test_predictions"])
    return res
