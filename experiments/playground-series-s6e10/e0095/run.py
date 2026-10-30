"""C14: e0092 (CatBoost, all 21 columns categorical, CTR <= 3, + teacher logit and
original-label TEs) trained WITH the original rows appended (their original-label features
out-of-fold within the original, as in original_rows_v3)."""
import numpy as np
import pandas as pd

from strategies.s6e10 import (FEATS, NUMS, cat_fit_predict, cv_loop, fb_v3_32, fold_with_original,
                              original_rows_v3)

PARAMS = {"learning_rate": 0.1, "depth": 6, "l2_leaf_reg": 3, "iterations": 3000,
          "max_ctr_complexity": 3, "one_hot_max_size": 4, "border_count": 128, "random_seed": 7}


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    fb = fb_v3_32(train_df, test_df)
    O_static, k_o, yo = original_rows_v3(fb)
    og_cols = [c for c in fb.S.columns if c.startswith("og_") and c.endswith("_te")
               and not c.startswith("og_dg_")] + ["og_model"]
    keep = NUMS + og_cols + ["is_original"]
    y = fb.y

    def frame(M):
        out = M[keep].astype("float32").copy()
        for c in FEATS:
            out["c_" + c] = M[c].fillna(-1).astype(int).astype(str)
        return out

    def fold_fn(k, tr, va):
        Xtr, ytr, Xva, Xte = fold_with_original(fb, O_static, k_o, yo, tr, va)
        Xtr, Xva, Xte = frame(Xtr), frame(Xva), frame(Xte)
        cats = ["c_" + c for c in FEATS]
        pv, pt, it, imp = cat_fit_predict(Xtr, ytr, Xva, y[va], Xte, PARAMS, cat_features=cats,
                                          early_stopping=200)
        return pv, pt, {"iters": it, "importances": imp}

    return cv_loop(y, folds, len(fb.Xt), fold_fn, "cat-allcat-orig")
