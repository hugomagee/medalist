"""A1b: does appending the 129,880 original rows help once the distance-key encodings exist?
Features: raw 21 + Kaggle-label TE of the distance/delay/digit keys (fold-safe; original rows
are encoded with the Kaggle fold-train mapping, so their own labels never enter) + is_original.
Arm A: Kaggle fold-train only. Arm B: + all original rows (returned). Same model."""
import numpy as np
import pandas as pd

from strategies.s6e10 import (DIGIT_KEYS, TARGET, V3_DIST_KEYS, base_frame, cv_loop, digit_frame,
                              int_key, key_features_fold, lgb_fit_predict, load_original)

PARAMS = {"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0}
KEYS = {**V3_DIST_KEYS, **DIGIT_KEYS}


def frame(df):
    b = base_frame(df).reset_index(drop=True)
    return pd.concat([b, digit_frame(b)], axis=1)


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    X, Xt = frame(train_df), frame(test_df)
    og = load_original()
    O = frame(og)
    yo = og[TARGET].to_numpy()
    y = train_df[TARGET].to_numpy()
    kx = {n: int_key(X, c) for n, c in KEYS.items()}
    kt = {n: int_key(Xt, c) for n, c in KEYS.items()}
    ko = {n: int_key(O, c) for n, c in KEYS.items()}
    X["is_original"], Xt["is_original"], O["is_original"] = 0, 0, 1
    arms = {}
    for use_orig in (False, True):
        def fold_fn(k, tr, va, use_orig=use_orig):
            ftr, (fva, fte, fo) = key_features_fold(
                {n: s.iloc[tr] for n, s in kx.items()}, y[tr],
                [{n: s.iloc[va] for n, s in kx.items()}, kt, ko])
            Xtr = pd.concat([X.iloc[tr].reset_index(drop=True), ftr], axis=1)
            Xva = pd.concat([X.iloc[va].reset_index(drop=True), fva], axis=1)
            Xte = pd.concat([Xt, fte], axis=1)
            ytr = y[tr]
            if use_orig:
                Xtr = pd.concat([Xtr, pd.concat([O, fo], axis=1)], ignore_index=True)
                ytr = np.concatenate([ytr, yo])
            pv, pt, it, imp = lgb_fit_predict(Xtr, ytr, Xva, y[va], Xte, PARAMS, early_stopping=200)
            return pv, pt, {"iters": it, "importances": imp}
        arms[use_orig] = cv_loop(y, folds, len(Xt), fold_fn, f"lgb-a1b-orig{int(use_orig)}")
    res = arms[True]
    res["extra"]["no_orig_cv"] = float(np.mean(arms[False]["fold_scores"]))
    res["extra"]["no_orig_folds"] = arms[False]["fold_scores"]
    res["extra"]["paired_gain"] = [a - b for a, b in zip(arms[True]["fold_scores"], arms[False]["fold_scores"])]
    return res
