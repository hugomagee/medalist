"""B6: digit-token features of Flight Distance. Residuals of e0052 grouped by the LAST 3
digits (z=53) or FIRST 3 digits (z=43) are more overdispersed than by the exact value
(z=34): the generator ties labels to digit chunks. Adds digit codes (numeric), fold-safe
Kaggle TE (nested) and original-label TE for each digit key, on top of the e0063 set."""
import pandas as pd

from strategies.s6e10 import (DIGIT_KEYS, TARGET, FeatureBuilder, base_frame, cv_loop,
                              digit_frame, int_key, lgb_fit_predict, load_original, og_stats_fast)

PARAMS = {"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0}
DIST_KEYS = {
    "D": ["Flight Distance"], "DD": ["Departure Delay in Minutes"], "AD": ["Arrival Delay in Minutes"],
    "D_T": ["Flight Distance", "Type of Travel"], "D_C": ["Flight Distance", "Class"],
    "D_CT": ["Flight Distance", "Customer Type"],
}


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    fb = FeatureBuilder(train_df, test_df, key_te=False, og_key_te=False, freq=False)
    dtr, dte = digit_frame(fb.X), digit_frame(fb.Xt)
    fb.X = pd.concat([fb.X, dtr], axis=1)
    fb.Xt = pd.concat([fb.Xt, dte], axis=1)
    fb.S = pd.concat([fb.S, dtr], axis=1)
    fb.St = pd.concat([fb.St, dte], axis=1)
    og = load_original()
    O = base_frame(og)
    O = pd.concat([O, digit_frame(O)], axis=1)
    a, b = og_stats_fast(O, og[TARGET].to_numpy(), list(DIGIT_KEYS.values()), [fb.X, fb.Xt])
    fb.S = pd.concat([fb.S, a], axis=1)
    fb.St = pd.concat([fb.St, b], axis=1)
    keys = {**DIST_KEYS, **DIGIT_KEYS}
    fb.keys, fb.key_te = keys, True
    fb.k_tr = {n: int_key(fb.X, c) for n, c in keys.items()}
    fb.k_te = {n: int_key(fb.Xt, c) for n, c in keys.items()}
    for n in DIST_KEYS:
        fr = pd.concat([fb.k_tr[n], fb.k_te[n]]).value_counts()
        fb.S["freq_" + n] = fb.k_tr[n].map(fr).to_numpy()
        fb.St["freq_" + n] = fb.k_te[n].map(fr).to_numpy()
    y = fb.y

    def fold_fn(k, tr, va):
        Xtr, Xva, Xte = fb.fold(tr, va)
        pv, pt, it, imp = lgb_fit_predict(Xtr, y[tr], Xva, y[va], Xte, PARAMS, early_stopping=200)
        return pv, pt, {"iters": it, "importances": imp}

    return cv_loop(y, folds, len(fb.Xt), fold_fn, "lgb-digits")
