"""B2: high-order key encodings. Keys = full rating profile (13 ratings) with/without the
4 categoricals and Age. For each key: Kaggle-train target mean (fold-safe, nested inner
OOF on fold-train rows), ORIGINAL-data target mean and count, and the key's frequency in
Kaggle train+test (unsupervised)."""
import numpy as np
import pandas as pd

from strategies.s6e10 import (CATS, RATINGS, TARGET, base_frame, cv_loop, int_key,
                              key_features_fold, lgb_fit_predict, load_original)

PARAMS = {"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0}
KEYS = {
    "R": RATINGS,
    "RC": RATINGS + CATS,
    "RCT": RATINGS + ["Class", "Type of Travel"],
    "RCA": RATINGS + CATS + ["Age"],
}


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    X, Xt = base_frame(train_df), base_frame(test_df)
    og = load_original()
    O = base_frame(og)
    yo = og[TARGET].to_numpy().astype(float)
    y = train_df[TARGET].to_numpy()
    g = yo.mean()
    k_tr = {n: int_key(X, c).reset_index(drop=True) for n, c in KEYS.items()}
    k_te = {n: int_key(Xt, c).reset_index(drop=True) for n, c in KEYS.items()}
    for n, cols in KEYS.items():
        ko = int_key(O, cols)
        st = pd.Series(yo).groupby(ko.to_numpy()).agg(["sum", "count"])
        te = (st["sum"] + 10 * g) / (st["count"] + 10)
        for F, k in ((X, k_tr[n]), (Xt, k_te[n])):
            F["og_te_" + n] = k.map(te).fillna(g).to_numpy()
            F["og_n_" + n] = k.map(st["count"]).fillna(0).to_numpy()
        freq = pd.concat([k_tr[n], k_te[n]]).value_counts()
        X["freq_" + n] = k_tr[n].map(freq).to_numpy()
        Xt["freq_" + n] = k_te[n].map(freq).to_numpy()

    def fold_fn(k, tr, va):
        ftr, (fva, fte) = key_features_fold(
            {n: s.iloc[tr] for n, s in k_tr.items()}, y[tr],
            [{n: s.iloc[va] for n, s in k_tr.items()}, k_te])
        Xtr = pd.concat([X.iloc[tr].reset_index(drop=True), ftr], axis=1)
        Xva = pd.concat([X.iloc[va].reset_index(drop=True), fva], axis=1)
        Xte = pd.concat([Xt.reset_index(drop=True), fte], axis=1)
        pv, pt, it, imp = lgb_fit_predict(Xtr, y[tr], Xva, y[va], Xte, PARAMS, early_stopping=200)
        return pv, pt, {"iters": it, "importances": imp}

    return cv_loop(y, folds, len(Xt), fold_fn, "lgb-keyte")
