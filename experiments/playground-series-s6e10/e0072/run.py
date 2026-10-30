"""B7: density features on top of the e0068 set. For several profile keys (13 ratings;
ratings+cats; ratings+Class+Type; ratings+cats+Age; all but distance) the frequency of the
row's profile in Kaggle train+test and its count in the ORIGINAL (no labels involved), plus
the original-label TE of those profiles. No Kaggle-label TE on these keys (e0058 showed it
overfits)."""
import numpy as np
import pandas as pd

from strategies.s6e10 import (CATS, FEATS, HIGH_KEYS, TARGET, base_frame, cv_loop, fb_v3, int_key,
                              lgb_fit_predict, load_original)

PARAMS = {"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0}
PROFILES = {**HIGH_KEYS, "NOD": [c for c in FEATS if c != "Flight Distance"]}


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    fb = fb_v3(train_df, test_df)
    og = load_original()
    O = base_frame(og)
    yo = og[TARGET].to_numpy().astype(float)
    g = yo.mean()
    for n, cols in PROFILES.items():
        ktr, kte, ko = int_key(fb.X, cols), int_key(fb.Xt, cols), int_key(O, cols)
        fr = pd.concat([ktr, kte]).value_counts()
        st = pd.Series(yo).groupby(ko.to_numpy()).agg(["sum", "count"])
        te = (st["sum"] + 10 * g) / (st["count"] + 10)
        for F, k in ((fb.S, ktr), (fb.St, kte)):
            F["pfreq_" + n] = np.log1p(k.map(fr).to_numpy())
            F["pog_n_" + n] = np.log1p(k.map(st["count"]).fillna(0).to_numpy())
            F["pog_te_" + n] = k.map(te).fillna(g).to_numpy()
    y = fb.y

    def fold_fn(k, tr, va):
        Xtr, Xva, Xte = fb.fold(tr, va)
        pv, pt, it, imp = lgb_fit_predict(Xtr, y[tr], Xva, y[va], Xte, PARAMS, early_stopping=200)
        return pv, pt, {"iters": it, "importances": imp}

    return cv_loop(y, folds, len(fb.Xt), fold_fn, "lgb-density")
