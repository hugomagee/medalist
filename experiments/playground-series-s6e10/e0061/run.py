"""A2c: original-label target means for ALL 210 column pairs and the 20 triples of the 6
strongest columns (plus singles), int-keyed. Counts kept for singles/pairs only.
LightGBM with feature_fraction 0.5 to cope with ~480 columns."""
from itertools import combinations

import pandas as pd

from strategies.s6e10 import (FEATS, TARGET, base_frame, cv_loop, lgb_fit_predict,
                              load_original, og_stats_fast)

PARAMS = {"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
          "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0}
TOP6 = ["Online boarding", "Inflight wifi service", "Type of Travel", "Class",
        "Inflight entertainment", "Customer Type"]


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    X, Xt = base_frame(train_df), base_frame(test_df)
    og = load_original()
    O = base_frame(og)
    yo = og[TARGET].to_numpy()
    g12 = [[c] for c in FEATS] + [list(p) for p in combinations(FEATS, 2)]
    g3 = [list(t) for t in combinations(TOP6, 3)]
    a, b = og_stats_fast(O, yo, g12, [X, Xt])
    a3, b3 = og_stats_fast(O, yo, g3, [X, Xt], counts=False)
    X = pd.concat([X, a, a3], axis=1)
    Xt = pd.concat([Xt, b, b3], axis=1)
    y = train_df[TARGET].to_numpy()

    def fold_fn(k, tr, va):
        pv, pt, it, imp = lgb_fit_predict(X.iloc[tr], y[tr], X.iloc[va], y[va], Xt, PARAMS,
                                          early_stopping=200)
        return pv, pt, {"iters": it, "importances": imp}

    res = cv_loop(y, folds, len(Xt), fold_fn, "lgb-ogte-all")
    res["extra"]["n_features"] = X.shape[1]
    return res
