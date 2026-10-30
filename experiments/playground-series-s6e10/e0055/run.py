"""A2: original-dataset target statistics as features. For every single column and every
pair among the 10 strongest columns, the ORIGINAL data's smoothed target mean and log count
per value are merged onto Kaggle train/test. Built from original labels only, so no fold
nesting is needed (Kaggle labels never enter the features)."""
from itertools import combinations

import pandas as pd

from strategies.s6e10 import (FEATS, TARGET, base_frame, cv_loop, lgb_fit_predict,
                              load_original, original_target_stats)

PARAMS = {"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0}
TOP = ["Online boarding", "Inflight wifi service", "Type of Travel", "Class", "Customer Type",
       "Inflight entertainment", "Seat comfort", "Leg room service", "Ease of Online booking", "Age"]


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    X, Xt = base_frame(train_df), base_frame(test_df)
    og = load_original()
    O = base_frame(og)
    O[TARGET] = og[TARGET].to_numpy()
    groups = [[c] for c in FEATS] + [list(p) for p in combinations(TOP, 2)]
    add_tr, add_te = [], []
    for cols in groups:
        a, b = original_target_stats(O, cols, [X, Xt])
        add_tr.append(a)
        add_te.append(b)
    X = pd.concat([X] + add_tr, axis=1)
    Xt = pd.concat([Xt] + add_te, axis=1)
    y = train_df[TARGET].to_numpy()

    def fold_fn(k, tr, va):
        pv, pt, it, imp = lgb_fit_predict(X.iloc[tr], y[tr], X.iloc[va], y[va], Xt, PARAMS,
                                          early_stopping=200)
        return pv, pt, {"iters": it, "importances": imp}

    res = cv_loop(y, folds, len(Xt), fold_fn, "lgb-ogte")
    res["extra"]["n_features"] = X.shape[1]
    return res
