"""B9: fb_v4 + fold-safe KAGGLE-label target encodings (nested inner OOF, prior 20) of all 171
pairs of the 19 non-distance columns and the 56 triples of the top-8 columns. Distance pairs
are already covered by the per-value distance keys. LightGBM as e0088."""
from itertools import combinations

import pandas as pd

from strategies.s6e10 import FEATS, TOP8, cv_loop, fb_v4, kaggle_pair_te_fold, lgb_fit_predict

PARAMS = {"learning_rate": 0.02, "num_leaves": 127, "min_data_in_leaf": 100,
          "feature_fraction": 0.3, "bagging_fraction": 0.7, "bagging_freq": 1,
          "lambda_l2": 5.0, "max_bin": 255}
NOD = [c for c in FEATS if c != "Flight Distance"]
GROUPS = [list(p) for p in combinations(NOD, 2)] + [list(t) for t in combinations(TOP8, 3)]


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    fb = fb_v4(train_df, test_df)
    y = fb.y

    def fold_fn(k, tr, va):
        Xtr, Xva, Xte = fb.fold(tr, va)
        a, b, c = kaggle_pair_te_fold(fb, GROUPS, tr, va)
        Xtr, Xva, Xte = (pd.concat([Xtr, a], axis=1), pd.concat([Xva, b], axis=1),
                         pd.concat([Xte, c], axis=1))
        pv, pt, it, imp = lgb_fit_predict(Xtr, y[tr], Xva, y[va], Xte, PARAMS, early_stopping=400)
        return pv, pt, {"iters": it, "importances": imp}

    return cv_loop(y, folds, len(fb.Xt), fold_fn, "lgb-v4-kpairs")
