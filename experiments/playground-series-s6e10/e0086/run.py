"""C10: e0070's LightGBM config (lr 0.02, 127 leaves, ff 0.5, L2 5) on fb_v3 with two NEW
seeds (bagging/feature sampling); returns their average. Combined with e0070 (seed 42) it
forms a 3-seed bag."""
import numpy as np
from sklearn.metrics import roc_auc_score

from strategies.s6e10 import cv_loop, fb_v3_32, lgb_fit_predict

PARAMS = {"learning_rate": 0.02, "num_leaves": 127, "min_data_in_leaf": 100,
          "feature_fraction": 0.5, "bagging_fraction": 0.7, "bagging_freq": 1,
          "lambda_l2": 5.0, "max_bin": 255}
SEEDS = (1, 2)


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    fb = fb_v3_32(train_df, test_df)
    y = fb.y
    per_seed = {s: np.zeros(len(y)) for s in SEEDS}

    def fold_fn(k, tr, va):
        Xtr, Xva, Xte = fb.fold(tr, va)
        pvs, pts, its = [], [], []
        for s in SEEDS:
            pv, pt, it, _ = lgb_fit_predict(Xtr, y[tr], Xva, y[va], Xte, dict(PARAMS, seed=s),
                                            early_stopping=400)
            per_seed[s][va] = pv
            pvs.append(pv); pts.append(pt); its.append(it)
        return np.mean(pvs, axis=0), np.mean(pts, axis=0), {"iters": its}

    res = cv_loop(y, folds, len(fb.Xt), fold_fn, "lgb-v3-slow-bag")
    res["extra"]["per_seed_oof_auc"] = {str(s): float(roc_auc_score(y, v)) for s, v in per_seed.items()}
    return res
