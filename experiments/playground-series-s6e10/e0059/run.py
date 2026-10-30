"""A2b: a "teacher" model trained ONLY on the 129,880 original rows (its own 5-fold CV for
early stopping, then averaged) predicts every Kaggle train/test row; that prediction (and
its logit) is a feature for the Kaggle LightGBM. Only original labels enter the teacher,
so the feature is fold-safe for Kaggle rows. The single-column/pair original TE features of
e0055 are kept."""
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from strategies.s6e10 import (FEATS, TARGET, base_frame, cv_loop, lgb_fit_predict,
                              load_original, original_target_stats)

PARAMS = {"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0}
TEACHER = {"learning_rate": 0.03, "num_leaves": 127, "min_data_in_leaf": 20,
           "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0}
TOP = ["Online boarding", "Inflight wifi service", "Type of Travel", "Class", "Customer Type",
       "Inflight entertainment", "Seat comfort", "Leg room service", "Ease of Online booking", "Age"]


def teacher(O, yo, targets):
    """5-fold LightGBM on the original only; Kaggle rows get the mean of the 5 fold models."""
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    T = pd.concat(targets, ignore_index=True)
    pred = np.zeros(len(T))
    oof = np.zeros(len(O))
    for tr, va in skf.split(O, yo):
        pv, pt, it, _ = lgb_fit_predict(O.iloc[tr], yo[tr], O.iloc[va], yo[va], T, TEACHER,
                                        early_stopping=200)
        oof[va] = pv
        pred += pt / 5
    out, s = [], 0
    for t in targets:
        out.append(pred[s:s + len(t)])
        s += len(t)
    return out, oof


def run(train_df, test_df, cv_splitter, metric):
    from sklearn.metrics import roc_auc_score
    folds = cv_splitter.split(train_df)
    X, Xt = base_frame(train_df), base_frame(test_df)
    og = load_original()
    O = base_frame(og)
    yo = og[TARGET].to_numpy()
    (p_tr, p_te), oof_o = teacher(O, yo, [X, Xt])
    Ot = O.copy()
    Ot[TARGET] = yo
    groups = [[c] for c in FEATS] + [list(p) for p in combinations(TOP, 2)]
    add_tr, add_te = [], []
    for cols in groups:
        a, b = original_target_stats(Ot, cols, [X, Xt])
        add_tr.append(a)
        add_te.append(b)
    X = pd.concat([X] + add_tr, axis=1)
    Xt = pd.concat([Xt] + add_te, axis=1)
    eps = 1e-6
    X["og_model"] = np.log((p_tr + eps) / (1 - p_tr + eps))
    Xt["og_model"] = np.log((p_te + eps) / (1 - p_te + eps))
    y = train_df[TARGET].to_numpy()

    def fold_fn(k, tr, va):
        pv, pt, it, imp = lgb_fit_predict(X.iloc[tr], y[tr], X.iloc[va], y[va], Xt, PARAMS,
                                          early_stopping=200)
        return pv, pt, {"iters": it, "importances": imp}

    res = cv_loop(y, folds, len(Xt), fold_fn, "lgb-teacher")
    res["extra"]["teacher_oof_auc_on_original"] = float(roc_auc_score(yo, oof_o))
    res["extra"]["teacher_auc_on_kaggle_train"] = float(roc_auc_score(y, p_tr))
    np.save("teacher_train.npy", p_tr)
    np.save("teacher_test.npy", p_te)
    return res
