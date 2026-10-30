"""A2d: a bank of 6 teachers trained only on the original (LightGBM on full / no-delay-
no-distance / ratings+cats / ratings-only subsets, CatBoost, one-hot logistic) adds 6 logit
features on top of the e0059 feature set."""
import pandas as pd

from strategies.s6e10 import (TARGET, FeatureBuilder, cv_loop, lgb_fit_predict,
                              load_original, base_frame, teacher_bank)

PARAMS = {"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0}


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    fb = FeatureBuilder(train_df, test_df, key_te=False, og_key_te=False, freq=False)
    og = load_original()
    (b_tr, b_te), aucs = teacher_bank(base_frame(og), og[TARGET].to_numpy(), [fb.X, fb.Xt])
    fb.S = pd.concat([fb.S, b_tr], axis=1)
    fb.St = pd.concat([fb.St, b_te], axis=1)
    y = fb.y

    def fold_fn(k, tr, va):
        Xtr, Xva, Xte = fb.fold(tr, va)
        pv, pt, it, imp = lgb_fit_predict(Xtr, y[tr], Xva, y[va], Xte, PARAMS, early_stopping=200)
        return pv, pt, {"iters": it, "importances": imp}

    res = cv_loop(y, folds, len(fb.Xt), fold_fn, "lgb-tbank")
    res["extra"]["teacher_aucs_on_original"] = aucs
    from sklearn.metrics import roc_auc_score
    res["extra"]["teacher_aucs_on_kaggle"] = {c: float(roc_auc_score(y, b_tr[c])) for c in b_tr.columns}
    return res
