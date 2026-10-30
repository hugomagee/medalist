"""B4: widen the high-cardinality key encodings that e0063 showed to matter. Keys: Distance
alone and crossed with Type/Class/CustType/Gender and CustType x Type, Type x Class; delays
alone and Arrival Delay x {Leg room, Online boarding, Checkin, Seat comfort, Age}. Each key
gets a fold-safe Kaggle TE (nested), an ORIGINAL-label TE+count and a train+test frequency.
Two smoothing priors (20 and 5; e0063 used 10) are run as arms; the result returns prior 5 and
stores prior 20 as extra arrays so the comparison is on record."""
import numpy as np
import pandas as pd

from strategies.s6e10 import (TARGET, FeatureBuilder, base_frame, cv_loop, int_key,
                              key_features_fold, lgb_fit_predict, load_original, og_stats_fast)

PARAMS = {"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0}
D, AD, DD = "Flight Distance", "Arrival Delay in Minutes", "Departure Delay in Minutes"
KEYS = {
    "D": [D], "DD": [DD], "AD": [AD],
    "D_T": [D, "Type of Travel"], "D_C": [D, "Class"], "D_CT": [D, "Customer Type"],
    "D_G": [D, "Gender"], "D_CT_T": [D, "Customer Type", "Type of Travel"],
    "D_T_C": [D, "Type of Travel", "Class"],
    "AD_LR": [AD, "Leg room service"], "AD_OB": [AD, "Online boarding"],
    "AD_CK": [AD, "Checkin service"], "AD_SC": [AD, "Seat comfort"], "AD_A": [AD, "Age"],
}


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    fb = FeatureBuilder(train_df, test_df, key_te=False, og_key_te=False, freq=False)
    og = load_original()
    a, b = og_stats_fast(base_frame(og), og[TARGET].to_numpy(), list(KEYS.values()), [fb.X, fb.Xt],
                         prior=20.0)
    fb.S = pd.concat([fb.S, a.reset_index(drop=True).add_prefix("k_")], axis=1)
    fb.St = pd.concat([fb.St, b.reset_index(drop=True).add_prefix("k_")], axis=1)
    k_tr = {n: int_key(fb.X, c) for n, c in KEYS.items()}
    k_te = {n: int_key(fb.Xt, c) for n, c in KEYS.items()}
    for n in KEYS:
        fr = pd.concat([k_tr[n], k_te[n]]).value_counts()
        fb.S["freq_" + n] = k_tr[n].map(fr).to_numpy()
        fb.St["freq_" + n] = k_te[n].map(fr).to_numpy()
    y = fb.y
    arms = {}
    for prior in (20.0, 5.0):
        def fold_fn(k, tr, va, prior=prior):
            ftr, (fva, fte) = key_features_fold(
                {n: s.iloc[tr] for n, s in k_tr.items()}, y[tr],
                [{n: s.iloc[va] for n, s in k_tr.items()}, k_te], prior=prior)
            Xtr = pd.concat([fb.S.iloc[tr].reset_index(drop=True), ftr], axis=1)
            Xva = pd.concat([fb.S.iloc[va].reset_index(drop=True), fva], axis=1)
            Xte = pd.concat([fb.St, fte], axis=1)
            pv, pt, it, imp = lgb_fit_predict(Xtr, y[tr], Xva, y[va], Xte, PARAMS, early_stopping=200)
            return pv, pt, {"iters": it, "importances": imp}
        arms[prior] = cv_loop(y, folds, len(fb.Xt), fold_fn, f"lgb-keys-p{int(prior)}")
    res = arms[5.0]
    res["extra"]["prior20_cv"] = float(np.mean(arms[20.0]["fold_scores"]))
    res["extra"]["prior20_folds"] = arms[20.0]["fold_scores"]
    np.save("oof_prior20.npy", arms[20.0]["oof_predictions"])
    np.save("test_prior20.npy", arms[20.0]["test_predictions"])
    return res
