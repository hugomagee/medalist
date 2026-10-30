"""E2: stack of sibling members. Arm A: fold-safe LR on member logits. Arm B: LightGBM
stacker (15 leaves, lr 0.03, fixed 400 rounds, no early stopping on the scored fold) on
member logits plus segment context (Type of Travel, Class, Customer Type); fold k is always
predicted by a stacker fit on the other folds' OOF rows. Returns the arm with the higher CV
(both recorded)."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from strategies.s6e10 import TARGET, base_frame, lgb_fit_predict, load_members, logit, stack_logistic

MEMBERS = []


def run(train_df, test_df, cv_splitter, metric):
    import lightgbm as lgb
    folds = cv_splitter.split(train_df)
    y = train_df[TARGET].to_numpy()
    O, T = load_members(MEMBERS)
    lr_oof, lr_test, coef = stack_logistic(O, T, y, folds)
    ctx = ["Type of Travel", "Class", "Customer Type"]
    Z = pd.DataFrame(logit(O), columns=MEMBERS)
    Zt = pd.DataFrame(logit(T), columns=MEMBERS)
    b, bt = base_frame(train_df), base_frame(test_df)
    for c in ctx:
        Z[c] = b[c].to_numpy()
        Zt[c] = bt[c].to_numpy()
    Z.columns = Zt.columns = [c.replace(" ", "_") for c in Z.columns]
    p = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 500,
         "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10,
         "verbosity": -1, "seed": 42, "num_threads": 5}
    gb_oof = np.zeros(len(y))
    gb_test = np.zeros(len(T))
    for tr, va in folds:
        m = lgb.train(p, lgb.Dataset(Z.iloc[tr], y[tr]), num_boost_round=400)
        gb_oof[va] = m.predict(Z.iloc[va])
        gb_test += m.predict(Zt) / len(folds)
    lr_cv = float(np.mean([roc_auc_score(y[v], lr_oof[v]) for _, v in folds]))
    gb_cv = float(np.mean([roc_auc_score(y[v], gb_oof[v]) for _, v in folds]))
    use_gb = gb_cv > lr_cv
    oof, test = (gb_oof, gb_test) if use_gb else (lr_oof, lr_test)
    extra = {"members": MEMBERS, "lr_cv": lr_cv, "lgb_stacker_cv": gb_cv, "returned": "lgb" if use_gb else "lr",
             "lr_coef": dict(zip(MEMBERS + ["intercept"], map(float, coef))),
             "member_auc": {m: float(roc_auc_score(y, O[:, j])) for j, m in enumerate(MEMBERS)}}
    np.save("lr_oof.npy", lr_oof); np.save("lr_test.npy", lr_test)
    np.save("gb_oof.npy", gb_oof); np.save("gb_test.npy", gb_test)
    return {"oof_predictions": oof, "test_predictions": test,
            "fold_scores": [float(roc_auc_score(y[v], oof[v])) for _, v in folds], "extra": extra}
