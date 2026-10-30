"""LightGBM on the 22 raw features; seeds 42/1/2 measure the run-to-run noise floor."""
import numpy as np
from sklearn.metrics import roc_auc_score

from strategies.s6e10 import TARGET, base_frame, cv_loop, lgb_fit_predict

PARAMS = {"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0}


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    X, Xt = base_frame(train_df), base_frame(test_df)
    y = train_df[TARGET].to_numpy()
    results = {}
    for seed in (42, 1, 2):
        p = dict(PARAMS, seed=seed)

        def fold_fn(k, tr, va, p=p):
            pv, pt, it, imp = lgb_fit_predict(X.iloc[tr], y[tr], X.iloc[va], y[va], Xt, p,
                                              early_stopping=200)
            return pv, pt, {"iters": it, "importances": imp}

        results[seed] = cv_loop(y, folds, len(Xt), fold_fn, f"lgb-raw-s{seed}")
    out = results[42]
    seed_cv = {s: float(np.mean(r["fold_scores"])) for s, r in results.items()}
    paired = {s: [a - b for a, b in zip(r["fold_scores"], out["fold_scores"])] for s, r in results.items()}
    out["extra"]["seed_cv"] = seed_cv
    out["extra"]["seed_cv_std"] = float(np.std(list(seed_cv.values()), ddof=1))
    out["extra"]["paired_fold_diffs_vs_42"] = paired
    avg = np.mean([r["oof_predictions"] for r in results.values()], axis=0)
    out["extra"]["three_seed_avg_oof_auc"] = float(roc_auc_score(y, avg))
    return out
