"""C9: CatBoost with NO categorical features (all fb_v3 columns numeric; the 4 categoricals
are already 0/1/2 codes), depth 8, lr 0.08: oblivious trees as a decorrelated GBM member
without the CTR cost that made e0057/e0067 too slow."""
from strategies.s6e10 import cat_fit_predict, cv_loop, fb_v3

PARAMS = {"learning_rate": 0.08, "depth": 8, "l2_leaf_reg": 5, "iterations": 6000,
          "border_count": 128, "bootstrap_type": "Bernoulli", "subsample": 0.8, "rsm": 0.5}


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    fb = fb_v3(train_df, test_df)
    y = fb.y

    def fold_fn(k, tr, va):
        Xtr, Xva, Xte = fb.fold(tr, va)
        pv, pt, it, imp = cat_fit_predict(Xtr, y[tr], Xva, y[va], Xte, PARAMS, cat_features=[],
                                          early_stopping=300)
        return pv, pt, {"iters": it, "importances": imp}

    return cv_loop(y, folds, len(fb.Xt), fold_fn, "cat-num")
