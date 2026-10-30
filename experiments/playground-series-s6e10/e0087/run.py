"""C12: numeric-only CatBoost (e0075 config) on fb_v3 WITH the original rows appended to
every fold's training set (features built without their own labels)."""
from strategies.s6e10 import cat_fit_predict, cv_loop, fb_v3_32, fold_with_original, original_rows_v3

PARAMS = {"learning_rate": 0.08, "depth": 8, "l2_leaf_reg": 5, "iterations": 6000,
          "border_count": 128, "bootstrap_type": "Bernoulli", "subsample": 0.8, "rsm": 0.5}


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    fb = fb_v3_32(train_df, test_df)
    O_static, k_o, yo = original_rows_v3(fb)
    y = fb.y

    def fold_fn(k, tr, va):
        Xtr, ytr, Xva, Xte = fold_with_original(fb, O_static, k_o, yo, tr, va)
        pv, pt, it, imp = cat_fit_predict(Xtr, ytr, Xva, y[va], Xte, PARAMS, cat_features=[],
                                          early_stopping=300)
        return pv, pt, {"iters": it, "importances": imp}

    return cv_loop(y, folds, len(fb.Xt), fold_fn, "cat-num-orig")
