"""C5: CatBoost with Flight Distance and both delays as CATEGORICAL features (plus the 4
categoricals), ordered target statistics and 2-way CTR combinations; ratings/Age numeric;
plus the e0059 original-label features (single/pair TE, teacher logit)."""
from strategies.s6e10 import CATS, FeatureBuilder, cat_fit_predict, cv_loop

PARAMS = {"learning_rate": 0.08, "depth": 8, "l2_leaf_reg": 3, "iterations": 5000,
          "border_count": 128, "max_ctr_complexity": 2, "one_hot_max_size": 4}
HC = ["Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]


def add_cats(df):
    out = df.copy()
    for c in HC:
        out["cat_" + c] = df[c].fillna(-1).astype(int).astype(str)
    return out


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    fb = FeatureBuilder(train_df, test_df, key_te=False, og_key_te=False, freq=False)
    fb.S, fb.St = add_cats(fb.S), add_cats(fb.St)
    cats = CATS + ["cat_" + c for c in HC]
    y = fb.y

    def fold_fn(k, tr, va):
        Xtr, Xva, Xte = fb.fold(tr, va)
        pv, pt, it, imp = cat_fit_predict(Xtr, y[tr], Xva, y[va], Xte, PARAMS,
                                          cat_features=cats, early_stopping=200)
        return pv, pt, {"iters": it, "importances": imp}

    return cv_loop(y, folds, len(fb.Xt), fold_fn, "cat-hc")
