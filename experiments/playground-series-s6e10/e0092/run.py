"""C13: CatBoost with ALL 21 raw columns as categorical features (ordered target statistics
plus CTR combinations up to 3 columns, which build high-order interaction encodings without
the self-label leakage that sank e0058), plus numeric inputs: the 4 raw numerics, the teacher
logit and the original-label single/pair TEs (FeatureBuilder, no Kaggle-label TE)."""
from strategies.s6e10 import FEATS, NUMS, FeatureBuilder, cat_fit_predict, cv_loop

PARAMS = {"learning_rate": 0.1, "depth": 6, "l2_leaf_reg": 3, "iterations": 3000,
          "max_ctr_complexity": 3, "one_hot_max_size": 4, "border_count": 128}


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    fb = FeatureBuilder(train_df, test_df, key_te=False, og_key_te=False, freq=False)
    og_cols = [c for c in fb.S.columns if c.startswith("og_") and c.endswith("_te")] + ["og_model"]

    def frame(S):
        out = S[NUMS + og_cols].astype("float32").copy()
        for c in FEATS:
            out["c_" + c] = S[c].fillna(-1).astype(int).astype(str)
        return out

    fb.S, fb.St = frame(fb.S), frame(fb.St)
    cats = ["c_" + c for c in FEATS]
    y = fb.y

    def fold_fn(k, tr, va):
        Xtr, Xva, Xte = fb.fold(tr, va)
        pv, pt, it, imp = cat_fit_predict(Xtr, y[tr], Xva, y[va], Xte, PARAMS, cat_features=cats,
                                          early_stopping=200)
        return pv, pt, {"iters": it, "importances": imp}

    return cv_loop(y, folds, len(fb.Xt), fold_fn, "cat-allcat-og")
