"""C1: CatBoost with every raw column as a categorical (ordered target statistics +
CatBoost's automatic categorical combinations) plus numeric copies of the 4 numerics."""
from strategies.s6e10 import FEATS, NUMS, TARGET, base_frame, cat_fit_predict, cv_loop

PARAMS = {"learning_rate": 0.1, "depth": 7, "l2_leaf_reg": 3, "iterations": 4000,
          "border_count": 254, "one_hot_max_size": 4}


def frame(df):
    b = base_frame(df)
    out = b[NUMS].copy()
    for c in FEATS:
        out["c_" + c] = b[c].fillna(-1).astype(int).astype(str)
    return out


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    X, Xt = frame(train_df), frame(test_df)
    cats = [c for c in X.columns if c.startswith("c_")]
    y = train_df[TARGET].to_numpy()

    def fold_fn(k, tr, va):
        pv, pt, it, imp = cat_fit_predict(X.iloc[tr], y[tr], X.iloc[va], y[va], Xt, PARAMS,
                                          cat_features=cats, early_stopping=200)
        return pv, pt, {"iters": it, "importances": imp}

    return cv_loop(y, folds, len(Xt), fold_fn, "cat-allcat")
