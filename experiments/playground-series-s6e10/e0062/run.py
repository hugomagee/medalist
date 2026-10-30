"""C2: XGBoost (hist, depthwise) on the e0059 feature set: raw + original-label single/pair
TE + teacher logit."""
from strategies.s6e10 import FeatureBuilder, cv_loop, xgb_fit_predict

PARAMS = {"eta": 0.05, "max_depth": 7, "min_child_weight": 5, "subsample": 0.8,
          "colsample_bytree": 0.7, "lambda": 2.0, "max_bin": 256}


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    fb = FeatureBuilder(train_df, test_df, key_te=False, og_key_te=False, freq=False)
    y = fb.y

    def fold_fn(k, tr, va):
        Xtr, Xva, Xte = fb.fold(tr, va)
        pv, pt, it, imp = xgb_fit_predict(Xtr, y[tr], Xva, y[va], Xte, PARAMS, early_stopping=200)
        return pv, pt, {"iters": it, "importances": imp}

    return cv_loop(y, folds, len(fb.Xt), fold_fn, "xgb-teacher")
