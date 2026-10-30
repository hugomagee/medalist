"""B3: per-value target encoding of Flight Distance (3,474 values; a GBM cannot isolate
single values) plus the delays and Distance x {Type of Travel, Class, Customer Type},
fold-safe with nested inner OOF on fold-train rows; plus value frequencies (train+test).
Added on top of the e0059 feature set."""
import pandas as pd

from strategies.s6e10 import FeatureBuilder, cv_loop, int_key, lgb_fit_predict

PARAMS = {"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0}
DIST_KEYS = {
    "D": ["Flight Distance"],
    "DD": ["Departure Delay in Minutes"],
    "AD": ["Arrival Delay in Minutes"],
    "D_T": ["Flight Distance", "Type of Travel"],
    "D_C": ["Flight Distance", "Class"],
    "D_CT": ["Flight Distance", "Customer Type"],
}


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    fb = FeatureBuilder(train_df, test_df, key_te=True, og_key_te=False, freq=False, keys=DIST_KEYS)
    for n, cols in DIST_KEYS.items():
        fr = pd.concat([int_key(fb.X, cols), int_key(fb.Xt, cols)]).value_counts()
        fb.S["freq_" + n] = int_key(fb.X, cols).map(fr).to_numpy()
        fb.St["freq_" + n] = int_key(fb.Xt, cols).map(fr).to_numpy()
    y = fb.y

    def fold_fn(k, tr, va):
        Xtr, Xva, Xte = fb.fold(tr, va)
        pv, pt, it, imp = lgb_fit_predict(Xtr, y[tr], Xva, y[va], Xte, PARAMS, early_stopping=200)
        return pv, pt, {"iters": it, "importances": imp}

    return cv_loop(y, folds, len(fb.Xt), fold_fn, "lgb-distte")
