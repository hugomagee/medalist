"""D2: embedding MLP (as e0060) whose numeric block also carries the e0059 original-label
features (single/pair TE, counts, teacher logit), standardized over train+test."""
import numpy as np

from strategies.s6e10 import FEATS, FeatureBuilder, cv_loop, mlp_fit_predict, nn_codes


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    fb = FeatureBuilder(train_df, test_df, key_te=False, og_key_te=False, freq=False)
    (C, Ct), (N, Nt), card = nn_codes([fb.X, fb.Xt])
    extra = [c for c in fb.S.columns if c not in FEATS]
    A = np.vstack([fb.S[extra].to_numpy(), fb.St[extra].to_numpy()]).astype(np.float64)
    A = ((A - A.mean(0)) / (A.std(0) + 1e-9)).astype(np.float32)
    N = np.hstack([N, A[: len(N)]])
    Nt = np.hstack([Nt, A[len(N):]])
    y = fb.y

    def fold_fn(k, tr, va):
        pv, pt, ep = mlp_fit_predict(C[tr], N[tr], y[tr], C[va], N[va], y[va], Ct, Nt, card)
        return pv, pt, {"best_epoch": ep}

    res = cv_loop(y, folds, len(Nt), fold_fn, "mlp-og")
    res["extra"]["n_numeric"] = int(N.shape[1])
    return res
