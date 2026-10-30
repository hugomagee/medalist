"""D3: embedding MLP with one embedding per EXACT Flight Distance value (~3.8k tokens) and
delays capped at 300, plus standardized original-label features, teacher logit and the
e0063-style fold-safe Kaggle TE of distance/delay keys (computed per fold)."""
import numpy as np

from strategies.s6e10 import (FEATS, FeatureBuilder, KEYS_V2, cv_loop, mlp_fit_predict, nn_codes)


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    fb = FeatureBuilder(train_df, test_df, key_te=True, og_key_te=True, freq=True, keys=KEYS_V2)
    (C, Ct), (N, Nt), card = nn_codes([fb.X, fb.Xt], dist_bins=None, delay_cap=300)
    y = fb.y

    def fold_fn(k, tr, va):
        Xtr, Xva, Xte = fb.fold(tr, va)
        extra = [c for c in Xtr.columns if c not in FEATS]
        A = [M[extra].to_numpy(dtype=np.float64) for M in (Xtr, Xva, Xte)]
        for M in A:
            np.nan_to_num(M, copy=False)
        mu, sd = A[0].mean(0), A[0].std(0) + 1e-9
        A = [((M - mu) / sd).astype(np.float32) for M in A]
        pv, pt, ep = mlp_fit_predict(C[tr], np.hstack([N[tr], A[0]]), y[tr],
                                     C[va], np.hstack([N[va], A[1]]), y[va],
                                     Ct, np.hstack([Nt, A[2]]), card, emb_dim=16)
        return pv, pt, {"best_epoch": ep}

    return cv_loop(y, folds, len(Ct), fold_fn, "mlp-v3")
