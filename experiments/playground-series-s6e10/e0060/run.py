"""D1: per-feature embedding MLP (PyTorch MPS). Every feature is a categorical embedding
(Flight Distance quantile-binned, delays capped at 60) plus 5 standardized numerics."""
from strategies.s6e10 import TARGET, base_frame, cv_loop, mlp_fit_predict, nn_codes


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    X, Xt = base_frame(train_df), base_frame(test_df)
    (C, Ct), (N, Nt), card = nn_codes([X, Xt])
    y = train_df[TARGET].to_numpy()

    def fold_fn(k, tr, va):
        pv, pt, ep = mlp_fit_predict(C[tr], N[tr], y[tr], C[va], N[va], y[va], Ct, Nt, card)
        return pv, pt, {"best_epoch": ep}

    return cv_loop(y, folds, len(Xt), fold_fn, "mlp-emb")
