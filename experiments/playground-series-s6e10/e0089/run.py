"""D7: e0084 with 5 NEW seeds (combine with e0084 for an 8-seed bag); wide embedding MLP (as e0081) trained on Kaggle fold-train PLUS the 129,880 original
rows (features built without the original rows' own labels, via original_rows_v3 /
fold_with_original), with an is_original input. 3 seeds averaged."""
import numpy as np
import pandas as pd

from strategies.s6e10 import (FEATS, cv_loop, digit_frame, fb_v3_32, fold_with_original,
                              mlp_fit_predict, nn_codes, original_rows_v3)

TOK = ["dg_last3", "dg_first3", "dg_last2", "age_units"]
SEEDS = (11, 23, 37, 51, 67)


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    fb = fb_v3_32(train_df, test_df)
    O_static, k_o, yo = original_rows_v3(fb)
    O_raw = O_static[FEATS].reset_index(drop=True)
    O_tok = digit_frame(O_raw)[TOK]
    (C, Ct, Co), (N, Nt, No), card = nn_codes([fb.X[FEATS], fb.Xt[FEATS], O_raw],
                                              dist_bins=None, delay_cap=300)
    allv = np.concatenate([fb.X[TOK].to_numpy(), fb.Xt[TOK].to_numpy(), O_tok.to_numpy()])
    ec = []
    for j in range(len(TOK)):
        u, inv = np.unique(allv[:, j], return_inverse=True)
        ec.append(inv.astype(np.int64))
        card.append(len(u))
    E2 = np.column_stack(ec)
    n1, n2 = len(C), len(C) + len(Ct)
    C, Ct, Co = np.hstack([C, E2[:n1]]), np.hstack([Ct, E2[n1:n2]]), np.hstack([Co, E2[n2:]])
    y = fb.y

    def fold_fn(k, tr, va):
        Xtr, ytr, Xva, Xte = fold_with_original(fb, O_static, k_o, yo, tr, va)
        cols = [c for c in Xtr.columns if c not in FEATS and c not in TOK and not c.startswith("dg_")]
        A = [np.nan_to_num(M[cols].to_numpy(dtype=np.float64)) for M in (Xtr, Xva, Xte)]
        mu, sd = A[0].mean(0), A[0].std(0) + 1e-9
        A = [((M - mu) / sd).astype(np.float32) for M in A]
        Ctr = np.vstack([C[tr], Co])
        Ntr = np.hstack([np.vstack([N[tr], No]), A[0]])
        pvs, pts, eps = [], [], []
        for s in SEEDS:
            pv, pt, ep = mlp_fit_predict(Ctr, Ntr, ytr, C[va], np.hstack([N[va], A[1]]), y[va],
                                         Ct, np.hstack([Nt, A[2]]), card, emb_dim=16, seed=s,
                                         hidden=(1024, 512, 256), dropout=0.4, epochs=14)
            pvs.append(pv); pts.append(pt); eps.append(ep)
        return np.mean(pvs, axis=0), np.mean(pts, axis=0), {"best_epochs": eps}

    return cv_loop(y, folds, len(Ct), fold_fn, "mlp-orig-5s")
