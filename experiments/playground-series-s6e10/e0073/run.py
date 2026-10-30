"""D4: embedding MLP on the e0068 feature set: exact-distance embedding plus extra token
embeddings for distance digit chunks (last3, first3, last2) and age units; numeric block =
standardized fb_v3 engineered columns (incl. per-fold Kaggle TE). 3 seeds averaged."""
import numpy as np
from sklearn.metrics import roc_auc_score

from strategies.s6e10 import FEATS, cv_loop, fb_v3, mlp_fit_predict, nn_codes

TOK = ["dg_last3", "dg_first3", "dg_last2", "age_units"]
SEEDS = (42, 7, 2024)


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    fb = fb_v3(train_df, test_df)
    (C, Ct), (N, Nt), card = nn_codes([fb.X[FEATS], fb.Xt[FEATS]], dist_bins=None, delay_cap=300)
    extra_c, extra_card = [], []
    allv = np.concatenate([fb.X[TOK].to_numpy(), fb.Xt[TOK].to_numpy()])
    for j in range(len(TOK)):
        u, inv = np.unique(allv[:, j], return_inverse=True)
        extra_c.append(inv.astype(np.int64))
        extra_card.append(len(u))
    E2 = np.column_stack(extra_c)
    C = np.hstack([C, E2[: len(C)]])
    Ct = np.hstack([Ct, E2[len(C):]])
    card = card + extra_card
    y = fb.y
    per_seed = {s: np.zeros(len(y)) for s in SEEDS}

    def fold_fn(k, tr, va):
        Xtr, Xva, Xte = fb.fold(tr, va)
        cols = [c for c in Xtr.columns if c not in FEATS and c not in TOK and not c.startswith("dg_")]
        A = [np.nan_to_num(M[cols].to_numpy(dtype=np.float64)) for M in (Xtr, Xva, Xte)]
        mu, sd = A[0].mean(0), A[0].std(0) + 1e-9
        A = [((M - mu) / sd).astype(np.float32) for M in A]
        pvs, pts, eps = [], [], []
        for s in SEEDS:
            pv, pt, ep = mlp_fit_predict(C[tr], np.hstack([N[tr], A[0]]), y[tr],
                                         C[va], np.hstack([N[va], A[1]]), y[va],
                                         Ct, np.hstack([Nt, A[2]]), card, emb_dim=16, seed=s,
                                         epochs=14)
            per_seed[s][va] = pv
            pvs.append(pv); pts.append(pt); eps.append(ep)
        return np.mean(pvs, axis=0), np.mean(pts, axis=0), {"best_epochs": eps}

    res = cv_loop(y, folds, len(Ct), fold_fn, "mlp-v4")
    res["extra"]["per_seed_oof_auc"] = {str(s): float(roc_auc_score(y, v)) for s, v in per_seed.items()}
    return res
