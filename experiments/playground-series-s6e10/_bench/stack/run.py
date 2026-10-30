"""E: fold-safe logistic-regression stack on member logits (members = sibling experiments).
Fold k of the harness folds is predicted by an LR fit on the other folds' OOF rows; the
test prediction uses an LR fit on all OOF rows. Plain logit-mean and rank-mean blends and
every member's AUC are recorded in extra for comparison."""
import numpy as np
from sklearn.metrics import roc_auc_score

from strategies.s6e10 import TARGET, load_members, logit, rank_mean, stack_logistic

MEMBERS = []  # filled per experiment


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    y = train_df[TARGET].to_numpy()
    O, T = load_members(MEMBERS)
    oof, test, coef = stack_logistic(O, T, y, folds)
    extra = {
        "members": MEMBERS,
        "member_auc": {m: float(roc_auc_score(y, O[:, j])) for j, m in enumerate(MEMBERS)},
        "coef": dict(zip(MEMBERS + ["intercept"], map(float, coef))),
        "logit_mean_auc": float(roc_auc_score(y, logit(O).mean(1))),
        "rank_mean_auc": float(roc_auc_score(y, rank_mean(O))),
    }
    return {"oof_predictions": oof, "test_predictions": test,
            "fold_scores": [float(roc_auc_score(y[v], oof[v])) for _, v in folds], "extra": extra}
