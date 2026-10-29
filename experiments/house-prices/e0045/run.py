"""Convex blend of diverse members on log scale; nested-CV-honest OOF."""
import numpy as np

from strategies.ames_blend import load_member, nested_blend

MEMBERS = ["e0037", "e0038", "e0039", "e0040", "e0041", "e0042", "e0043", "e0044"]


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    y = np.log1p(train_df["SalePrice"].to_numpy(dtype=np.float64))
    oof = np.column_stack([load_member(m, "oof.parquet", train_df["Id"].to_numpy()) for m in MEMBERS])
    test = np.column_stack([load_member(m, "test_pred.parquet", test_df["Id"].to_numpy()) for m in MEMBERS])
    b_oof, b_test, w, per_fold = nested_blend(oof, test, y, folds)
    corr = np.corrcoef((oof - y[:, None]).T)
    return {
        "oof_predictions": np.expm1(b_oof),
        "test_predictions": np.expm1(b_test),
        "fold_scores": [float(np.sqrt(np.mean((b_oof[v] - y[v]) ** 2))) for _, v in folds],
        "extra": {
            "weights": {m: round(float(x), 4) for m, x in zip(MEMBERS, w)},
            "member_rmse": {m: round(float(np.sqrt(np.mean((oof[:, i] - y) ** 2))), 5) for i, m in enumerate(MEMBERS)},
            "residual_corr": np.round(corr, 3).tolist(),
            "rmse_log_all": float(np.sqrt(np.mean((b_oof - y) ** 2))),
        },
    }
