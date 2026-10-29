from strategies.baselines import constant_baseline


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    oof, test_pred = constant_baseline(train_df["SalePrice"], folds, n_test=len(test_df), strategy="median")
    return {"oof_predictions": oof, "test_predictions": test_pred, "fold_scores": []}
