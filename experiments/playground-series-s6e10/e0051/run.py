from strategies.baselines import constant_baseline


def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    oof, test = constant_baseline(train_df["satisfaction"], folds, len(test_df))
    return {"oof_predictions": oof, "test_predictions": test}
