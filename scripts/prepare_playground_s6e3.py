"""One-off prep for playground-series-s6e3 (see DECISIONS.md).

Kaggle never publishes Playground private test labels, so we carve a seeded
20% holdout from the original train BEFORE the agent loop sees the data:

  - original train/test/sample_submission -> private/original_*.csv
  - 80% of original train                 -> data/train.csv     (agent trains here)
  - 20% holdout, features only            -> data/test.csv      (operative test set)
  - 20% holdout, id + target              -> private/solution.csv
  - leaderboard export normalized: UTF-8 BOM stripped, 'Score' -> 'score'

Adaptations vs the s3e14 prep (binary churn classification, metric ROC AUC):
  - target 'Churn' is mapped Yes/No -> 1/0 everywhere the harness sees it
    (the metric registry needs numeric labels; the mapping is loss-free)
  - the holdout is STRATIFIED on the target (matches the bundle's stratified
    CV policy and preserves the 22.5% churn rate in both halves)
  - sample_submission filler is the train-side churn prevalence

Idempotent: refuses to run twice.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
HOLDOUT_FRAC = 0.20
COMP = Path(__file__).resolve().parents[1] / "competitions" / "playground-series-s6e3"
TARGET = "Churn"
ID = "id"


def main() -> None:
    data, private = COMP / "data", COMP / "private"
    if (private / "original_train.csv").exists():
        raise SystemExit("already prepared — refusing to run twice")

    full = pd.read_csv(data / "train.csv")
    mapped = full[TARGET].map({"Yes": 1, "No": 0})
    if mapped.isna().any():
        raise SystemExit(f"unexpected {TARGET} values: {full[TARGET].unique()}")
    full[TARGET] = mapped.astype("int64")

    train80, holdout = train_test_split(
        full, test_size=HOLDOUT_FRAC, stratify=full[TARGET], random_state=SEED
    )
    train80 = train80.sort_values(ID).reset_index(drop=True)
    holdout = holdout.sort_values(ID).reset_index(drop=True)

    # stash originals out of the agent's data/ view
    (data / "train.csv").rename(private / "original_train.csv")
    (data / "test.csv").rename(private / "original_kaggle_test.csv")
    (data / "sample_submission.csv").rename(private / "original_sample_submission.csv")

    train80.to_csv(data / "train.csv", index=False)
    holdout.drop(columns=[TARGET]).to_csv(data / "test.csv", index=False)
    pd.DataFrame({ID: holdout[ID], TARGET: float(train80[TARGET].mean())}).to_csv(
        data / "sample_submission.csv", index=False
    )
    holdout[[ID, TARGET]].to_csv(private / "solution.csv", index=False)

    lb = pd.read_csv(private / "leaderboard.csv", encoding="utf-8-sig")
    lb = lb.rename(columns={"Score": "score"})
    lb.to_csv(private / "leaderboard.csv", index=False)

    print(
        f"train={len(train80)} (churn {train80[TARGET].mean():.4f}) "
        f"holdout={len(holdout)} (churn {holdout[TARGET].mean():.4f}) "
        f"lb_teams={len(lb)} lb_cols={list(lb.columns)}"
    )


if __name__ == "__main__":
    main()
