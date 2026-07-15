"""One-off prep for playground-s3e14 (see DECISIONS.md).

Kaggle never publishes Playground private test labels, so we carve a seeded
20% holdout from the original train BEFORE the agent loop sees the data:

  - original train/test/sample_submission -> private/original_*.csv
  - 80% of original train                 -> data/train.csv     (agent trains here)
  - 20% holdout, features only            -> data/test.csv      (operative test set)
  - 20% holdout, id + target              -> private/solution.csv
  - leaderboard export normalized: UTF-8 BOM stripped, 'Score' -> 'score'

Idempotent: refuses to run twice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
HOLDOUT_FRAC = 0.20
COMP = Path(__file__).resolve().parents[1] / "competitions" / "playground-s3e14"
TARGET = "yield"
ID = "id"


def main() -> None:
    data, private = COMP / "data", COMP / "private"
    if (private / "original_train.csv").exists():
        raise SystemExit("already prepared — refusing to run twice")

    full = pd.read_csv(data / "train.csv")
    rng = np.random.default_rng(SEED)
    n_holdout = int(round(len(full) * HOLDOUT_FRAC))
    holdout_idx = rng.choice(len(full), size=n_holdout, replace=False)
    mask = np.zeros(len(full), dtype=bool)
    mask[holdout_idx] = True
    holdout, train80 = full[mask].reset_index(drop=True), full[~mask].reset_index(drop=True)

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
        f"train={len(train80)} holdout={len(holdout)} "
        f"lb_teams={len(lb)} lb_cols={list(lb.columns)}"
    )


if __name__ == "__main__":
    main()
