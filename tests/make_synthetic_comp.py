"""Generate a synthetic tabular regression competition for harness dry runs.

Nonlinear target with a categorical effect, plus a synthetic final
leaderboard so grading produces a percentile with zero human input.
Runnable directly: python -m tests.make_synthetic_comp <root>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

N_TRAIN = 2000
N_TEST = 500
N_TEAMS = 100


def _frame(n: int, rng: np.random.Generator, start_id: int) -> pd.DataFrame:
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    x4 = rng.uniform(-2, 2, size=n)
    cat = rng.integers(0, 10, size=n)
    cat_effect = np.array([3.0, -2.0, 0.5, 1.5, -1.0, 4.0, 0.0, -3.0, 2.0, -0.5])[cat]
    y = (
        5.0 * x1
        - 3.0 * np.square(x2)
        + 2.0 * x1 * x2
        + 1.5 * x4
        + cat_effect
        + rng.normal(scale=1.0, size=n)
    )
    return pd.DataFrame(
        {
            "id": range(start_id, start_id + n),
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "x4": x4,
            "cat": [f"c{c}" for c in cat],
            "target": y,
        }
    )


def make_synthetic_comp(
    root: Path,
    slug: str = "synthetic-mae",
    n_train: int = N_TRAIN,
    n_test: int = N_TEST,
    seed: int = 0,
) -> Path:
    rng = np.random.default_rng(seed)
    comp_dir = root / "competitions" / slug
    data_dir = comp_dir / "data"
    private_dir = comp_dir / "private"
    data_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(exist_ok=True)

    train = _frame(n_train, rng, start_id=0)
    test_full = _frame(n_test, rng, start_id=n_train)

    train.to_csv(data_dir / "train.csv", index=False)
    test_full.drop(columns=["target"]).to_csv(data_dir / "test.csv", index=False)
    pd.DataFrame({"id": test_full["id"], "target": train["target"].mean()}).to_csv(
        data_dir / "sample_submission.csv", index=False
    )
    test_full[["id", "target"]].to_csv(private_dir / "solution.csv", index=False)

    # synthetic final leaderboard: MAE scores from strong (~1.0, irreducible
    # noise) to weak (~ baseline MAE of a constant predictor)
    baseline_mae = float(np.abs(train["target"] - train["target"].mean()).mean())
    scores = np.linspace(1.0, baseline_mae, N_TEAMS)
    pd.DataFrame({"rank": range(1, N_TEAMS + 1), "score": scores}).to_csv(
        private_dir / "leaderboard.csv", index=False
    )

    bundle = {
        "slug": slug,
        "title": "Synthetic MAE dry-run competition",
        "metric": "mae",
        "metric_direction": "minimize",
        "target_column": "target",
        "id_column": "id",
        "files": {
            "train": "data/train.csv",
            "test": "data/test.csv",
            "sample_submission": "data/sample_submission.csv",
        },
        "cv_policy": "auto",
        "cv_params": {"n_splits": 5, "seed": 42},
        "time_column": None,
        "group_column": None,
        "leaderboard": "private/leaderboard.csv",
        "private_labels": "private/solution.csv",
    }
    (comp_dir / "bundle.yaml").write_text(yaml.safe_dump(bundle, sort_keys=False))
    return comp_dir


if __name__ == "__main__":
    target_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    print(f"wrote {make_synthetic_comp(target_root)}")
