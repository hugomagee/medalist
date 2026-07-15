from pathlib import Path

import pandas as pd
import pytest
import yaml


def write_bundle(comp_dir: Path, **overrides: object) -> Path:
    """Write a minimal valid competition bundle into comp_dir; return bundle path."""
    data_dir = comp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train = pd.DataFrame(
        {
            "id": range(20),
            "feat_a": [float(i) for i in range(20)],
            "feat_b": [i % 3 for i in range(20)],
            "yield": [100.0 + i for i in range(20)],
        }
    )
    test = pd.DataFrame(
        {
            "id": range(20, 30),
            "feat_a": [float(i) for i in range(10)],
            "feat_b": [i % 3 for i in range(10)],
        }
    )
    sample = pd.DataFrame({"id": range(20, 30), "yield": [100.0] * 10})
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    sample.to_csv(data_dir / "sample_submission.csv", index=False)

    bundle: dict[str, object] = {
        "slug": comp_dir.name,
        "title": "Test Competition",
        "metric": "mae",
        "metric_direction": "minimize",
        "target_column": "yield",
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
        "leaderboard": None,
        "private_labels": None,
    }
    bundle.update(overrides)
    bundle_path = comp_dir / "bundle.yaml"
    bundle_path.write_text(yaml.safe_dump(bundle))
    return bundle_path


@pytest.fixture
def comp_dir(tmp_path: Path) -> Path:
    d = tmp_path / "competitions" / "test-comp"
    write_bundle(d)
    return d
