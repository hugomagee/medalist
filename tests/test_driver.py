"""Direct tests for core._driver — the child process that runs inside the sandbox.

The runner tests exercise this module only through a subprocess, so its
result-shape validation and artifact writing had no direct coverage.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core._driver import main

GOOD_RUN = """
import numpy as np

def run(train_df, test_df, cv_splitter, metric):
    oof = np.zeros(len(train_df))
    for tr, va in cv_splitter.split():
        oof[va] = train_df["target"].iloc[tr].mean()
    scores = [metric(train_df["target"].iloc[va], oof[va]) for _, va in cv_splitter.split()]
    return {
        "oof_predictions": oof,
        "test_predictions": np.full(len(test_df), 1.5),
        "fold_scores": scores,
        "feature_importances": {"x": 3.0},
        "extra": {"note": "hello"},
    }
"""


def _make_payload(tmp_path: Path, run_source: str) -> tuple[Path, Path]:
    """Write run.py, data csvs and a driver payload; return (exp_dir, payload_path)."""
    exp_dir = tmp_path / "e0001"
    exp_dir.mkdir()
    (exp_dir / "run.py").write_text(run_source)

    train = pd.DataFrame(
        {"id": range(100, 110), "x": np.arange(10.0), "target": np.arange(10.0) * 2}
    )
    test = pd.DataFrame({"id": range(200, 205), "x": np.arange(5.0)})
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)

    payload = {
        "exp_dir": str(exp_dir),
        "train_path": str(tmp_path / "train.csv"),
        "test_path": str(tmp_path / "test.csv"),
        "folds": [
            [list(range(5)), list(range(5, 10))],
            [list(range(5, 10)), list(range(5))],
        ],
        "metric": "mae",
        "id_column": "id",
    }
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(payload))
    return exp_dir, payload_path


class TestHappyPath:
    def test_writes_aligned_prediction_artifacts(self, tmp_path: Path) -> None:
        exp_dir, payload_path = _make_payload(tmp_path, GOOD_RUN)
        main(str(payload_path))

        oof = pd.read_parquet(exp_dir / "oof.parquet")
        assert list(oof.columns) == ["id", "prediction"]
        assert oof["id"].tolist() == list(range(100, 110))

        test_pred = pd.read_parquet(exp_dir / "test_pred.parquet")
        assert test_pred["id"].tolist() == list(range(200, 205))
        assert (test_pred["prediction"] == 1.5).all()

    def test_child_result_carries_fold_scores_and_extra(self, tmp_path: Path) -> None:
        exp_dir, payload_path = _make_payload(tmp_path, GOOD_RUN)
        main(str(payload_path))

        child = json.loads((exp_dir / "child_result.json").read_text())
        assert len(child["claimed_fold_scores"]) == 2
        assert all(isinstance(s, float) for s in child["claimed_fold_scores"])
        assert child["extra"] == {"note": "hello"}

    def test_importances_written_when_provided(self, tmp_path: Path) -> None:
        exp_dir, payload_path = _make_payload(tmp_path, GOOD_RUN)
        main(str(payload_path))
        assert json.loads((exp_dir / "importances.json").read_text()) == {"x": 3.0}

    def test_no_importances_file_when_absent(self, tmp_path: Path) -> None:
        source = GOOD_RUN.replace('"feature_importances": {"x": 3.0},\n', "")
        exp_dir, payload_path = _make_payload(tmp_path, source)
        main(str(payload_path))
        assert not (exp_dir / "importances.json").exists()


class TestResultValidation:
    def test_run_py_without_run_fn_rejected(self, tmp_path: Path) -> None:
        _, payload_path = _make_payload(tmp_path, "X = 1\n")
        with pytest.raises(RuntimeError, match="must define run"):
            main(str(payload_path))

    def test_non_dict_result_rejected(self, tmp_path: Path) -> None:
        source = "def run(train_df, test_df, cv_splitter, metric):\n    return [1, 2, 3]\n"
        _, payload_path = _make_payload(tmp_path, source)
        with pytest.raises(RuntimeError, match="must return a dict"):
            main(str(payload_path))

    def test_result_missing_test_predictions_rejected(self, tmp_path: Path) -> None:
        source = (
            "def run(train_df, test_df, cv_splitter, metric):\n"
            "    return {'oof_predictions': [0.0] * len(train_df)}\n"
        )
        _, payload_path = _make_payload(tmp_path, source)
        with pytest.raises(RuntimeError, match="oof_predictions and test_predictions"):
            main(str(payload_path))


class TestMisalignment:
    def test_misaligned_oof_still_written_for_parent_to_fail(self, tmp_path: Path) -> None:
        # The driver must not crash on wrong-length predictions: it writes them
        # without the id column so the parent's alignment check fails the run.
        source = (
            "def run(train_df, test_df, cv_splitter, metric):\n"
            "    return {'oof_predictions': [0.0] * 3,\n"
            "            'test_predictions': [0.0] * len(test_df)}\n"
        )
        exp_dir, payload_path = _make_payload(tmp_path, source)
        main(str(payload_path))
        oof = pd.read_parquet(exp_dir / "oof.parquet")
        assert list(oof.columns) == ["prediction"]
        assert len(oof) == 3
