"""Child-process driver for one experiment.

Runs inside the sandbox subprocess (cwd = experiment dir, private/ made
unreadable by the parent). Loads data, reconstructs the harness-fixed CV
folds, calls the experiment's run(), and writes raw prediction artifacts.
The parent — never this process's experiment code — computes ledger scores.
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.cv import PrecomputedSplitter
from core.scoring import METRICS

GLOBAL_SEED = 42


def _load_run_fn(run_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("experiment_run", run_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {run_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run"):
        raise RuntimeError("run.py must define run(train_df, test_df, cv_splitter, metric)")
    return module.run


def _write_predictions(
    path: Path, id_col: str, ids: np.ndarray, predictions: np.ndarray
) -> None:
    # write even misaligned predictions; the parent validates and fails the run
    if len(predictions) == len(ids):
        frame = pd.DataFrame({id_col: ids, "prediction": predictions})
    else:
        frame = pd.DataFrame({"prediction": predictions})
    frame.to_parquet(path)


def main(payload_path: str) -> None:
    payload = json.loads(Path(payload_path).read_text())
    exp_dir = Path(payload["exp_dir"])

    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    train_df = pd.read_csv(payload["train_path"])
    test_df = pd.read_csv(payload["test_path"])
    folds = [
        (np.asarray(t, dtype=np.int64), np.asarray(v, dtype=np.int64))
        for t, v in payload["folds"]
    ]
    splitter = PrecomputedSplitter(folds)
    metric_fn = METRICS[payload["metric"]].fn

    def metric(y_true: Any, y_pred: Any) -> float:
        return metric_fn(
            np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
        )

    run = _load_run_fn(exp_dir / "run.py")
    result = run(train_df, test_df, splitter, metric)
    if not isinstance(result, dict) or not {"oof_predictions", "test_predictions"} <= set(result):
        raise RuntimeError(
            "run() must return a dict with at least oof_predictions and test_predictions"
        )

    id_col = payload["id_column"]
    oof = np.asarray(result["oof_predictions"], dtype=np.float64).ravel()
    test_pred = np.asarray(result["test_predictions"], dtype=np.float64).ravel()
    _write_predictions(exp_dir / "oof.parquet", id_col, train_df[id_col].to_numpy(), oof)
    _write_predictions(
        exp_dir / "test_pred.parquet", id_col, test_df[id_col].to_numpy(), test_pred
    )

    child_result: dict[str, Any] = {
        "claimed_fold_scores": [float(s) for s in result.get("fold_scores", [])],
        "extra": result.get("extra"),
    }
    (exp_dir / "child_result.json").write_text(json.dumps(child_result))
    importances = result.get("feature_importances")
    if importances is not None:
        (exp_dir / "importances.json").write_text(
            json.dumps({k: float(v) for k, v in dict(importances).items()})
        )


if __name__ == "__main__":
    main(sys.argv[1])
