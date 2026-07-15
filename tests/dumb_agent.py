"""Scripted 'dumb agent' (SPEC §13 M5): fixed experiment sequence
baseline -> LightGBM raw -> LightGBM + FE -> ensemble, then submit + grade.
Proves the harness end-to-end with zero LLM involvement.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from core.budget import Budget
from core.competition import load_bundle
from core.ledger import Ledger
from core.memory import write_memory
from core.runner import execute_experiment
from core.submission import Grade, build_submission, grade_submission, write_result_report

BASELINE_RUN = """
from strategies.baselines import constant_baseline

def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    oof, test_pred = constant_baseline(train_df["target"], folds, n_test=len(test_df))
    return {"oof_predictions": oof, "test_predictions": test_pred, "fold_scores": []}
"""

LGBM_RAW_RUN = """
from strategies.baselines import fit_predict_lgbm

def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    train_df = train_df.copy()
    test_df = test_df.copy()
    for frame in (train_df, test_df):
        frame["cat_code"] = frame["cat"].astype("category").cat.codes
    features = ["x1", "x2", "x3", "x4", "cat_code"]
    oof, test_pred, imp = fit_predict_lgbm(
        train_df, test_df, features, "target", folds, task="regression"
    )
    return {
        "oof_predictions": oof,
        "test_predictions": test_pred,
        "fold_scores": [],
        "feature_importances": imp,
    }
"""

LGBM_FE_RUN = """
from strategies.baselines import fit_predict_lgbm
from strategies.features import add_group_aggregates, target_encode_oof

def run(train_df, test_df, cv_splitter, metric):
    folds = cv_splitter.split(train_df)
    train_df = train_df.copy()
    test_df = test_df.copy()
    enc_oof, enc_test = target_encode_oof(
        train_df["cat"], train_df["target"], test_df["cat"], folds
    )
    train_df["cat_te"] = enc_oof
    test_df["cat_te"] = enc_test
    train_df, test_df = add_group_aggregates(train_df, test_df, by="cat", num_cols=["x1", "x2"])
    features = [c for c in train_df.columns if c not in ("id", "target", "cat")]
    oof, test_pred, imp = fit_predict_lgbm(
        train_df, test_df, features, "target", folds, task="regression"
    )
    return {
        "oof_predictions": oof,
        "test_predictions": test_pred,
        "fold_scores": [],
        "feature_importances": imp,
    }
"""

ENSEMBLE_RUN_TEMPLATE = """
import pandas as pd

from strategies.ensembling import blend

# sibling experiment dirs, relative to this experiment's cwd
MEMBERS = {members!r}

def run(train_df, test_df, cv_splitter, metric):
    oofs = [pd.read_parquet(m + "/oof.parquet")["prediction"].to_numpy() for m in MEMBERS]
    tests = [pd.read_parquet(m + "/test_pred.parquet")["prediction"].to_numpy() for m in MEMBERS]
    return {{
        "oof_predictions": blend(oofs),
        "test_predictions": blend(tests),
        "fold_scores": [],
    }}
"""

SEQUENCE = [
    (
        "e0001",
        BASELINE_RUN,
        "A fold-safe mean baseline establishes the floor score for this metric.",
        {"family": "baseline", "model_family": "constant"},
    ),
    (
        "e0002",
        LGBM_RAW_RUN,
        "LightGBM on raw features should comfortably beat the constant baseline.",
        {"family": "gbm", "model_family": "lightgbm"},
    ),
    (
        "e0003",
        LGBM_FE_RUN,
        "Fold-safe target encoding plus group aggregates should edge out raw features.",
        {"family": "gbm-fe", "model_family": "lightgbm"},
    ),
    (
        "e0004",
        None,  # rendered from ENSEMBLE_RUN_TEMPLATE with member paths
        "Blending the two diverse GBM OOF predictions should beat either alone.",
        {"family": "ensemble", "model_family": "blend", "ensemble_of": ["e0002", "e0003"]},
    ),
]


def run_dumb_agent(root: Path, slug: str) -> Grade:
    comp = load_bundle(root / "competitions" / slug)
    ledger = Ledger(root / "ledger.db")
    budget = Budget()
    exp_root = root / "experiments" / slug

    for exp_id, code, hypothesis, approach in SEQUENCE:
        exp_dir = exp_root / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        if code is None:
            members = [f"../{m}" for m in approach["ensemble_of"]]
            code = ENSEMBLE_RUN_TEMPLATE.format(members=members)
        (exp_dir / "run.py").write_text(code)
        (exp_dir / "meta.yaml").write_text(
            yaml.safe_dump({"hypothesis": hypothesis, "approach": approach})
        )
        record = execute_experiment(
            comp, exp_dir, ledger, budget, final=(approach["family"] == "ensemble")
        )
        if record.status != "completed":
            raise RuntimeError(f"{exp_id} did not complete: {record.status} {record.error}")
        write_memory(comp, ledger, budget)

    best = ledger.best(slug, comp.metric_direction)
    assert best is not None and best.artifact_dir is not None
    reports_dir = root / "reports" / slug
    submission = build_submission(
        comp, Path(best.artifact_dir) / "test_pred.parquet", reports_dir / "submission.csv"
    )
    grade = grade_submission(comp, submission)
    write_result_report(comp, grade, ledger, budget, reports_dir)
    return grade
