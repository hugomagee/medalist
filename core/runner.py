"""Sandboxed experiment executor (SPEC §7).

The anti-cheating boundary: experiments run in a subprocess with cwd set to
the experiment dir and the competition's private/ directory made unreadable;
run.py is also static-checked for the string "private". The parent — never
the experiment — computes the fold scores recorded in the ledger.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from core.budget import Budget
from core.competition import Competition
from core.cv import CVPlan, make_folds, validate_alignment
from core.ledger import Experiment, Ledger, LedgerError
from core.scoring import score

STDERR_TAIL_CHARS = 2000


class _PrivateGuard:
    """Makes the competition's private/ dir unreadable while a child runs."""

    def __init__(self, comp_root: Path) -> None:
        self.private_dir = comp_root / "private"
        self._original_mode: int | None = None

    def __enter__(self) -> None:
        if self.private_dir.is_dir():
            self._original_mode = stat.S_IMODE(self.private_dir.stat().st_mode)
            os.chmod(self.private_dir, 0o000)

    def __exit__(self, *exc_info: object) -> None:
        if self._original_mode is not None:
            os.chmod(self.private_dir, self._original_mode or 0o755)


def _read_meta(exp_dir: Path) -> dict[str, Any]:
    meta_path = exp_dir / "meta.yaml"
    if not meta_path.is_file():
        raise LedgerError(f"no meta.yaml in {exp_dir}; it must declare hypothesis and approach")
    meta: dict[str, Any] = yaml.safe_load(meta_path.read_text()) or {}
    return meta


def _spawn(payload_path: Path, exp_dir: Path, timeout: float) -> tuple[int, str]:
    """Run the driver subprocess; returns (returncode, stderr tail). Raises TimeoutExpired."""
    env = dict(os.environ, PYTHONHASHSEED="42")
    proc = subprocess.Popen(
        [sys.executable, "-m", "core._driver", str(payload_path)],
        cwd=exp_dir,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()  # SIGKILL
        proc.wait()
        raise
    return proc.returncode, (stderr or "")[-STDERR_TAIL_CHARS:]


def _score_folds(
    comp: Competition, plan: CVPlan, train_df: pd.DataFrame, oof: np.ndarray
) -> list[float]:
    y = train_df[comp.target_column].to_numpy(dtype=np.float64)
    return [score(comp.metric, y[valid_idx], oof[valid_idx]) for _, valid_idx in plan.folds]


def execute_experiment(
    comp: Competition,
    exp_dir: Path,
    ledger: Ledger,
    budget: Budget,
    final: bool = False,
) -> Experiment:
    """Queue and execute one experiment; returns the final ledger row."""
    # resolve: the child runs with cwd=exp_dir, so relative paths (as passed
    # by the CLI, whose --root defaults to ".") would not survive the hop
    exp_dir = Path(exp_dir).resolve()
    meta = _read_meta(exp_dir)

    budget.check_can_queue(used=len(ledger.list(comp.slug)), final=final)
    exp_id = ledger.queue(
        comp.slug,
        hypothesis=str(meta.get("hypothesis", "")),
        approach=dict(meta.get("approach") or {}),
        parent_id=meta.get("parent_id"),
    )
    if exp_dir.name != exp_id:
        ledger.mark_running(exp_id)
        ledger.fail(exp_id, f"experiment dir '{exp_dir.name}' != assigned id '{exp_id}'", 0.0)
        return ledger.get(exp_id)

    ledger.mark_running(exp_id)
    start = time.monotonic()

    def _fail(message: str) -> Experiment:
        ledger.fail(exp_id, message, wall_seconds=time.monotonic() - start)
        return ledger.get(exp_id)

    run_source = (exp_dir / "run.py").read_text()
    if "private" in run_source.lower():
        return _fail(
            "static check failed: run.py contains the string 'private' — experiments "
            "must never touch the private/ directory"
        )

    train_df = comp.load_train()
    test_df = comp.load_test()
    plan = make_folds(
        train_df,
        comp.target_column,
        comp.cv_policy,
        comp.cv_params,
        time_column=comp.time_column,
        group_column=comp.group_column,
    )

    payload = {
        "exp_dir": str(exp_dir),
        "train_path": str(comp.files["train"].resolve()),
        "test_path": str(comp.files["test"].resolve()),
        "id_column": comp.id_column,
        "metric": comp.metric,
        "folds": [[t.tolist(), v.tolist()] for t, v in plan.folds],
    }
    payload_path = exp_dir / "payload.json"
    payload_path.write_text(json.dumps(payload))

    with _PrivateGuard(comp.root):
        try:
            returncode, stderr_tail = _spawn(
                payload_path, exp_dir, timeout=budget.per_experiment_seconds
            )
        except subprocess.TimeoutExpired:
            ledger.mark_timeout(exp_id, wall_seconds=time.monotonic() - start)
            return ledger.get(exp_id)

    if returncode != 0:
        return _fail(f"experiment subprocess exited {returncode}; stderr tail:\n{stderr_tail}")

    try:
        oof = pd.read_parquet(exp_dir / "oof.parquet")["prediction"].to_numpy(dtype=np.float64)
        test_pred = pd.read_parquet(exp_dir / "test_pred.parquet")["prediction"].to_numpy(
            dtype=np.float64
        )
    except FileNotFoundError as exc:
        return _fail(f"experiment produced no prediction artifacts: {exc}")

    try:
        validate_alignment(oof, test_pred, n_train=len(train_df), n_test=len(test_df))
    except ValueError as exc:
        return _fail(f"alignment validation failed: {exc}")

    fold_scores = _score_folds(comp, plan, train_df, oof)
    wall_seconds = time.monotonic() - start
    child_result_path = exp_dir / "child_result.json"
    child_result = (
        json.loads(child_result_path.read_text()) if child_result_path.is_file() else {}
    )
    result = {
        "exp_id": exp_id,
        "comp_slug": comp.slug,
        "resolved_cv_policy": plan.resolved_policy,
        "cv_score_mean": float(np.mean(fold_scores)),
        "cv_score_std": float(np.std(fold_scores)),
        "cv_fold_scores": fold_scores,
        "claimed_fold_scores": child_result.get("claimed_fold_scores"),
        "extra": child_result.get("extra"),
        "wall_seconds": wall_seconds,
    }
    (exp_dir / "result.json").write_text(json.dumps(result, indent=2))

    ledger.complete(
        exp_id,
        cv_score_mean=float(np.mean(fold_scores)),
        cv_score_std=float(np.std(fold_scores)),
        cv_fold_scores=fold_scores,
        wall_seconds=wall_seconds,
        artifact_dir=str(exp_dir),
    )
    return ledger.get(exp_id)
