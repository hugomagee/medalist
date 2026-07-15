"""Experiment ledger: append-only SQLite record of every experiment.

Rules (SPEC §6.2):
- Append-only: rows are never deleted; only status/score/notes transitions
  of an in-flight experiment are permitted (plus post-completion notes, §9.4).
- A hypothesis under 20 characters is rejected at queue time.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MIN_HYPOTHESIS_LEN = 20

_SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    exp_id TEXT PRIMARY KEY,
    comp_slug TEXT NOT NULL,
    parent_id TEXT,
    hypothesis TEXT NOT NULL,
    approach TEXT NOT NULL,
    status TEXT NOT NULL,
    cv_score_mean REAL,
    cv_score_std REAL,
    cv_fold_scores TEXT,
    wall_seconds REAL,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    artifact_dir TEXT,
    error TEXT,
    notes TEXT
);
"""


class LedgerError(Exception):
    """Raised on any violation of ledger rules."""


@dataclass(frozen=True)
class Experiment:
    exp_id: str
    comp_slug: str
    parent_id: str | None
    hypothesis: str
    approach: dict[str, Any]
    status: str
    cv_score_mean: float | None
    cv_score_std: float | None
    cv_fold_scores: list[float] | None
    wall_seconds: float | None
    created_at: str
    finished_at: str | None
    artifact_dir: str | None
    error: str | None
    notes: str | None


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Ledger:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # -- writes ------------------------------------------------------------

    def queue(
        self,
        comp_slug: str,
        hypothesis: str,
        approach: dict[str, Any],
        parent_id: str | None = None,
    ) -> str:
        if len(hypothesis.strip()) < MIN_HYPOTHESIS_LEN:
            raise LedgerError(
                f"hypothesis must be at least {MIN_HYPOTHESIS_LEN} characters "
                f"(got {len(hypothesis.strip())}); state what this experiment tests and why"
            )
        with self._connect() as conn:
            if parent_id is not None:
                row = conn.execute(
                    "SELECT 1 FROM experiments WHERE exp_id = ?", (parent_id,)
                ).fetchone()
                if row is None:
                    raise LedgerError(f"parent experiment '{parent_id}' does not exist")
            (count,) = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()
            exp_id = f"e{count + 1:04d}"
            conn.execute(
                "INSERT INTO experiments "
                "(exp_id, comp_slug, parent_id, hypothesis, approach, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, 'queued', ?)",
                (exp_id, comp_slug, parent_id, hypothesis, json.dumps(approach), _now()),
            )
        return exp_id

    def mark_running(self, exp_id: str) -> None:
        self._transition(exp_id, from_status=("queued",), set_sql="status = 'running'", params=())

    def complete(
        self,
        exp_id: str,
        cv_score_mean: float,
        cv_score_std: float,
        cv_fold_scores: list[float],
        wall_seconds: float,
        artifact_dir: str,
    ) -> None:
        self._transition(
            exp_id,
            from_status=("running",),
            set_sql=(
                "status = 'completed', cv_score_mean = ?, cv_score_std = ?, "
                "cv_fold_scores = ?, wall_seconds = ?, artifact_dir = ?, finished_at = ?"
            ),
            params=(
                cv_score_mean,
                cv_score_std,
                json.dumps(cv_fold_scores),
                wall_seconds,
                artifact_dir,
                _now(),
            ),
        )

    def fail(self, exp_id: str, error: str, wall_seconds: float) -> None:
        self._transition(
            exp_id,
            from_status=("running",),
            set_sql="status = 'failed', error = ?, wall_seconds = ?, finished_at = ?",
            params=(error, wall_seconds, _now()),
        )

    def mark_timeout(self, exp_id: str, wall_seconds: float) -> None:
        self._transition(
            exp_id,
            from_status=("running",),
            set_sql="status = 'timeout', wall_seconds = ?, finished_at = ?",
            params=(wall_seconds, _now()),
        )

    def set_notes(self, exp_id: str, notes: str) -> None:
        self._transition(
            exp_id,
            from_status=("running", "completed", "failed", "timeout"),
            set_sql="notes = ?",
            params=(notes,),
        )

    def _transition(
        self, exp_id: str, from_status: tuple[str, ...], set_sql: str, params: tuple[Any, ...]
    ) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM experiments WHERE exp_id = ?", (exp_id,)
            ).fetchone()
            if row is None:
                raise LedgerError(f"experiment '{exp_id}' does not exist")
            (status,) = row
            if status not in from_status:
                raise LedgerError(
                    f"experiment '{exp_id}' is '{status}'; "
                    f"this transition requires status in {from_status} "
                    f"(e.g. only a running experiment can be completed)"
                )
            conn.execute(
                f"UPDATE experiments SET {set_sql} WHERE exp_id = ?",  # noqa: S608
                (*params, exp_id),
            )

    # -- reads -------------------------------------------------------------

    def get(self, exp_id: str) -> Experiment:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM experiments WHERE exp_id = ?", (exp_id,)
            ).fetchone()
        if row is None:
            raise LedgerError(f"experiment '{exp_id}' does not exist")
        return _row_to_experiment(row)

    def list(self, comp_slug: str) -> list[Experiment]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM experiments WHERE comp_slug = ? ORDER BY exp_id", (comp_slug,)
            ).fetchall()
        return [_row_to_experiment(r) for r in rows]

    def best(self, comp_slug: str, direction: str) -> Experiment | None:
        if direction not in ("minimize", "maximize"):
            raise LedgerError(f"direction must be 'minimize' or 'maximize', got '{direction}'")
        order = "ASC" if direction == "minimize" else "DESC"
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM experiments WHERE comp_slug = ? AND status = 'completed' "
                f"AND cv_score_mean IS NOT NULL ORDER BY cv_score_mean {order} LIMIT 1",
                (comp_slug,),
            ).fetchone()
        return _row_to_experiment(row) if row is not None else None


def _row_to_experiment(row: sqlite3.Row) -> Experiment:
    return Experiment(
        exp_id=row["exp_id"],
        comp_slug=row["comp_slug"],
        parent_id=row["parent_id"],
        hypothesis=row["hypothesis"],
        approach=json.loads(row["approach"]),
        status=row["status"],
        cv_score_mean=row["cv_score_mean"],
        cv_score_std=row["cv_score_std"],
        cv_fold_scores=(
            json.loads(row["cv_fold_scores"]) if row["cv_fold_scores"] is not None else None
        ),
        wall_seconds=row["wall_seconds"],
        created_at=row["created_at"],
        finished_at=row["finished_at"],
        artifact_dir=row["artifact_dir"],
        error=row["error"],
        notes=row["notes"],
    )
