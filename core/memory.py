"""Agent-facing digest of ledger state: MEMORY.md per competition (SPEC §6.2).

Top-5 by score, last-5 attempted, graveyard of families that failed >= 2
times, current best submission source, remaining budget. Under 300 lines.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from core.budget import Budget
from core.competition import Competition
from core.ledger import Experiment, Ledger
from core.templates import render

GRAVEYARD_THRESHOLD = 2
TOP_N = 5
LAST_N = 5


def _family(exp: Experiment) -> str:
    return str(exp.approach.get("family") or exp.approach.get("model_family") or "unknown")


def write_memory(
    comp: Competition, ledger: Ledger, budget: Budget, out_path: Path | None = None
) -> Path:
    experiments = ledger.list(comp.slug)
    completed = [e for e in experiments if e.status == "completed" and e.cv_score_mean is not None]
    reverse = comp.metric_direction == "maximize"
    top = sorted(completed, key=lambda e: e.cv_score_mean or 0.0, reverse=reverse)[:TOP_N]
    last = experiments[-LAST_N:][::-1]

    failure_counts = Counter(
        _family(e) for e in experiments if e.status in ("failed", "timeout")
    )
    graveyard = sorted(
        (item for item in failure_counts.items() if item[1] >= GRAVEYARD_THRESHOLD),
        key=lambda item: -item[1],
    )

    best = ledger.best(comp.slug, comp.metric_direction)
    elapsed = sum(e.wall_seconds or 0.0 for e in experiments)

    def _view(e: Experiment) -> dict[str, Any]:
        return {
            "exp_id": e.exp_id,
            "status": e.status,
            "cv_score_mean": e.cv_score_mean,
            "cv_score_std": e.cv_score_std,
            "hypothesis": e.hypothesis,
            "family": _family(e),
            "artifact_dir": e.artifact_dir,
            "notes": e.notes,
        }

    text = render(
        "memory.md.j2",
        comp=comp,
        top=[_view(e) for e in top],
        last=[_view(e) for e in last],
        graveyard=graveyard,
        best=_view(best) if best is not None else None,
        budget_summary=budget.summary(used=len(experiments), elapsed=elapsed),
    )
    path = out_path if out_path is not None else comp.root / "MEMORY.md"
    path.write_text(text)
    return path
