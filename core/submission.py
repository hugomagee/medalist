"""Submission builder + format validator, and historical-comp grading (SPEC §11).

Grading scores a submission against private labels, locates it in the final
private leaderboard, and reports rank-equivalent, percentile, and medal band
using Kaggle's published thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from core.budget import Budget  # noqa: E402
from core.competition import Competition  # noqa: E402
from core.ledger import Ledger  # noqa: E402
from core.scoring import score  # noqa: E402
from core.templates import render  # noqa: E402


class SubmissionError(Exception):
    """Raised when a submission fails format validation or grading is impossible."""


def build_submission(comp: Competition, test_pred_path: Path, out_path: Path) -> Path:
    """Build a submission CSV from a test_pred.parquet artifact, ordered and
    formatted exactly like sample_submission."""
    preds = pd.read_parquet(test_pred_path)
    if comp.id_column not in preds.columns or "prediction" not in preds.columns:
        raise SubmissionError(
            f"{test_pred_path} must have columns [{comp.id_column}, prediction]"
        )
    sample = comp.load_sample_submission()
    merged = sample[[comp.id_column]].merge(preds, on=comp.id_column, how="left")
    if merged["prediction"].isna().any():
        missing = int(merged["prediction"].isna().sum())
        raise SubmissionError(
            f"predictions missing for {missing} test id(s); cannot build submission"
        )
    out = pd.DataFrame(
        {comp.id_column: merged[comp.id_column], comp.target_column: merged["prediction"]}
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    validate_submission(comp, out_path)
    return out_path


def validate_submission(comp: Competition, submission_path: Path) -> None:
    """Format check against sample_submission: columns, ids, no NaN."""
    sub = pd.read_csv(submission_path)
    sample = comp.load_sample_submission()
    if list(sub.columns) != list(sample.columns):
        raise SubmissionError(
            f"submission columns {list(sub.columns)} != sample {list(sample.columns)}"
        )
    if len(sub) != len(sample):
        raise SubmissionError(f"submission has {len(sub)} rows, sample has {len(sample)}")
    if not sub[comp.id_column].reset_index(drop=True).equals(
        sample[comp.id_column].reset_index(drop=True)
    ):
        raise SubmissionError("submission ids do not match sample_submission ids/order")
    if sub[comp.target_column].isna().any():
        raise SubmissionError("submission contains NaN predictions")


def medal_thresholds(n_teams: int) -> dict[str, int]:
    """Kaggle medal rank cutoffs (rank <= cutoff earns the medal)."""
    if n_teams < 100:
        return {
            "gold": int(n_teams * 0.10),
            "silver": int(n_teams * 0.20),
            "bronze": int(n_teams * 0.40),
        }
    if n_teams < 250:
        return {"gold": 10, "silver": int(n_teams * 0.20), "bronze": int(n_teams * 0.40)}
    if n_teams < 1000:
        return {"gold": 10 + int(n_teams * 0.002), "silver": 50, "bronze": 100}
    return {
        "gold": 10 + int(n_teams * 0.002),
        "silver": int(n_teams * 0.05),
        "bronze": int(n_teams * 0.10),
    }


@dataclass(frozen=True)
class Grade:
    score: float
    rank: int
    n_teams: int
    percentile: float
    medal: str | None


def grade_submission(comp: Competition, submission_path: Path) -> Grade:
    """Score against private labels and locate in the final leaderboard."""
    if comp.private_labels is None or not comp.private_labels.is_file():
        raise SubmissionError(
            "grading needs private labels; set private_labels in bundle.yaml "
            "and assemble private/solution.csv (see competitions/README.md)"
        )
    if comp.leaderboard is None or not comp.leaderboard.is_file():
        raise SubmissionError(
            "grading needs the final leaderboard; set leaderboard in bundle.yaml"
        )
    validate_submission(comp, submission_path)
    sub = pd.read_csv(submission_path)
    solution = pd.read_csv(comp.private_labels)
    merged = sub.merge(solution, on=comp.id_column, suffixes=("_pred", "_true"))
    if len(merged) != len(sub):
        raise SubmissionError("submission ids do not fully match private solution ids")
    raw = score(
        comp.metric,
        merged[f"{comp.target_column}_true"],
        merged[f"{comp.target_column}_pred"],
    )

    lb = pd.read_csv(comp.leaderboard)
    if "score" not in lb.columns:
        raise SubmissionError("leaderboard.csv must have a 'score' column")
    lb_scores = lb["score"].to_numpy(dtype=float)
    if comp.metric_direction == "minimize":
        better = int((lb_scores < raw).sum())
    else:
        better = int((lb_scores > raw).sum())
    rank = better + 1
    n_teams = len(lb_scores)
    thresholds = medal_thresholds(n_teams)
    medal = next((band for band in ("gold", "silver", "bronze") if rank <= thresholds[band]), None)
    return Grade(
        score=raw,
        rank=rank,
        n_teams=n_teams,
        percentile=rank / n_teams,
        medal=medal,
    )


def _progression_chart(ledger: Ledger, comp: Competition, out_path: Path) -> None:
    completed = [
        e
        for e in ledger.list(comp.slug)
        if e.status == "completed" and e.cv_score_mean is not None
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = [int(e.exp_id[1:]) for e in completed]
    ys = [float(e.cv_score_mean) for e in completed if e.cv_score_mean is not None]
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("experiment number")
    ax.set_ylabel(f"CV {comp.metric}")
    ax.set_title(f"{comp.slug}: score vs experiment")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_result_report(
    comp: Competition, grade: Grade, ledger: Ledger, budget: Budget, out_dir: Path
) -> Path:
    """RESULT.md: score, percentile, medal, budget consumed, progression chart,
    top-3 experiment hypotheses."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _progression_chart(ledger, comp, out_dir / "progression.png")

    experiments = ledger.list(comp.slug)
    completed = [e for e in experiments if e.status == "completed" and e.cv_score_mean is not None]
    reverse = comp.metric_direction == "maximize"
    top3 = sorted(completed, key=lambda e: e.cv_score_mean or 0.0, reverse=reverse)[:3]
    elapsed = sum(e.wall_seconds or 0.0 for e in experiments)

    report_path = out_dir / "RESULT.md"
    report_path.write_text(
        render(
            "result.md.j2",
            comp=comp,
            grade=grade,
            n_experiments=len(experiments),
            n_completed=len(completed),
            elapsed=elapsed,
            budget=budget,
            top3=top3,
        )
    )
    return report_path
