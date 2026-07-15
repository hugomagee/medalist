"""CLI surface (SPEC §12). M1 ships init/validate; the rest are stubs
that fill in as their modules land in later milestones."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from core.budget import Budget, BudgetExhausted
from core.competition import BundleError, load_bundle
from core.eda import write_eda_report
from core.ledger import Ledger, LedgerError
from core.memory import write_memory
from core.runner import execute_experiment

app = typer.Typer(no_args_is_help=True, help="medalist: autonomous ML competition harness")

BUNDLE_TEMPLATE: dict[str, object] = {
    "slug": "",
    "title": "",
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
    "leaderboard": None,
    "private_labels": None,
}


def _comp_dir(slug_or_path: str, root: Path | None = None) -> Path:
    """Resolve a slug (under <root>/competitions or <root>) or a direct path."""
    direct = Path(slug_or_path)
    if direct.is_dir():
        return direct
    base = root if root is not None else Path.cwd()
    for candidate in (base / "competitions" / slug_or_path, base / slug_or_path):
        if candidate.is_dir():
            return candidate
    return direct


@app.command()
def init(
    slug: str,
    root: Path = typer.Option(Path("competitions"), help="parent directory for the new bundle"),
) -> None:
    """Scaffold a competition dir + bundle template."""
    comp = root / slug
    (comp / "data").mkdir(parents=True, exist_ok=True)
    (comp / "private").mkdir(exist_ok=True)
    bundle = dict(BUNDLE_TEMPLATE)
    bundle["slug"] = slug
    bundle["title"] = slug
    (comp / "bundle.yaml").write_text(yaml.safe_dump(bundle, sort_keys=False))
    typer.echo(f"Scaffolded {comp}. Edit bundle.yaml and drop data files into data/.")


@app.command()
def validate(slug: str) -> None:
    """Bundle + data sanity checks."""
    try:
        comp = load_bundle(_comp_dir(slug))
    except BundleError as exc:
        typer.echo(f"INVALID: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"OK: bundle '{comp.slug}' valid (metric={comp.metric}, cv={comp.cv_policy})")


@app.command()
def eda(
    slug: str,
    root: Path = typer.Option(Path("."), help="repo root (reports/ written here)"),
) -> None:
    """Write reports/<slug>/eda.md."""
    comp = load_bundle(_comp_dir(slug, root))
    path = write_eda_report(comp, root / "reports" / comp.slug)
    typer.echo(f"wrote {path}")


def _load_budget(root: Path) -> "Budget":
    budget_path = root / "budget.yaml"
    if budget_path.is_file():
        raw = yaml.safe_load(budget_path.read_text()) or {}
        return Budget(**raw)
    return Budget()


def _elapsed(ledger: "Ledger", slug: str) -> float:
    return sum(e.wall_seconds or 0.0 for e in ledger.list(slug))


@app.command()
def run(
    slug: str,
    exp: str = typer.Option(..., help="experiment id, e.g. e0007"),
    root: Path = typer.Option(Path("."), help="repo root (ledger.db, experiments/)"),
    final: bool = typer.Option(False, help="use a reserved budget slot (final ensemble)"),
) -> None:
    """Execute one experiment."""
    try:
        comp = load_bundle(_comp_dir(slug, root))
    except BundleError as exc:
        typer.echo(f"INVALID: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    ledger = Ledger(root / "ledger.db")
    budget = _load_budget(root)
    exp_dir = root / "experiments" / comp.slug / exp
    try:
        record = execute_experiment(comp, exp_dir, ledger, budget, final=final)
    except (BudgetExhausted, LedgerError) as exc:
        typer.echo(f"REFUSED: {exc}", err=True)
        typer.echo(budget.summary(len(ledger.list(comp.slug)), _elapsed(ledger, comp.slug)))
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"{record.exp_id}: {record.status}"
        + (f" cv={record.cv_score_mean:.6f} ±{record.cv_score_std:.6f}"
           if record.cv_score_mean is not None else "")
        + (f" ({record.error})" if record.error else "")
    )
    typer.echo(budget.summary(len(ledger.list(comp.slug)), _elapsed(ledger, comp.slug)))
    if record.status != "completed":
        raise typer.Exit(code=1)


@app.command()
def status(
    slug: str,
    root: Path = typer.Option(Path("."), help="repo root (ledger.db, experiments/)"),
) -> None:
    """Ledger summary + budget remaining."""
    comp = load_bundle(_comp_dir(slug, root))
    ledger = Ledger(root / "ledger.db")
    budget = _load_budget(root)
    experiments = ledger.list(comp.slug)
    for e in experiments:
        line = f"{e.exp_id}  {e.status:9s}"
        if e.cv_score_mean is not None:
            line += f"  cv={e.cv_score_mean:.6f}"
        line += f"  {e.hypothesis[:60]}"
        typer.echo(line)
    best = ledger.best(comp.slug, comp.metric_direction)
    if best is not None:
        typer.echo(f"best: {best.exp_id} cv={best.cv_score_mean:.6f} ({comp.metric})")
    typer.echo(budget.summary(len(experiments), _elapsed(ledger, comp.slug)))


@app.command()
def memory(
    slug: str,
    root: Path = typer.Option(Path("."), help="repo root (ledger.db lives here)"),
) -> None:
    """Regenerate MEMORY.md."""
    comp = load_bundle(_comp_dir(slug, root))
    ledger = Ledger(root / "ledger.db")
    path = write_memory(comp, ledger, _load_budget(root))
    typer.echo(f"wrote {path}")


@app.command()
def submit(slug: str) -> None:
    """Build + validate submission from best/ensemble. (Lands in M4.)"""
    typer.echo("submit: not implemented until M4", err=True)
    raise typer.Exit(code=2)


@app.command()
def grade(slug: str) -> None:
    """Historical comps: score vs private LB. (Lands in M4.)"""
    typer.echo("grade: not implemented until M4", err=True)
    raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
