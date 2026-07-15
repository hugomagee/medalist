"""CLI surface (SPEC §12). M1 ships init/validate; the rest are stubs
that fill in as their modules land in later milestones."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from core.competition import BundleError, load_bundle

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
def eda(slug: str) -> None:
    """Write reports/<slug>/eda.md. (Lands in M3.)"""
    typer.echo("eda: not implemented until M3", err=True)
    raise typer.Exit(code=2)


@app.command()
def run(slug: str, exp: str = typer.Option(..., help="experiment id, e.g. e0007")) -> None:
    """Execute one experiment. (Lands in M2.)"""
    typer.echo("run: not implemented until M2", err=True)
    raise typer.Exit(code=2)


@app.command()
def status(slug: str) -> None:
    """Ledger summary + budget remaining. (Lands in M2.)"""
    typer.echo("status: not implemented until M2", err=True)
    raise typer.Exit(code=2)


@app.command()
def memory(slug: str) -> None:
    """Regenerate MEMORY.md. (Lands in M3.)"""
    typer.echo("memory: not implemented until M3", err=True)
    raise typer.Exit(code=2)


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
