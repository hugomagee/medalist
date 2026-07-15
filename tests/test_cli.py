from pathlib import Path

from typer.testing import CliRunner

from core.cli import app

runner = CliRunner()


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ["init", "validate", "eda", "run", "status", "memory", "submit", "grade"]:
        assert cmd in result.output


def test_validate_ok_on_valid_bundle(comp_dir: Path) -> None:
    result = runner.invoke(app, ["validate", str(comp_dir)])
    assert result.exit_code == 0
    assert "OK" in result.output


def test_validate_fails_on_broken_bundle(comp_dir: Path) -> None:
    (comp_dir / "data" / "train.csv").unlink()
    result = runner.invoke(app, ["validate", str(comp_dir)])
    assert result.exit_code != 0


def _setup_repo(tmp_path: Path, comp_dir: Path) -> Path:
    """Arrange a repo root with competitions/<slug> and an experiment ready to run."""
    root = tmp_path / "repo"
    (root / "competitions").mkdir(parents=True)
    (root / "competitions" / comp_dir.name).symlink_to(comp_dir)
    exp_dir = root / "experiments" / comp_dir.name / "e0001"
    exp_dir.mkdir(parents=True)
    (exp_dir / "run.py").write_text(
        "import numpy as np\n"
        "def run(train_df, test_df, cv_splitter, metric):\n"
        "    y = train_df['yield'].to_numpy(dtype=float)\n"
        "    oof = np.full(len(train_df), y.mean())\n"
        "    return {'oof_predictions': oof,\n"
        "            'test_predictions': np.full(len(test_df), y.mean())}\n"
    )
    (exp_dir / "meta.yaml").write_text(
        "hypothesis: a global mean baseline anchors the ledger\n"
        "approach: {model_family: baseline}\n"
    )
    return root


def test_run_executes_experiment_and_prints_budget(comp_dir: Path, tmp_path: Path) -> None:
    root = _setup_repo(tmp_path, comp_dir)
    result = runner.invoke(
        app, ["run", comp_dir.name, "--exp", "e0001", "--root", str(root)]
    )
    assert result.exit_code == 0, result.output
    assert "completed" in result.output
    assert "budget" in result.output.lower()
    assert (root / "experiments" / comp_dir.name / "e0001" / "result.json").exists()


def test_status_summarises_ledger_and_budget(comp_dir: Path, tmp_path: Path) -> None:
    root = _setup_repo(tmp_path, comp_dir)
    runner.invoke(app, ["run", comp_dir.name, "--exp", "e0001", "--root", str(root)])
    result = runner.invoke(app, ["status", comp_dir.name, "--root", str(root)])
    assert result.exit_code == 0, result.output
    assert "e0001" in result.output
    assert "completed" in result.output
    assert "budget" in result.output.lower()


def test_init_scaffolds_competition_dir(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", "my-comp", "--root", str(tmp_path)])
    assert result.exit_code == 0
    comp = tmp_path / "my-comp"
    assert (comp / "bundle.yaml").exists()
    assert (comp / "data").is_dir()
    assert (comp / "private").is_dir()
