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


def test_init_scaffolds_competition_dir(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init", "my-comp", "--root", str(tmp_path)])
    assert result.exit_code == 0
    comp = tmp_path / "my-comp"
    assert (comp / "bundle.yaml").exists()
    assert (comp / "data").is_dir()
    assert (comp / "private").is_dir()
