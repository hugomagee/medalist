"""M5 acceptance: a scripted dumb agent runs end-to-end on a synthetic
competition and produces a graded RESULT.md — zero LLM involvement."""

from pathlib import Path

from core.competition import load_bundle
from core.ledger import Ledger
from tests.dumb_agent import run_dumb_agent
from tests.make_synthetic_comp import make_synthetic_comp


def test_dry_run_end_to_end(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    comp_dir = make_synthetic_comp(root, slug="synthetic-mae", seed=0)
    comp = load_bundle(comp_dir)
    assert comp.metric == "mae"

    outcome = run_dumb_agent(root, "synthetic-mae")

    ledger = Ledger(root / "ledger.db")
    experiments = ledger.list("synthetic-mae")
    assert [e.status for e in experiments] == ["completed"] * 4

    # later experiments must beat the naive baseline
    baseline = experiments[0]
    best = ledger.best("synthetic-mae", "minimize")
    assert best is not None and best.exp_id != baseline.exp_id
    assert best.cv_score_mean < 0.7 * baseline.cv_score_mean

    # graded result report exists and carries a percentile + medal line
    result_md = root / "reports" / "synthetic-mae" / "RESULT.md"
    assert result_md.exists()
    text = result_md.read_text()
    assert "percentile" in text.lower()
    assert (root / "reports" / "synthetic-mae" / "progression.png").exists()
    assert (root / "reports" / "synthetic-mae" / "submission.csv").exists()

    assert 0.0 < outcome.percentile <= 1.0
    assert outcome.rank >= 1


def test_synthetic_comp_is_valid_and_regenerable(tmp_path: Path) -> None:
    a = make_synthetic_comp(tmp_path / "a", slug="s", seed=7)
    b = make_synthetic_comp(tmp_path / "b", slug="s", seed=7)
    ta = (a / "data" / "train.csv").read_text()
    tb = (b / "data" / "train.csv").read_text()
    assert ta == tb  # deterministic generation
    comp = load_bundle(a)
    assert (a / "private" / "solution.csv").exists()
    assert (a / "private" / "leaderboard.csv").exists()
    assert len(comp.load_sample_submission()) == len(comp.load_test())
