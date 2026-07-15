from pathlib import Path

import pytest

from core.budget import Budget
from core.competition import load_bundle
from core.ledger import Ledger
from core.memory import write_memory


@pytest.fixture
def populated(comp_dir: Path, tmp_path: Path):
    comp = load_bundle(comp_dir)
    ledger = Ledger(tmp_path / "ledger.db")
    scores = [5.0, 3.0, 4.0, 2.0, 6.0, 7.0, 1.0]
    for i, s in enumerate(scores):
        exp_id = ledger.queue(
            comp.slug,
            f"experiment number {i} tests a distinct modelling idea",
            {"family": f"fam{i % 3}"},
        )
        ledger.mark_running(exp_id)
        ledger.complete(exp_id, s, 0.1, [s], 5.0, f"experiments/{comp.slug}/{exp_id}")
    # two failures in the same family -> graveyard
    for _ in range(2):
        exp_id = ledger.queue(comp.slug, "neural nets will surely fix everything here", {"family": "neural"})
        ledger.mark_running(exp_id)
        ledger.fail(exp_id, "OOM", 5.0)
    return comp, ledger


def test_memory_sections_and_content(populated, tmp_path: Path) -> None:
    comp, ledger = populated
    path = write_memory(comp, ledger, Budget())
    text = path.read_text()
    assert path.parent == comp.root
    assert "Top" in text
    assert "Last" in text
    assert "graveyard" in text.lower()
    assert "neural" in text  # family failed twice
    assert "budget" in text.lower()


def test_top5_is_best_first_for_minimize(populated) -> None:
    comp, ledger = populated
    text = write_memory(comp, ledger, Budget()).read_text()
    # best score is 1.0 (e0007); it must appear before the 2.0 experiment
    assert text.index("e0007") < text.index("e0004")


def test_memory_under_300_lines(populated) -> None:
    comp, ledger = populated
    text = write_memory(comp, ledger, Budget()).read_text()
    assert len(text.splitlines()) < 300


def test_single_failure_family_not_in_graveyard(populated) -> None:
    comp, ledger = populated
    exp_id = ledger.queue(comp.slug, "a one-off failing idea about interactions", {"family": "oneoff"})
    ledger.mark_running(exp_id)
    ledger.fail(exp_id, "boom", 1.0)
    text = write_memory(comp, ledger, Budget()).read_text()
    graveyard = text.lower().split("graveyard")[1]
    assert "oneoff" not in graveyard
