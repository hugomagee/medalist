from pathlib import Path

import numpy as np
import pandas as pd

from core.competition import load_bundle
from core.eda import write_eda_report
from tests.conftest import write_bundle


def test_eda_report_sections(comp_dir: Path, tmp_path: Path) -> None:
    comp = load_bundle(comp_dir)
    report_path = write_eda_report(comp, tmp_path / "reports" / comp.slug)
    assert report_path.name == "eda.md"
    text = report_path.read_text()
    for section in ("dtype", "Missing", "Cardinality", "Target", "Correlation", "Adversarial"):
        assert section.lower() in text.lower(), section
    assert "feat_a" in text


def test_adversarial_auc_high_on_drifted_test(tmp_path: Path) -> None:
    comp_dir = tmp_path / "drifted"
    write_bundle(comp_dir)
    r = np.random.default_rng(0)
    n = 200
    train = pd.DataFrame(
        {
            "id": range(n),
            "feat_a": r.normal(0, 1, n),
            "feat_b": r.integers(0, 3, n),
            "yield": r.normal(100, 10, n),
        }
    )
    test = pd.DataFrame(
        {
            "id": range(n, n + n),
            "feat_a": r.normal(1000, 1, n),  # massive drift
            "feat_b": r.integers(0, 3, n),
        }
    )
    train.to_csv(comp_dir / "data" / "train.csv", index=False)
    test.to_csv(comp_dir / "data" / "test.csv", index=False)
    pd.DataFrame({"id": test["id"], "yield": 100.0}).to_csv(
        comp_dir / "data" / "sample_submission.csv", index=False
    )
    comp = load_bundle(comp_dir)
    report_path = write_eda_report(comp, tmp_path / "reports" / comp.slug)
    text = report_path.read_text()
    auc_line = next(line for line in text.splitlines() if "adversarial" in line.lower())
    auc = float(auc_line.rstrip().split()[-1].strip("*`"))
    assert auc > 0.8
