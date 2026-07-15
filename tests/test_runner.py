import json
import time
from pathlib import Path

import pytest
import yaml

from core.budget import Budget, BudgetExhausted
from core.competition import load_bundle
from core.ledger import Ledger
from core.runner import execute_experiment

MEAN_RUN = """
import numpy as np

def run(train_df, test_df, cv_splitter, metric):
    y = train_df["yield"].to_numpy(dtype=float)
    oof = np.empty(len(train_df))
    fold_scores = []
    for train_idx, valid_idx in cv_splitter.split(train_df):
        mean = y[train_idx].mean()
        oof[valid_idx] = mean
        fold_scores.append(metric(y[valid_idx], oof[valid_idx]))
    test_pred = np.full(len(test_df), y.mean())
    return {"oof_predictions": oof, "test_predictions": test_pred, "fold_scores": fold_scores}
"""


def make_exp(
    exp_root: Path,
    exp_id: str,
    code: str,
    hypothesis: str = "A mean baseline anchors the ledger with a floor score.",
) -> Path:
    exp_dir = exp_root / exp_id
    exp_dir.mkdir(parents=True)
    (exp_dir / "run.py").write_text(code)
    (exp_dir / "meta.yaml").write_text(
        yaml.safe_dump({"hypothesis": hypothesis, "approach": {"model_family": "baseline"}})
    )
    return exp_dir


@pytest.fixture
def env(comp_dir: Path, tmp_path: Path):
    comp = load_bundle(comp_dir)
    ledger = Ledger(tmp_path / "ledger.db")
    budget = Budget(per_experiment_seconds=120)
    exp_root = tmp_path / "experiments" / comp.slug
    return comp, ledger, budget, exp_root


class TestSuccessfulRun:
    def test_completes_and_writes_artifacts(self, env) -> None:
        comp, ledger, budget, exp_root = env
        exp_dir = make_exp(exp_root, "e0001", MEAN_RUN)
        exp = execute_experiment(comp, exp_dir, ledger, budget)
        assert exp.status == "completed"
        assert exp.cv_score_mean is not None and exp.cv_score_mean > 0
        assert exp.cv_fold_scores is not None and len(exp.cv_fold_scores) == 5
        assert exp.wall_seconds is not None and exp.wall_seconds > 0
        for artifact in ("oof.parquet", "test_pred.parquet", "result.json", "run.py"):
            assert (exp_dir / artifact).exists(), artifact
        result = json.loads((exp_dir / "result.json").read_text())
        assert result["cv_score_mean"] == pytest.approx(exp.cv_score_mean)

    def test_harness_recomputes_scores_ignoring_experiment_claims(self, env) -> None:
        comp, ledger, budget, exp_root = env
        cheat = MEAN_RUN.replace(
            'return {"oof_predictions": oof, "test_predictions": test_pred, "fold_scores": fold_scores}',
            'return {"oof_predictions": oof, "test_predictions": test_pred, "fold_scores": [0.0]*5}',
        )
        exp_dir = make_exp(exp_root, "e0001", cheat)
        exp = execute_experiment(comp, exp_dir, ledger, budget)
        assert exp.status == "completed"
        assert exp.cv_score_mean is not None and exp.cv_score_mean > 0.5  # not the claimed 0.0


class TestDeterminism:
    def test_seeded_rerun_reproduces_cv_mean(self, env) -> None:
        comp, ledger, budget, exp_root = env
        noisy = MEAN_RUN.replace(
            "oof = np.empty(len(train_df))",
            "oof = np.empty(len(train_df)); noise = np.random.normal(size=len(train_df))",
        ).replace(
            "oof[valid_idx] = mean",
            "oof[valid_idx] = mean + noise[valid_idx]",
        )
        e1 = execute_experiment(comp, make_exp(exp_root, "e0001", noisy), ledger, budget)
        e2 = execute_experiment(comp, make_exp(exp_root, "e0002", noisy), ledger, budget)
        assert e1.cv_score_mean == pytest.approx(e2.cv_score_mean, abs=1e-9)


class TestTimeout:
    def test_hung_experiment_is_killed(self, env) -> None:
        comp, ledger, _, exp_root = env
        budget = Budget(per_experiment_seconds=2)
        hang = "import time\n\ndef run(train_df, test_df, cv_splitter, metric):\n    time.sleep(60)\n"
        exp_dir = make_exp(exp_root, "e0001", hang)
        start = time.monotonic()
        exp = execute_experiment(comp, exp_dir, ledger, budget)
        assert time.monotonic() - start < 30
        assert exp.status == "timeout"


class TestAntiCheat:
    def test_static_check_rejects_private_string(self, env) -> None:
        comp, ledger, budget, exp_root = env
        snoop = MEAN_RUN + "\nPATH = 'competitions/x/private/solution.csv'\n"
        exp = execute_experiment(comp, make_exp(exp_root, "e0001", snoop), ledger, budget)
        assert exp.status == "failed"
        assert exp.error is not None and "private" in exp.error.lower()

    def test_runtime_read_of_private_dir_fails(self, env, comp_dir: Path) -> None:
        comp, ledger, budget, exp_root = env
        private = comp_dir / "private"
        private.mkdir()
        (private / "solution.csv").write_text("id,yield\n1,1.0\n")
        # path assembled to dodge the static string check; returns valid
        # predictions, so the run can only fail if the read itself is blocked
        sneaky = (
            "import numpy as np\n"
            "def run(train_df, test_df, cv_splitter, metric):\n"
            f"    base = {str(comp_dir)!r}\n"
            "    data = open(base + '/pri' + 'vate/solution.csv').read()\n"
            '    return {"oof_predictions": np.zeros(len(train_df)),\n'
            '            "test_predictions": np.zeros(len(test_df))}\n'
        )
        exp = execute_experiment(comp, make_exp(exp_root, "e0001", sneaky), ledger, budget)
        assert exp.status == "failed"
        # and the private dir is readable again afterwards (restored by the runner)
        assert (private / "solution.csv").read_text().startswith("id")

    def test_misaligned_oof_rejected(self, env) -> None:
        comp, ledger, budget, exp_root = env
        short = MEAN_RUN.replace(
            '"oof_predictions": oof', '"oof_predictions": oof[:-3]'
        )
        exp = execute_experiment(comp, make_exp(exp_root, "e0001", short), ledger, budget)
        assert exp.status == "failed"
        assert exp.error is not None and "align" in exp.error.lower()


class TestQueueDiscipline:
    def test_budget_exhaustion_blocks_queue(self, env) -> None:
        comp, ledger, _, exp_root = env
        budget = Budget(max_experiments=2, min_reserve_experiments=2)
        exp_dir = make_exp(exp_root, "e0001", MEAN_RUN)
        with pytest.raises(BudgetExhausted):
            execute_experiment(comp, exp_dir, ledger, budget)

    def test_short_hypothesis_never_runs(self, env) -> None:
        comp, ledger, budget, exp_root = env
        exp_dir = make_exp(exp_root, "e0001", MEAN_RUN, hypothesis="short")
        from core.ledger import LedgerError

        with pytest.raises(LedgerError):
            execute_experiment(comp, exp_dir, ledger, budget)
        assert (exp_dir / "run.py").exists()
        assert not (exp_dir / "result.json").exists()
