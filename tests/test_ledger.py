from pathlib import Path

import pytest

from core.ledger import Ledger, LedgerError


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "ledger.db")


HYP = "LightGBM on raw features beats the naive mean baseline."


class TestQueue:
    def test_queue_returns_monotonic_exp_ids(self, ledger: Ledger) -> None:
        e1 = ledger.queue("comp", HYP, {"model_family": "lgbm"})
        e2 = ledger.queue("comp", HYP, {"model_family": "xgb"})
        assert e1 == "e0001"
        assert e2 == "e0002"

    def test_queued_row_is_retrievable(self, ledger: Ledger) -> None:
        exp_id = ledger.queue("comp", HYP, {"model_family": "lgbm"}, parent_id=None)
        exp = ledger.get(exp_id)
        assert exp.comp_slug == "comp"
        assert exp.hypothesis == HYP
        assert exp.status == "queued"
        assert exp.approach == {"model_family": "lgbm"}
        assert exp.created_at is not None

    def test_short_hypothesis_rejected(self, ledger: Ledger) -> None:
        with pytest.raises(LedgerError, match="hypothesis"):
            ledger.queue("comp", "too short", {})

    def test_parent_must_exist(self, ledger: Ledger) -> None:
        with pytest.raises(LedgerError, match="parent"):
            ledger.queue("comp", HYP, {}, parent_id="e9999")


class TestLifecycle:
    def test_queued_to_running_to_completed(self, ledger: Ledger) -> None:
        exp_id = ledger.queue("comp", HYP, {})
        ledger.mark_running(exp_id)
        assert ledger.get(exp_id).status == "running"
        ledger.complete(
            exp_id,
            cv_score_mean=1.5,
            cv_score_std=0.1,
            cv_fold_scores=[1.4, 1.5, 1.6],
            wall_seconds=10.0,
            artifact_dir="experiments/comp/" + exp_id,
        )
        exp = ledger.get(exp_id)
        assert exp.status == "completed"
        assert exp.cv_score_mean == 1.5
        assert exp.cv_fold_scores == [1.4, 1.5, 1.6]
        assert exp.finished_at is not None

    def test_fail_records_error(self, ledger: Ledger) -> None:
        exp_id = ledger.queue("comp", HYP, {})
        ledger.mark_running(exp_id)
        ledger.fail(exp_id, error="Traceback: boom", wall_seconds=2.0)
        exp = ledger.get(exp_id)
        assert exp.status == "failed"
        assert exp.error is not None and "boom" in exp.error

    def test_timeout_status(self, ledger: Ledger) -> None:
        exp_id = ledger.queue("comp", HYP, {})
        ledger.mark_running(exp_id)
        ledger.mark_timeout(exp_id, wall_seconds=900.0)
        assert ledger.get(exp_id).status == "timeout"


class TestAppendOnly:
    def test_cannot_complete_a_queued_experiment(self, ledger: Ledger) -> None:
        exp_id = ledger.queue("comp", HYP, {})
        with pytest.raises(LedgerError, match="running"):
            ledger.complete(exp_id, 1.0, 0.0, [1.0], 1.0, "x")

    def test_cannot_rerun_a_completed_experiment(self, ledger: Ledger) -> None:
        exp_id = ledger.queue("comp", HYP, {})
        ledger.mark_running(exp_id)
        ledger.complete(exp_id, 1.0, 0.0, [1.0], 1.0, "x")
        with pytest.raises(LedgerError):
            ledger.mark_running(exp_id)

    def test_cannot_complete_twice(self, ledger: Ledger) -> None:
        exp_id = ledger.queue("comp", HYP, {})
        ledger.mark_running(exp_id)
        ledger.complete(exp_id, 1.0, 0.0, [1.0], 1.0, "x")
        with pytest.raises(LedgerError):
            ledger.complete(exp_id, 2.0, 0.0, [2.0], 1.0, "x")

    def test_no_delete_api(self, ledger: Ledger) -> None:
        assert not hasattr(ledger, "delete")

    def test_notes_can_be_set_after_completion(self, ledger: Ledger) -> None:
        # §9 step 4: the agent appends error-analysis findings after completion.
        exp_id = ledger.queue("comp", HYP, {})
        ledger.mark_running(exp_id)
        ledger.complete(exp_id, 1.0, 0.0, [1.0], 1.0, "x")
        ledger.set_notes(exp_id, "residuals skew high for large targets")
        assert ledger.get(exp_id).notes == "residuals skew high for large targets"

    def test_notes_rejected_on_queued_experiment(self, ledger: Ledger) -> None:
        exp_id = ledger.queue("comp", HYP, {})
        with pytest.raises(LedgerError):
            ledger.set_notes(exp_id, "premature notes on a queued row")


class TestQueries:
    def _completed(self, ledger: Ledger, slug: str, score: float) -> str:
        exp_id = ledger.queue(slug, HYP, {})
        ledger.mark_running(exp_id)
        ledger.complete(exp_id, score, 0.0, [score], 1.0, "x")
        return exp_id

    def test_list_filters_by_slug(self, ledger: Ledger) -> None:
        self._completed(ledger, "a", 1.0)
        self._completed(ledger, "b", 2.0)
        assert [e.comp_slug for e in ledger.list("a")] == ["a"]

    def test_best_minimize(self, ledger: Ledger) -> None:
        self._completed(ledger, "c", 3.0)
        best_id = self._completed(ledger, "c", 1.0)
        self._completed(ledger, "c", 2.0)
        best = ledger.best("c", direction="minimize")
        assert best is not None and best.exp_id == best_id

    def test_best_maximize(self, ledger: Ledger) -> None:
        self._completed(ledger, "c", 0.7)
        best_id = self._completed(ledger, "c", 0.9)
        best = ledger.best("c", direction="maximize")
        assert best is not None and best.exp_id == best_id

    def test_best_ignores_failed(self, ledger: Ledger) -> None:
        exp_id = ledger.queue("c", HYP, {})
        ledger.mark_running(exp_id)
        ledger.fail(exp_id, "boom", 1.0)
        assert ledger.best("c", direction="minimize") is None

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        db = tmp_path / "ledger.db"
        exp_id = Ledger(db).queue("comp", HYP, {})
        assert Ledger(db).get(exp_id).hypothesis == HYP
