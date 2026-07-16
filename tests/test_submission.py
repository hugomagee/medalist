from pathlib import Path

import pandas as pd
import pytest

from core.budget import Budget
from core.competition import load_bundle
from core.ledger import Ledger
from core.submission import (
    SubmissionError,
    build_submission,
    grade_submission,
    medal_thresholds,
    validate_submission,
    write_result_report,
)


@pytest.fixture
def comp(comp_dir: Path):
    return load_bundle(comp_dir)


def _test_pred_parquet(path: Path, ids: list[int], preds: list[float]) -> Path:
    pd.DataFrame({"id": ids, "prediction": preds}).to_parquet(path)
    return path


class TestBuildAndValidate:
    def test_build_orders_rows_like_sample_submission(self, comp, tmp_path: Path) -> None:
        ids = list(range(20, 30))
        pred_path = _test_pred_parquet(
            tmp_path / "p.parquet", ids[::-1], [float(i) for i in ids[::-1]]
        )
        out = build_submission(comp, pred_path, tmp_path / "submission.csv")
        sub = pd.read_csv(out)
        assert list(sub.columns) == ["id", "yield"]
        assert sub["id"].tolist() == ids  # sample submission order
        assert sub["yield"].tolist() == [float(i) for i in ids]

    def test_validate_rejects_wrong_columns(self, comp, tmp_path: Path) -> None:
        bad = tmp_path / "bad.csv"
        bad.write_text("id,wrong\n20,1.0\n")
        with pytest.raises(SubmissionError, match="column"):
            validate_submission(comp, bad)

    def test_validate_rejects_wrong_ids(self, comp, tmp_path: Path) -> None:
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"id": range(99, 109), "yield": 1.0}).to_csv(bad, index=False)
        with pytest.raises(SubmissionError, match="id"):
            validate_submission(comp, bad)

    def test_validate_rejects_nan(self, comp, tmp_path: Path) -> None:
        bad = tmp_path / "bad.csv"
        frame = pd.DataFrame({"id": range(20, 30), "yield": [1.0] * 9 + [float("nan")]})
        frame.to_csv(bad, index=False)
        with pytest.raises(SubmissionError, match="NaN"):
            validate_submission(comp, bad)

    def test_build_rejects_missing_test_ids(self, comp, tmp_path: Path) -> None:
        pred_path = _test_pred_parquet(tmp_path / "p.parquet", [20, 21], [1.0, 2.0])
        with pytest.raises(SubmissionError):
            build_submission(comp, pred_path, tmp_path / "submission.csv")


class TestMedalThresholds:
    # hand-computed fixtures per Kaggle's published tiers
    @pytest.mark.parametrize(
        "n_teams,gold,silver,bronze",
        [
            (50, 5, 10, 20),      # <100: 10% / 20% / 40%
            (150, 10, 30, 60),    # 100-249: top 10 / 20% / 40%
            (500, 11, 50, 100),   # 250-999: 10+0.2% / top 50 / top 100
            (2000, 14, 100, 200), # 1000+: 10+0.2% / 5% / 10%
        ],
    )
    def test_thresholds(self, n_teams: int, gold: int, silver: int, bronze: int) -> None:
        t = medal_thresholds(n_teams)
        assert t == {"gold": gold, "silver": silver, "bronze": bronze}


class TestGrading:
    def _graded_comp(self, comp_dir: Path, lb_scores: list[float]):
        private = comp_dir / "private"
        private.mkdir(exist_ok=True)
        solution = pd.DataFrame({"id": range(20, 30), "yield": [100.0 + i for i in range(10)]})
        solution.to_csv(private / "solution.csv", index=False)
        pd.DataFrame({"rank": range(1, len(lb_scores) + 1), "score": lb_scores}).to_csv(
            private / "leaderboard.csv", index=False
        )
        from tests.conftest import write_bundle

        write_bundle(
            comp_dir,
            leaderboard="private/leaderboard.csv",
            private_labels="private/solution.csv",
        )
        return load_bundle(comp_dir)

    def test_grade_rank_percentile_medal(self, comp_dir: Path, tmp_path: Path) -> None:
        comp = self._graded_comp(comp_dir, lb_scores=[float(i) for i in range(1, 11)])
        # perfect submission -> MAE 0.0, better than every LB score
        sub = tmp_path / "submission.csv"
        pd.DataFrame({"id": range(20, 30), "yield": [100.0 + i for i in range(10)]}).to_csv(
            sub, index=False
        )
        grade = grade_submission(comp, sub)
        assert grade.score == pytest.approx(0.0)
        assert grade.rank == 1
        assert grade.n_teams == 10
        assert grade.percentile == pytest.approx(0.1)
        assert grade.medal == "gold"

    def test_grade_mid_pack(self, comp_dir: Path, tmp_path: Path) -> None:
        comp = self._graded_comp(comp_dir, lb_scores=[float(i) for i in range(1, 11)])
        sub = tmp_path / "submission.csv"
        # constant prediction offset by 5.0 -> MAE ~ 5.0... construct exact:
        pd.DataFrame({"id": range(20, 30), "yield": [100.0 + i + 5.5 for i in range(10)]}).to_csv(
            sub, index=False
        )
        grade = grade_submission(comp, sub)  # score 5.5, beats 6,7,8,9,10 -> rank 6
        assert grade.score == pytest.approx(5.5)
        assert grade.rank == 6
        assert grade.medal is None  # bronze cutoff at n=10 is top 4

    def test_grade_requires_private_labels(self, comp, tmp_path: Path) -> None:
        sub = tmp_path / "submission.csv"
        pd.DataFrame({"id": range(20, 30), "yield": 1.0}).to_csv(sub, index=False)
        with pytest.raises(SubmissionError, match="private"):
            grade_submission(comp, sub)


class TestGradingMaximize:
    """Direction coverage for maximize metrics (AUC): a higher score must rank
    better, and the leaderboard comparison must not be the minimize one."""

    SOLUTION = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # ids 20..29

    def _auc_comp(self, comp_dir: Path, lb_scores: list[float]):
        private = comp_dir / "private"
        private.mkdir(exist_ok=True)
        pd.DataFrame({"id": range(20, 30), "churn": self.SOLUTION}).to_csv(
            private / "solution.csv", index=False
        )
        pd.DataFrame({"rank": range(1, len(lb_scores) + 1), "score": lb_scores}).to_csv(
            private / "leaderboard.csv", index=False
        )
        from tests.conftest import write_bundle

        write_bundle(
            comp_dir,
            metric="auc",
            metric_direction="maximize",
            target_column="churn",
            cv_policy="stratified",
            leaderboard="private/leaderboard.csv",
            private_labels="private/solution.csv",
        )
        # write_bundle writes regression CSVs; replace with a binary target
        data = comp_dir / "data"
        pd.DataFrame(
            {"id": range(20), "feat_a": [float(i) for i in range(20)], "churn": [i % 2 for i in range(20)]}
        ).to_csv(data / "train.csv", index=False)
        pd.DataFrame({"id": range(20, 30), "feat_a": [float(i) for i in range(10)]}).to_csv(
            data / "test.csv", index=False
        )
        pd.DataFrame({"id": range(20, 30), "churn": [0.5] * 10}).to_csv(
            data / "sample_submission.csv", index=False
        )
        return load_bundle(comp_dir)

    def _submission(self, tmp_path: Path, preds: list[float]) -> Path:
        sub = tmp_path / "submission.csv"
        pd.DataFrame({"id": range(20, 30), "churn": preds}).to_csv(sub, index=False)
        return sub

    def test_perfect_auc_ranks_first(self, comp_dir: Path, tmp_path: Path) -> None:
        # 10 teams, all below 1.0 -> perfect submission must be rank 1 / gold
        comp = self._auc_comp(comp_dir, lb_scores=[0.91 - 0.01 * i for i in range(10)])
        sub = self._submission(tmp_path, [float(v) for v in self.SOLUTION])
        grade = grade_submission(comp, sub)
        assert grade.score == pytest.approx(1.0)
        assert grade.rank == 1
        assert grade.n_teams == 10
        assert grade.percentile == pytest.approx(0.1)
        assert grade.medal == "gold"

    def test_mid_pack_rank_counts_higher_scores_as_better(
        self, comp_dir: Path, tmp_path: Path
    ) -> None:
        # Constant predictions -> AUC exactly 0.5 (average ranks). The LB is
        # asymmetric around 0.5 on purpose: 8 teams above, 1 tie, 1 below.
        # Correct maximize ranking -> rank 9; a minimize comparison would give 2.
        lb = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.4]
        comp = self._auc_comp(comp_dir, lb_scores=lb)
        grade = grade_submission(comp, self._submission(tmp_path, [0.5] * 10))
        assert grade.score == pytest.approx(0.5)
        assert grade.rank == 9  # ties share the better rank, Kaggle-style
        assert grade.percentile == pytest.approx(0.9)
        assert grade.medal is None

    def test_result_report_maximize_lists_highest_cv_first(
        self, comp_dir: Path, tmp_path: Path
    ) -> None:
        comp = self._auc_comp(comp_dir, lb_scores=[0.91 - 0.01 * i for i in range(10)])
        ledger = Ledger(tmp_path / "ledger.db")
        for tag, cv in (("alpha", 0.6), ("bravo", 0.8), ("charlie", 0.7)):
            exp_id = ledger.queue(
                comp.slug, f"{tag} hypothesis about ranking quality here", {"family": "lgbm"}
            )
            ledger.mark_running(exp_id)
            ledger.complete(exp_id, cv, 0.01, [cv], 5.0, "x")
        grade = grade_submission(
            comp, self._submission(tmp_path, [float(v) for v in self.SOLUTION])
        )
        out_dir = tmp_path / "reports" / comp.slug
        text = write_result_report(comp, grade, ledger, Budget(), out_dir).read_text()
        assert text.index("bravo") < text.index("charlie") < text.index("alpha")


class TestResultReport:
    def test_result_md_written_with_chart(self, comp_dir: Path, tmp_path: Path) -> None:
        comp = TestGrading()._graded_comp(comp_dir, [float(i) for i in range(1, 11)])
        ledger = Ledger(tmp_path / "ledger.db")
        for score_value in (5.0, 3.0, 2.0):
            exp_id = ledger.queue(
                comp.slug, "progressively better models drive the score down", {"family": "lgbm"}
            )
            ledger.mark_running(exp_id)
            ledger.complete(exp_id, score_value, 0.1, [score_value], 5.0, "x")
        sub = tmp_path / "submission.csv"
        pd.DataFrame({"id": range(20, 30), "yield": [100.0 + i for i in range(10)]}).to_csv(
            sub, index=False
        )
        grade = grade_submission(comp, sub)
        out_dir = tmp_path / "reports" / comp.slug
        report = write_result_report(comp, grade, ledger, Budget(), out_dir)
        text = report.read_text()
        assert "gold" in text.lower()
        assert "0.1" in text or "10.0%" in text  # percentile
        assert (out_dir / "progression.png").exists()
        assert text.count("progressively better") >= 1  # top-3 hypotheses shown
