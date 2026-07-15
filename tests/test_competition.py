from pathlib import Path

import pytest

from core.competition import BundleError, Competition, load_bundle
from tests.conftest import write_bundle


class TestLoadValidBundle:
    def test_returns_competition_with_fields(self, comp_dir: Path) -> None:
        comp = load_bundle(comp_dir)
        assert isinstance(comp, Competition)
        assert comp.slug == "test-comp"
        assert comp.metric == "mae"
        assert comp.metric_direction == "minimize"
        assert comp.target_column == "yield"
        assert comp.id_column == "id"
        assert comp.cv_policy == "auto"
        assert comp.cv_params == {"n_splits": 5, "seed": 42}

    def test_file_paths_resolved_relative_to_comp_dir(self, comp_dir: Path) -> None:
        comp = load_bundle(comp_dir)
        assert comp.files["train"] == comp_dir / "data" / "train.csv"
        assert comp.files["train"].exists()


class TestBundleValidation:
    def test_missing_bundle_yaml(self, tmp_path: Path) -> None:
        with pytest.raises(BundleError, match="bundle.yaml"):
            load_bundle(tmp_path / "nope")

    def test_missing_train_file(self, comp_dir: Path) -> None:
        (comp_dir / "data" / "train.csv").unlink()
        with pytest.raises(BundleError, match="train"):
            load_bundle(comp_dir)

    def test_unregistered_metric(self, comp_dir: Path) -> None:
        write_bundle(comp_dir, metric="not_a_metric")
        with pytest.raises(BundleError, match="not_a_metric"):
            load_bundle(comp_dir)

    def test_target_column_missing_from_train(self, comp_dir: Path) -> None:
        write_bundle(comp_dir, target_column="nonexistent")
        with pytest.raises(BundleError, match="nonexistent"):
            load_bundle(comp_dir)

    def test_id_column_missing_from_test(self, comp_dir: Path) -> None:
        write_bundle(comp_dir, id_column="nonexistent")
        with pytest.raises(BundleError, match="nonexistent"):
            load_bundle(comp_dir)

    def test_bad_metric_direction(self, comp_dir: Path) -> None:
        write_bundle(comp_dir, metric_direction="sideways")
        with pytest.raises(BundleError, match="sideways"):
            load_bundle(comp_dir)

    def test_sample_submission_must_have_id_and_target(self, comp_dir: Path) -> None:
        (comp_dir / "data" / "sample_submission.csv").write_text("wrong,cols\n1,2\n")
        with pytest.raises(BundleError, match="sample_submission"):
            load_bundle(comp_dir)

    def test_timeseries_policy_requires_time_column(self, comp_dir: Path) -> None:
        write_bundle(comp_dir, cv_policy="timeseries", time_column=None)
        with pytest.raises(BundleError, match="time_column"):
            load_bundle(comp_dir)

    def test_group_policy_requires_group_column(self, comp_dir: Path) -> None:
        write_bundle(comp_dir, cv_policy="group", group_column=None)
        with pytest.raises(BundleError, match="group_column"):
            load_bundle(comp_dir)

    def test_unknown_cv_policy(self, comp_dir: Path) -> None:
        write_bundle(comp_dir, cv_policy="magic")
        with pytest.raises(BundleError, match="magic"):
            load_bundle(comp_dir)
