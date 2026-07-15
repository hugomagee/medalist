import numpy as np
import pytest

from strategies.ensembling import blend, rank_average, stack


class TestBlend:
    def test_equal_weights_by_default(self) -> None:
        out = blend([np.array([0.0, 2.0]), np.array([2.0, 4.0])])
        assert out.tolist() == [1.0, 3.0]

    def test_explicit_weights_normalised(self) -> None:
        out = blend([np.array([0.0]), np.array([10.0])], weights=[3.0, 1.0])
        assert out[0] == pytest.approx(2.5)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            blend([np.array([1.0]), np.array([1.0, 2.0])])


class TestRankAverage:
    def test_hand_computed(self) -> None:
        # ranks (0-based, normalised to [0, 1]): a -> [0, .5, 1], b -> [.5, 0, 1]
        a = np.array([10.0, 20.0, 30.0])
        b = np.array([5.0, 1.0, 9.0])
        out = rank_average([a, b])
        assert out.tolist() == [0.25, 0.25, 1.0]

    def test_scale_invariant(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        assert rank_average([a, a * 1000]).tolist() == rank_average([a, a]).tolist()


class TestStack:
    def test_recovers_linear_combination(self) -> None:
        r = np.random.default_rng(0)
        n = 200
        y = r.normal(size=n)
        oof_a = y + r.normal(scale=0.1, size=n)
        oof_b = r.normal(size=n)  # pure noise
        test_a = np.linspace(-1, 1, 50)
        test_b = r.normal(size=50)
        folds = [
            (np.arange(0, 100), np.arange(100, 200)),
            (np.arange(100, 200), np.arange(0, 100)),
        ]
        meta_oof, meta_test = stack([oof_a, oof_b], [test_a, test_b], y, folds)
        # stacker should lean on the informative model: meta OOF tracks y closely
        assert abs(np.corrcoef(meta_oof, y)[0, 1]) > 0.95
        assert meta_test.shape == (50,)
