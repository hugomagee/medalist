import numpy as np

from strategies.ames_blend import convex_weights, nested_blend


def test_convex_weights_recover_mixture_and_stay_on_simplex():
    rng = np.random.default_rng(0)
    y = rng.normal(12, 0.4, 500)
    a = y + rng.normal(0, 0.10, 500)
    b = y + rng.normal(0, 0.10, 500)
    c = y + rng.normal(0, 0.50, 500)
    w = convex_weights(np.column_stack([a, b, c]), y)
    assert np.all(w >= 0) and abs(w.sum() - 1) < 1e-9
    assert w[2] < 0.1 and abs(w[0] - w[1]) < 0.2


def test_nested_blend_scores_each_fold_with_weights_fit_elsewhere():
    rng = np.random.default_rng(1)
    y = rng.normal(12, 0.4, 100)
    P = np.column_stack([y + rng.normal(0, 0.1, 100), y + rng.normal(0, 0.2, 100)])
    idx = np.arange(100)
    folds = [(np.setdiff1d(idx, idx[k::4]), idx[k::4]) for k in range(4)]
    oof, test, w, per_fold = nested_blend(P, P[:10], y, folds)
    assert len(per_fold) == 4 and oof.shape == (100,) and test.shape == (10,)
    for (tr, va), wf in zip(folds, per_fold):
        assert np.allclose(oof[va], P[va] @ wf)
