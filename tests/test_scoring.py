import numpy as np
import pytest

from trd.metrics import auroc, cluster_bootstrap, pair_win_rate, tpr_at_fpr
from trd.scoring import delta_score, keep_mask


def test_keep_mask_k0_keeps_all():
    assert keep_mask(np.array([-1.0, -5.0, -2.0]), 0.0).all()


def test_keep_mask_drops_most_surprising_old_tokens():
    lp_old = np.array([-1.0, -9.0, -2.0, -8.0])
    m = keep_mask(lp_old, 0.5)
    assert m.tolist() == [True, False, True, False]


def test_delta_score_mean_difference():
    lp_new = np.array([-1.0, -2.0, -3.0])
    lp_old = np.array([-2.0, -2.0, -5.0])
    assert delta_score(lp_new, lp_old) == pytest.approx((1 + 0 + 2) / 3)


def test_delta_score_with_filter_ignores_unknown_tokens():
    lp_new = np.array([-1.0, -1.0, -1.0, -1.0])
    lp_old = np.array([-1.5, -20.0, -1.5, -1.5])  # token 1 is a post-cutoff term
    assert delta_score(lp_new, lp_old, k=0.25) == pytest.approx(0.5)


def test_delta_score_length_mismatch_raises():
    with pytest.raises(ValueError):
        delta_score(np.zeros(3), np.zeros(4))


def test_metrics_on_separable_scores():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert auroc(labels, scores) == 1.0
    assert tpr_at_fpr(labels, scores, 0.01) == 1.0
    assert pair_win_rate([0.8, 0.9], [0.1, 0.95]) == 0.5


def test_cluster_bootstrap_brackets_point_estimate():
    rng = np.random.default_rng(1)
    n = 200
    labels = np.repeat([0, 1], n)
    scores = np.concatenate([rng.normal(0, 1, n), rng.normal(1, 1, n)])
    clusters = np.tile(np.arange(n), 2)  # pair i = (human i, AI i)
    lo, hi = cluster_bootstrap(auroc, labels, scores, clusters, n_boot=300)
    assert lo < auroc(labels, scores) < hi
    assert 0.6 < lo and hi < 0.9


def test_cluster_bootstrap_keeps_pairs_together():
    # if whole pairs are resampled, every resample has as many AI texts as human texts
    labels = np.array([0, 1] * 5)
    scores = np.arange(10.0)
    clusters = np.repeat(np.arange(5), 2)
    balance = lambda l, s: float((l == 1).sum() - (l == 0).sum())
    assert cluster_bootstrap(balance, labels, scores, clusters, n_boot=200) == (0.0, 0.0)
