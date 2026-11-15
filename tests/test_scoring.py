import numpy as np
import pytest

from trd.metrics import auroc, pair_win_rate, tpr_at_fpr
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


from trd.data import common_lengths, model_key, normalise_rows
from trd.metrics import auprc, cluster_bootstrap_ci, cohens_d


def test_normalise_pairs_and_groups():
    items = normalise_rows([
        {"id": 1, "human": "h", "ai": "a", "generator": "qwen"},
        {"id": "x", "text": "t", "group": "D", "pair_id": "1"},
    ])
    assert [(i["group"], i["label"], i["pair_id"]) for i in items] == [("B", 0, "1"), ("C", 1, "1"), ("D", 1, "1")]


def test_normalise_rejects_unknown_group():
    with pytest.raises(ValueError):
        normalise_rows([{"id": 1, "text": "t", "group": "Z"}])


def test_common_lengths_cut_a_topic_to_its_shortest_text():
    assert common_lengths(["p1", "p1", "p2", "p1", "p2"], [199, 150, 120, 180, 199]) == [150, 150, 120, 150, 120]


def test_model_key():
    assert model_key("datedgpt/datedgpt-2021-base") == "2021"
    assert model_key("manelalab/chrono-gpt-v1-20241231") == "2024"
    assert model_key("gpt2") == "gpt2"


def test_cohens_d_and_auprc():
    assert cohens_d([1, 2, 3], [1, 2, 3]) == 0
    assert cohens_d([2, 3, 4], [1, 2, 3]) == pytest.approx(1.0)
    assert auprc([0, 1], [0.1, 0.9]) == 1.0


def test_cluster_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    x = rng.normal(1.0, 1.0, 200)
    clusters = np.repeat(np.arange(100), 2)
    lo, hi = cluster_bootstrap_ci(clusters, lambda idx: float(x[idx].mean()), n_boot=300)
    assert lo < x.mean() < hi and lo > 0.5


from trd.scoring import with_bos


def test_with_bos_prepends_once_and_counts_bos_in_max_tokens():
    assert with_bos([5, 6, 7], 0, 3) == [0, 5, 6]
    assert with_bos([0, 5, 6], 0, 200) == [0, 5, 6]  # tokenizer already added it
    assert with_bos([5, 6], None, 200) == [5, 6]     # no BOS requested
    assert with_bos([], 0, 200) == [0]
