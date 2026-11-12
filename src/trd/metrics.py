"""Detection metrics. Convention: label 1 = AI, higher score = more AI-like."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auroc(labels, scores) -> float:
    return float(roc_auc_score(labels, scores))


def tpr_at_fpr(labels, scores, fpr_target: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    ok = fpr <= fpr_target
    return float(tpr[ok].max()) if ok.any() else 0.0


def pair_win_rate(ai_scores, human_scores) -> float:
    """Fraction of matched pairs where the AI text scores higher than its human counterpart."""
    a, h = np.asarray(ai_scores), np.asarray(human_scores)
    return float(np.mean(a > h))


def cluster_bootstrap(stat, labels, scores, clusters, n_boot: int = 1000, seed: int = 0,
                      alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI of stat(labels, scores), resampling whole clusters (e.g. a human/AI pair).

    Resampling pairs rather than single texts keeps the two halves of a pair together, which is the
    right unit when texts were built in matched pairs.
    """
    labels, scores, clusters = np.asarray(labels), np.asarray(scores), np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inv == c) for c in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = np.concatenate([members[c] for c in rng.integers(0, len(uniq), len(uniq))])
        if len(np.unique(labels[pick])) < 2:
            continue
        vals.append(stat(labels[pick], scores[pick]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
