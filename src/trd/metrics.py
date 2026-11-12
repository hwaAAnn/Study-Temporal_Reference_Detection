"""Detection metrics. Convention: label 1 = AI, higher score = more AI-like."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def auroc(labels, scores) -> float:
    return float(roc_auc_score(labels, scores))


def auprc(labels, scores) -> float:
    return float(average_precision_score(labels, scores))


def tpr_at_fpr(labels, scores, fpr_target: float = 0.01) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    ok = fpr <= fpr_target
    return float(tpr[ok].max()) if ok.any() else 0.0


def pair_win_rate(ai_scores, human_scores) -> float:
    """Fraction of matched pairs where the AI text scores higher than its human counterpart."""
    a, h = np.asarray(ai_scores), np.asarray(human_scores)
    return float(np.mean(a > h))


def cohens_d(x, y) -> float:
    """Effect size of mean(x) - mean(y) with pooled SD."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return float((x.mean() - y.mean()) / sp) if sp > 0 else float("nan")


def cluster_bootstrap_ci(clusters, stat: Callable[[np.ndarray], float], n_boot: int = 1000,
                         alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile CI of stat(idx) when resampling whole clusters (e.g. pair_id) with replacement.

    stat receives an index array into the original items and returns a float.
    Resampling by cluster keeps a human text and its matched AI text together.
    """
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    members = {c: np.flatnonzero(clusters == c) for c in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[c] for c in pick])
        try:
            v = stat(idx)
        except ValueError:  # e.g. a resample with one class only
            continue
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
