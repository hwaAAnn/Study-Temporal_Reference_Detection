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
