"""Model-free checks of the baseline detector formulas (experiment 04): per-token loops vs the batched code,
and invariance to padding."""

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from baseline_scores import binoculars_terms, fdgpt_terms  # noqa: E402


def _rand(b, L, V, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, L, V, generator=g) * 2


def _logsm(x):
    x = np.asarray(x, dtype=np.float64)
    return x - x.max() - math.log(np.exp(x - x.max()).sum())


def test_binoculars_matches_a_per_token_loop():
    L, V = 6, 11
    p_log, q_log = _rand(1, L, V, 0), _rand(1, L, V, 1)
    ids = torch.randint(0, V, (1, L), generator=torch.Generator().manual_seed(2))
    nll, xent = binoculars_terms(p_log, q_log, ids, torch.ones(1, L, dtype=torch.long))
    a = b = 0.0
    for t in range(L - 1):
        q = _logsm(q_log[0, t].numpy())
        p = np.exp(_logsm(p_log[0, t].numpy()))
        a += -q[ids[0, t + 1]]
        b += -(p * q).sum()
    assert abs(nll.item() - a / (L - 1)) < 1e-4 and abs(xent.item() - b / (L - 1)) < 1e-4


def test_fdgpt_matches_a_per_token_loop():
    L, V = 7, 9
    r_log, s_log = _rand(1, L, V, 3), _rand(1, L, V, 4)
    ids = torch.randint(0, V, (1, L), generator=torch.Generator().manual_seed(5))
    disc, ll = fdgpt_terms(r_log, s_log, ids, torch.ones(1, L, dtype=torch.long))
    num = var = lls = 0.0
    for t in range(L - 1):
        s = _logsm(s_log[0, t].numpy())
        r = np.exp(_logsm(r_log[0, t].numpy()))
        mu = (r * s).sum()
        num += s[ids[0, t + 1]] - mu
        var += (r * s * s).sum() - mu * mu
        lls += s[ids[0, t + 1]]
    assert abs(disc.item() - num / math.sqrt(var)) < 1e-4 and abs(ll.item() - lls / (L - 1)) < 1e-4


def test_padding_does_not_change_scores():
    V = 13
    long_l, short_l = _rand(1, 8, V, 6), _rand(1, 5, V, 7)
    ids_long = torch.randint(0, V, (1, 8), generator=torch.Generator().manual_seed(8))
    ids_short = torch.randint(0, V, (1, 5), generator=torch.Generator().manual_seed(9))
    pad = torch.cat([short_l, torch.randn(1, 3, V)], 1)
    ids = torch.cat([ids_long, torch.cat([ids_short, torch.zeros(1, 3, dtype=torch.long)], 1)])
    mask = torch.tensor([[1] * 8, [1] * 5 + [0] * 3])
    logits = torch.cat([long_l, pad])
    for fn in (binoculars_terms, fdgpt_terms):
        both = fn(logits, logits * 0.7, ids, mask)
        alone = fn(short_l, short_l * 0.7, ids_short, torch.ones(1, 5, dtype=torch.long))
        for x, y in zip(both, alone):
            assert abs(x[1].item() - y[0].item()) < 1e-4
