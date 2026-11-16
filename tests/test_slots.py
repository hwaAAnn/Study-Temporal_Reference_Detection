"""Model-free checks for the 08 analysis: AUROC with ties, and the verdict on planted and null synthetic grids."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import slots_grid as sg  # noqa: E402

V = ["2013", "2019", "2022", "2023", "2024"]
POST = {"2023": 1.0, "2024": 1.0}


def test_auroc_ties_and_direction():
    assert sg.auroc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert sg.auroc(np.array([0.0, 1.0]), np.array([2.0, 3.0])) == 0.0
    assert sg.auroc(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.5


def fake_grid(scorer_gain: float, ref_loss: float, n: int = 400, seed: int = 1) -> list[dict]:
    """AI texts get +scorer_gain when s is post-ChatGPT and -ref_loss when r is; humans get noise only."""
    rng = np.random.default_rng(seed)
    rows = []
    for hg, ag in (("H19", "C19"), ("B", "C"), ("H24", "C24")):
        for k in range(n):
            base_h, base_a = rng.normal(0, 1), rng.normal(2.0, 1)
            for g, lab, base in ((hg, 0, base_h), (ag, 1, base_a)):
                cells = {}
                for r in V:
                    for s in V:
                        f = base + rng.normal(0, 0.3)
                        if lab:
                            f += scorer_gain * POST.get(s, 0) - ref_loss * POST.get(r, 0)
                        cells[f"{r}_{s}"] = {"fdgpt": f, "bino": -f, "ll": f, "ppl": 0.0, "xppl": 0.0}
                rows.append({"id": f"{hg}{k}:{lab}", "pair_id": f"{hg}{k}", "group": g, "label": lab,
                             "generator": "g", "n_tokens": 100, "cells": cells})
    return rows


def test_planted_reference_loss_is_adopted():
    res = sg.evaluate(fake_grid(scorer_gain=0.8, ref_loss=0.8), n_boot=200)
    v = res["verdict"]
    assert v["direction"] == "adopt"
    main = v["per_set"]["H19/C19"]["F"]
    assert main["H1"] == "holds" and main["H3"] == "holds" and main["H4"] == "holds"


def test_scorer_gain_only_is_dropped():
    res = sg.evaluate(fake_grid(scorer_gain=0.8, ref_loss=0.0), n_boot=200)
    v = res["verdict"]
    assert v["direction"] == "drop"
    assert v["per_set"]["H19/C19"]["F"]["H1"] == "holds"
    assert v["per_set"]["H19/C19"]["F"]["H3"] == "fails"


def test_null_is_dropped():
    res = sg.evaluate(fake_grid(scorer_gain=0.0, ref_loss=0.0), n_boot=200)
    assert res["verdict"]["direction"] == "drop"
    assert res["verdict"]["per_set"]["H19/C19"]["F"]["H1"] == "fails"
