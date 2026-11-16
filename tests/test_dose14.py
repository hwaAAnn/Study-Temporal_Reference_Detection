"""Model-free checks for the 14 analysis: Cohen's d rows, D1 status boundaries, and planted synthetic grids."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import dose14 as dz  # noqa: E402
from trd.metrics import cohens_d  # noqa: E402

G_TRUE = {
    "DG": {"2013": 0.95, "2019": 0.95, "2022": 0.95, "2023": 1.15, "2024": 1.22},
    "TiC": {"201305": 1.07, "201901": 1.07, "202101": 1.07, "202301": 1.07, "202407": 1.25},
    "Falcon": {"base": 0.80, "inst": 1.05},
}


def test_cohens_d_rows_matches_metrics():
    rng = np.random.default_rng(0)
    a, h = rng.normal(1, 1, 50), rng.normal(0, 2, 50)
    assert abs(dz.cohens_d_rows(a[None, :], h[None, :])[0] - cohens_d(a, h)) < 1e-12


def test_d1_status_boundaries():
    assert dz.d1_status([0.1, 0.4], 0.25, 0.25, [0.5, 1.0]) == "proportional"
    assert dz.d1_status([0.6, 0.9], 0.75, 0.25, [0.5, 1.0]) == "graded"
    assert dz.d1_status([-0.1, 0.2], 0.05, 0.25, [0.5, 1.0]) == "step"
    assert dz.d1_status([0.7, 1.2], 0.95, 0.25, [0.5, 1.0]) == "no-dose"
    assert dz.d1_status([-0.2, 1.3], 0.5, 0.25, [0.5, 1.0]) == "wide"
    assert dz.d1_status([0.1, 0.4], 0.25, 0.25, [-0.1, 1.0]) == "undecidable"


def fake_family(fam: str, loss, n: int = 400, seed: int = 1) -> dict:
    """ll carries G_TRUE exactly; AI detector scores get +gain_scale * loss(r, s) on top of N(0,1) noise."""
    rng = np.random.default_rng(seed)
    ms = dz.FAMILIES[fam]["models"]
    rows = {}
    for hg, ag, _ in dz.SETS:
        for k in range(n):
            pid = f"{hg}{k}"
            noise_h, noise_a = rng.normal(0, 1, 2)
            for g, lab in ((hg, 0), (ag, 1)):
                cells = {}
                text_z = rng.normal(0, 1)  # real grids: one text scores alike in every cell
                for r in ms:
                    for s in ms:
                        z = text_z + rng.normal(0, 0.3)
                        f = z + (2.0 + loss(fam, r, s) if lab else 0.0)
                        ll = -3.0 + (noise_a if lab else noise_h) * 0.01 + (G_TRUE[fam][s] if lab else 0.0)
                        cells[f"{r}_{s}"] = {"fdgpt": f, "bino": -f, "ll": ll}
                rows[f"{pid}:{lab}"] = {"id": f"{pid}:{lab}", "pair_id": pid, "group": g, "label": lab,
                                        "generator": "gen", "cells": cells}
    return rows


def proportional(fam, r, s):
    new = dz.FAMILIES[fam]["new"]
    return 8.0 * (G_TRUE[fam][new] - G_TRUE[fam][r]) if s == new else 0.0


def step(fam, r, s):
    new = dz.FAMILIES[fam]["new"]
    pre = G_TRUE[fam][r] < G_TRUE[fam][new] - 0.1  # 2023 counts as absorbed
    return 2.0 if (s == new and pre) else 0.0


def run(loss):
    fams = {f: fake_family(f, loss) for f in dz.FAMILIES}
    return dz.evaluate(fams, n_boot=200)


def test_planted_proportional():
    res = run(proportional)
    assert abs(res["dose"]["DG"]["2024"] - res["dose"]["DG"]["2022"] - 0.27) < 0.01
    assert res["verdict"]["D1"] == {"F": "proportional", "Bi": "proportional"}
    assert res["verdict"]["dose_response"] == "quantitative"
    assert res["verdict"]["D4"]["F"]["TiC"] == "same scale"


def test_planted_step():
    res = run(step)
    assert res["verdict"]["D1"] == {"F": "step", "Bi": "step"}
    assert res["verdict"]["dose_response"] == "step"
