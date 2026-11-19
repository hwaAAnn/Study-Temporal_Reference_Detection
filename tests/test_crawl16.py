"""Model-free checks for experiment 16: planted per-year marker rates give the pre-registered verdicts."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import crawl16 as c16  # noqa: E402


def fake(rise: dict[int, float], n_rg: int = 8, seed: int = 0) -> c16.Counts:
    """Row groups of 1,000 docs x 500 words; mark rate 2 per 1,000 words + planted rise, Poisson counts."""
    rng = np.random.default_rng(seed)
    rows = []
    for y in range(2013, 2026):
        for g in range(n_rg):
            for d in range(50):  # 50 docs per row group keep the test fast
                lam = 500 * (2.0 + rise.get(y, 0.0)) / 1000
                rows.append({"year": y, "dump": f"CC-MAIN-{y}-01", "file": 0, "rg": g, "words": 500,
                             "mark": int(rng.poisson(lam)), "rare": int(rng.poisson(lam / 4)),
                             "phrase": int(rng.poisson(lam / 20))})
    return c16.Counts(rows)


def test_rise_from_2023_supports_first_link():
    res = c16.analyse(fake({2023: 0.4, 2024: 0.6, 2025: 0.7}), fake({2023: 0.1, 2024: 0.15}, seed=1), n_boot=200)
    assert res["criteria"]["C1"] and res["criteria"]["C2"] and res["criteria"]["C3"] and res["criteria"]["C4"]
    assert res["verdict"] == "first link supported"


def test_rise_only_in_2024():
    res = c16.analyse(fake({2024: 0.6}), fake({2024: 0.6}, seed=1), n_boot=200)
    assert res["criteria"]["C1"] and not res["criteria"]["C2"]
    assert res["verdict"] == "visible only in 2024"


def test_flat_is_not_visible():
    res = c16.analyse(fake({}), fake({}, seed=1), n_boot=200)
    assert res["verdict"] == "not visible"
