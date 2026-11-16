"""Model-free checks for the 09 analysis: the verdict on planted and null synthetic scores, and partial grids."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import det09  # noqa: E402

MAIN_V = ["2018", "2021", "2022", "2024"]
CELLS = [f"{r}_{s}" for r in MAIN_V for s in MAIN_V]


def fake(effect: dict[str, float], n: int = 300, seed: int = 3, rest_cells: bool = False):
    """Two main pools (RAID chat sampling, APT strong) with every cell; AI texts get +effect[key]."""
    rng = np.random.default_rng(seed)
    meta, sc = {}, {}

    def add(i, group, label, cluster, **kw):
        meta[i] = dict(group=group, label=label, cluster=cluster, **kw)
        cells = ["2022_2022", "2022_2024", "2024_2022", "2024_2024"] if rest_cells and label else CELLS
        s = {f"{d}:{c}": rng.normal() + (effect.get(c, 0.0) if label else 0.0) for d in det09.DETS for c in cells}
        for b in det09.BASE:
            s[b] = rng.normal() + (effect.get(b, 0.0) if label else 0.0)
        sc[i] = s

    for k in range(n):
        add(f"rh{k}", "RAID.H", 0, f"r{k}")
        add(f"ra{k}", "RAID.chatgpt.sampling", 1, f"r{k}")
        add(f"ah{k}", "APT.H", 0, f"a{k}", domain="blog")
        add(f"aa{k}", "APT.GPT-4o.major", 1, f"a{k}", domain="blog", polisher="GPT-4o", level="major")
    return meta, sc


PLANTED = {"2022_2024": 2.0, "2022_2022": 0.8, "2024_2024": 0.3, "2018_2021": 0.3, "2021_2021": 0.8,
           "bino1b": 0.8, "s09": 0.8}


def test_planted_effects_hold():
    meta, sc = fake(PLANTED)
    res = det09.evaluate(meta, sc, n_boot=300)
    v = res["verdict"]
    for p in (det09.P1, det09.P2):
        for d in det09.DETS:
            for h in ("H1", "H2", "H3", "H4"):
                assert v["per_pool"][p][d][h] == "holds", (p, d, h)
    for d in det09.DETS:
        assert v["per_pool"][det09.P2][d]["H5a"] == "holds"
        assert v["per_pool"][det09.P2][d]["H5b"] == "holds"
        assert "H5a" not in v["per_pool"][det09.P1][d]
    assert v["generalises"] is True


def test_null_fails():
    meta, sc = fake({})
    res = det09.evaluate(meta, sc, n_boot=300)
    v = res["verdict"]
    assert v["generalises"] is False
    held = [h for p in v["per_pool"].values() for d in p.values() for h, x in d.items() if x == "holds"]
    assert len(held) <= 2, held  # 20 tests at 2.5% each; tests sharing the 2022_2024 cell move together


def test_boundary_control_cancels_a_plain_two_model_gain():
    """If any r != s cell gains the same, H2 holds but H3 (boundary) fails, up to 2.5%-level chance.

    Noise is independent per cell here, so each H3 test can still hold by chance; allow one of four per seed.
    """
    eff = {c: (1.5 if c.split("_")[0] != c.split("_")[1] else 0.5) for c in CELLS}
    for seed in (3, 4, 5):
        meta, sc = fake(eff, seed=seed)
        v = det09.evaluate(meta, sc, n_boot=300)["verdict"]
        h3 = [v["per_pool"][p][d]["H3"] for p in (det09.P1, det09.P2) for d in det09.DETS]
        assert all(v["per_pool"][p][d]["H2"] == "holds" for p in (det09.P1, det09.P2) for d in det09.DETS)
        assert h3.count("holds") <= 1, (seed, h3)


def test_pool_with_fewer_cells_uses_the_shared_ones():
    meta, sc = fake(PLANTED, rest_cells=True)
    res = det09.evaluate(meta, sc, n_boot=50)
    a = res["pools"][det09.P1]["auroc"]
    assert "F:2022_2024" in a and "F:2018_2021" not in a
    con = res["pools"][det09.P1]["contrasts"]["F"]
    assert "H1" in con and "H3" not in con
