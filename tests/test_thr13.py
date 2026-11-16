"""Model-free checks for the 13 threshold-transfer analysis on synthetic scores."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import thr13  # noqa: E402

KEYS = [f"{d}:{c}" for d in ("F", "Bi") for c in thr13.DG_CELLS]


def fake(ai_shift: dict[str, float], late_human_shift: dict[str, float], n: int = 400, seed: int = 1):
    # n = 400 matches the real calibration (H19) and RAID.H sizes; T2's "CI upper <= 0.10" depends on n
    """Humans ~ N(0) except H24/B humans get +late_human_shift[cell]; AI get +ai_shift[cell]."""
    rng = np.random.default_rng(seed)
    meta, sc = {}, {}
    sets = [("H19", "C19"), ("RAID.H", "RAID.chatgpt.sampling"), ("APT.H", "APT.GPT-4o.major"), ("H24", "C24"),
            ("B", "C")]
    for hg, ag in sets:
        for k in range(n):
            pid = f"{hg}{k}"
            for lab, g in ((0, hg), (1, ag)):
                i = f"{pid}:{lab}"
                meta[i] = (g, lab, pid)
                sc[i] = {}
                for key in KEYS:
                    cell = key.split(":")[1]
                    m = ai_shift.get(cell, 0.0) if lab else (late_human_shift.get(cell, 0.0) if hg in ("H24", "B") else 0.0)
                    sc[i][key] = rng.normal() + m
    return meta, sc


def run(meta, sc):
    return thr13.evaluate(meta, sc, thr13.TARGETS["local"], n_boot=200, seed=0)


def test_planted_rule_survives_and_late_humans_flagged():
    res = run(*fake({"2022_2024": 2.5, "2022_2022": 1.5, "2024_2024": 1.0}, {"2022_2024": 1.0}))
    v = thr13.verdict(res)
    for d in ("F", "Bi"):
        assert v[f"T1 {d}"]["status"] == "holds"
        assert v[f"T2 {d}"]["status"] == "holds"
        assert v[f"T3 {d}"]["status"] == "holds"
    assert v["rule survives transfer (T1 for F and Bi)"] is True


def test_t2_catches_an_inflated_fpr():
    """RAID humans shifted +0.6 sd toward AI: true FPR at the H19 threshold is ~0.15, so T2 must fail."""
    meta, sc = fake({"2022_2024": 2.5, "2022_2022": 1.5, "2024_2024": 1.0}, {})
    for i, (g, lab, _) in meta.items():
        if g == "RAID.H":
            for key in sc[i]:
                sc[i][key] += 0.6
    v = thr13.verdict(run(meta, sc))
    for d in ("F", "Bi"):
        assert v[f"T2 {d}"]["status"] == "fails", v[f"T2 {d}"]


def test_null_fails():
    res = run(*fake({c: 1.5 for c in thr13.DG_CELLS}, {}))
    v = thr13.verdict(res)
    for d in ("F", "Bi"):
        assert v[f"T1 {d}"]["status"] == "fails"
        assert v[f"T3 {d}"]["status"] == "fails"
    assert v["rule survives transfer (T1 for F and Bi)"] is False


def test_transferred_threshold_is_the_calibration_95th_percentile():
    meta, sc = fake({}, {})
    res = run(meta, sc)
    t = res["targets"]["H19/C19 (calibration set, no transfer)"]
    for key in KEYS:
        assert t["fpr"][key] <= 0.05 + 1e-9  # in-set: at most 5% of the calibration humans lie above
