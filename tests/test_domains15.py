"""Model-free checks for experiment 15: planted vintage curves give the pre-registered verdicts."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import domains15 as d15  # noqa: E402
import year_curve as yc  # noqa: E402
from build_15_input import GENS  # noqa: E402

DOMS = ["abstracts", "books", "news", "poetry", "recipes", "reddit", "reviews", "wiki"]
JUMP = {2023: 0.15, 2024: 0.25}


def fake_scores(jumping: set[str], n_src: int = 30, n_tok: int = 40, seed: int = 0) -> yc.Scores:
    """Per-token log p = text offset + 1.0 for AI text + JUMP for generators in `jumping` + small noise."""
    rng = np.random.default_rng(seed)
    sc = object.__new__(yc.Scores)
    sc.cs, sc.meta = {}, {}
    for dom in DOMS:
        for s in range(n_src):
            for model in ["H"] + GENS:
                i = f"{dom}:{s}:{model}"
                off = rng.normal(0, 0.3) + (0.0 if model == "H" else 1.0)
                lp = np.empty((len(yc.YEARS), n_tok))
                for k, y in enumerate(yc.YEARS):
                    jump = JUMP.get(y, 0.0) if model in jumping else 0.0
                    lp[k] = off + jump + rng.normal(0, 0.05, n_tok)
                sc.cs[i] = np.cumsum(lp, axis=1)
                sc.meta[i] = {"pair_id": str(s), "group": f"{dom}.{model}", "label": int(model != "H"),
                              "generator": "" if model == "H" else model, "n_tokens": n_tok}
    return sc


def test_chat_only_jump_is_general_chat_style():
    res = d15.analyse(fake_scores(set(d15.CHAT)), n_boot=100)
    j = res["judge"]
    assert j["verdict"] == "general (chat style)"
    assert j["counts"] == {"D1": 8, "D2": 8, "D3": 8, "D4": 8}
    assert j["D5"] == {"chatgpt": True, "gpt4": True}
    assert abs(res["curves"]["G_chat[all]"]["r2024"] - 0.25) < 0.02
    assert abs(res["curves"]["K[all]"]["ratio_to_G_chat"] - 1.0) < 0.1


def test_every_ai_text_jumps_is_general_ai_text():
    res = d15.analyse(fake_scores(set(GENS)), n_boot=100)
    j = res["judge"]
    assert j["counts"]["D1"] == 8 and j["counts"]["D3"] == 0 and j["counts"]["D4"] == 0
    assert j["verdict"] == "general (AI text)"


def test_no_jump_is_narrow():
    res = d15.analyse(fake_scores(set()), n_boot=100)
    assert res["judge"]["counts"]["D1"] <= 1
    assert res["judge"]["verdict"] == "narrow"
