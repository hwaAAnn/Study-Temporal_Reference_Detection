"""Experiment 03: sweep of the surprisal filter k (drop the fraction k of tokens the OLD model is most
surprised by, scoring.keep_mask) over the Delta-based results of 00/02/05, with a random-removal control,
a knowledge check (B vs GPT-Neo D') and the types of the removed tokens.

Input: the per-vintage score files of 01 (score_texts.py --no-pair-truncate; per-token log p and token ids).
Every comparison cuts its two texts to the shorter length first (as in year_curve.py), then filters.
Delta_k(x; new, old) = mean over kept tokens of lp_new - lp_old; the mask comes from lp_old.

Quantities per k (pair = Delta 2024-2021 unless named):
  DiD = gap_post - gap_pre on B/C        gap = mean over pairs of Delta_k(ai) - Delta_k(human)
  gap_pre (placebo, Delta 2021-2018)      D' gap = mean Delta_k(D') - Delta_k(B)  (knowledge effect at k = 0)
  period gaps H19, H22, H24, B; U = gap(H22) - gap(H24) (polish_24), U_B = gap(H22) - gap(B)
  s_post, s_pre = sum e * (Delta_k(p) - Delta_k(h)) / sum e^2 over the 05-a versions
  detection C vs B: AUROC, TPR@5% FPR for Delta 2024-2021 and 2024-2022, filter and random removal
CIs: 2000 bootstrap draws over pairs (papers for polishing); every k in a draw uses the same resample.
Removed-token types (one k, Delta 2024-2021): enrichment of each type among removed tokens, B, C, D'.

Example:
  python scripts/k_sweep.py --scores outputs/yc_20*.jsonl --polish data/polish_2022.jsonl \
      --json outputs/k_sweep.json > outputs/k_sweep.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trd.metrics import auroc, tpr_at_fpr  # noqa: E402
from trd.scoring import keep_mask  # noqa: E402

NEED = ["2018", "2021", "2022", "2024"]
POST, PRE, POST22 = ("2024", "2021"), ("2021", "2018"), ("2024", "2022")
KS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]
PERIODS = [("H19", "C19"), ("H22", "C22"), ("H24", "C24"), ("B", "C")]

# ---- token types (rules on SentencePiece strings; no POS tagger) ----
FUNCTION_WORDS = set("""a an the this that these those it its they them their there here we our us you your he she
his her i me my of in on at by for with from to into onto over under about as than then and or nor but if so
because while although though whether not no is are was were be been being am do does did done have has had
can could may might must shall should will would which who whom whose what when where why how all any each
both either neither some such only also very more most less least much many few other another same own just
via per upon within without between among across through during before after above below up down out off""".split())
LLM_EXACT = {"llm", "llms", "mllm", "mllms", "vlm", "vlms", "rlhf", "lora", "rag", "chatgpt"}
LLM_PREFIX = ("gpt", "prompt", "hallucinat", "diffusion", "transformer", "fine-tun", "finetun", "instruction",
              "in-context", "zero-shot", "few-shot", "multimodal", "multi-modal", "agent", "chatgpt", "llm")
SENT_END = (".", "!", "?", ":")
TYPES = ["word_piece", "capitalized", "digit", "symbol", "function_word", "llm_term"]


def _text(t: str) -> str:
    return "" if t.startswith("<0x") else t.replace("▁", "")


def token_types(toks: list[str]) -> dict[str, np.ndarray]:
    """Boolean flags per token (they overlap: a digit can also be a word piece).

    Words: a token starting with '▁' (or a byte token such as a newline, or the token after one) opens a word; the
    tokens after it that do not open a word belong to it, so '(BERT),' and 'fine-tuning' are one word each.
      word_piece    an alphabetic token after the word's first alphabetic token
      capitalized   the word's first alphabetic token, upper case, word not at a sentence start (names, acronyms)
      digit, symbol a token with a digit / with no letter or digit
      function_word, llm_term   every alphanumeric token of a word in FUNCTION_WORDS / matching the LLM term list
    """
    n = len(toks)
    words: list[list[int]] = []
    for i, t in enumerate(toks):
        if i == 0 or t.startswith("▁") or t.startswith("<0x") or toks[i - 1].startswith("<0x"):
            words.append([])
        words[-1].append(i)
    flags = {k: np.zeros(n, dtype=bool) for k in TYPES}
    sent_start = True
    for w in words:
        s = "".join(_text(toks[i]) for i in w)
        wl = re.sub(r"^\W+|\W+$", "", s).lower()
        alpha = [i for i in w if any(c.isalpha() for c in _text(toks[i]))]
        if alpha and not sent_start:
            first = next(c for c in _text(toks[alpha[0]]) if c.isalpha())
            flags["capitalized"][alpha[0]] = first.isupper()
        for i in w:
            ts = _text(toks[i])
            an = any(c.isalnum() for c in ts)
            flags["symbol"][i] = not an
            flags["digit"][i] = any(c.isdigit() for c in ts)
            flags["word_piece"][i] = bool(alpha) and i in alpha and i != alpha[0]
            flags["function_word"][i] = an and wl in FUNCTION_WORDS
            flags["llm_term"][i] = an and (wl in LLM_EXACT or wl.startswith(LLM_PREFIX))
        newline = any(toks[i] == "<0x0A>" for i in w)
        if s.strip() or newline:
            sent_start = newline or s.rstrip().endswith(SENT_END)
    return flags


class Tok:
    """Per-token log p (needed vintages only) and token ids for every text."""

    def __init__(self, paths: list[str]):
        self.lp: dict[str, dict[str, np.ndarray]] = {}
        self.ids: dict[str, list[int]] = {}
        self.meta: dict[str, dict] = {}
        for p in paths:
            for line in Path(p).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                for y, v in r["lp"].items():
                    if y in NEED:
                        self.lp.setdefault(r["id"], {})[y] = np.asarray(v, dtype=np.float64)
                self.ids.setdefault(r["id"], r["ids"])
                self.meta[r["id"]] = {k: r.get(k) for k in ("pair_id", "group", "generator")}
        missing = [i for i, d in self.lp.items() if sorted(d) != NEED]
        if missing:
            raise SystemExit(f"{len(missing)} texts lack one of the vintages {NEED} (e.g. {missing[0]})")

    def by_group(self, g: str) -> dict[str, str]:
        return {m["pair_id"]: i for i, m in self.meta.items() if m["group"] == g}

    def n(self, i: str) -> int:
        return len(self.lp[i]["2024"])

    def delta(self, i: str, n: int, pair: tuple[str, str], k: float, mode: str = "filter") -> float:
        new, old = self.lp[i][pair[0]][:n], self.lp[i][pair[1]][:n]
        if mode == "filter":
            m = keep_mask(old, k)
        else:  # random removal of the same number of tokens, reproducible per (text, k, pair)
            m = np.ones(n, dtype=bool)
            n_drop = int(np.floor(k * n))
            if n_drop:
                rng = np.random.default_rng(zlib.crc32(f"{i}|{k}|{pair}".encode()))
                m[rng.choice(n, n_drop, replace=False)] = False
        return float(np.mean(new[m] - old[m]))


def pairs_of(t: Tok, hg: str, ag: str) -> list[tuple[str, str, str, int]]:
    hs, as_ = t.by_group(hg), t.by_group(ag)
    out = [(p, hs[p], as_[p], min(t.n(hs[p]), t.n(as_[p]))) for p in sorted(set(hs) & set(as_))]
    if not out:
        raise SystemExit(f"no {hg}/{ag} pairs")
    return out


def gap_rows(t: Tok, prs, pair, ks, mode="filter") -> np.ndarray:
    """(pairs, k) of Delta_k(ai) - Delta_k(human)."""
    return np.array([[t.delta(a, n, pair, k, mode) - t.delta(h, n, pair, k, mode) for k in ks] for _, h, a, n in prs])


def removed_types(t: Tok, bc, bd, k: float, tokenizer: str) -> tuple[dict, dict]:
    """Share of each token type among all tokens and among the tokens the filter removes (Delta 2024-2021)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer)
    print(f"\n=== removed tokens at k = {k:g} (Delta 2024-2021, 2021 model's surprisal) ===")
    print("share among all tokens -> among removed (enrichment x)")
    types, tops = {}, {}
    groups = (("B", [(h, n) for _, h, _, n in bc]), ("C", [(a, n) for _, _, a, n in bc]),
              ("D'", [(a, n) for _, _, a, n in bd]))
    for gname, items in groups:
        allf = {k_: [] for k_ in TYPES}
        rem, dtok = [], []
        counts: dict[str, int] = {}
        for i, n in items:
            toks = tok.convert_ids_to_tokens([int(v) for v in t.ids[i][:n]])
            f = token_types(toks)
            keep = keep_mask(t.lp[i]["2021"][:n], k)
            for k_ in TYPES:
                allf[k_].append(f[k_])
            rem.append(~keep)
            dtok.append(t.lp[i]["2024"][:n] - t.lp[i]["2021"][:n])
            for s in np.array(toks)[~keep]:
                counts[s] = counts.get(s, 0) + 1
        rem_a, d_a = np.concatenate(rem), np.concatenate(dtok)
        g = {"n_tokens": int(len(rem_a)), "n_removed": int(rem_a.sum()),
             "delta_removed": float(d_a[rem_a].mean()), "delta_kept": float(d_a[~rem_a].mean())}
        line = []
        for k_ in TYPES:
            fa = np.concatenate(allf[k_])
            s_all, s_rem = float(fa.mean()), float(fa[rem_a].mean())
            g[k_] = {"all": s_all, "removed": s_rem, "enrich": s_rem / s_all if s_all > 0 else float("nan")}
            line.append(f"{k_} {s_all:.3f}->{s_rem:.3f} (x{g[k_]['enrich']:.2f})")
        types[gname] = g
        tops[gname] = sorted(counts.items(), key=lambda kv: -kv[1])[:30]
        print(f"{gname:<3} tokens {g['n_tokens']}, removed {g['n_removed']}; token Delta removed "
              f"{g['delta_removed']:+.3f} vs kept {g['delta_kept']:+.3f}")
        print("    " + "; ".join(line))
    for gname, top in tops.items():
        print(f"most often removed in {gname}: " + " ".join(f"{s}({c})" for s, c in top))
    return types, tops


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True, help="01 score files (need 2018, 2021, 2022, 2024)")
    ap.add_argument("--polish", required=True, help="data/polish_2022.jsonl (edit_sim, orig_id)")
    ap.add_argument("--ks", type=float, nargs="+", default=KS)
    ap.add_argument("--type-k", type=float, default=0.1, help="k for the removed-token types")
    ap.add_argument("--tokenizer", default="datedgpt/datedgpt-2024-base")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    args = ap.parse_args()
    ks = sorted(set(args.ks) | {0.0})
    ki = {k: j for j, k in enumerate(ks)}
    rng = np.random.default_rng(args.seed)

    t = Tok(args.scores)
    bc, bd = pairs_of(t, "B", "C"), pairs_of(t, "B", "D")
    print(f"texts: {len(t.lp)}; B/C {len(bc)} pairs, B/D' {len(bd)} pairs; k = {ks}")

    # ---- per-row values ----
    R = {"post": gap_rows(t, bc, POST, ks), "pre": gap_rows(t, bc, PRE, ks),
         "post_rand": gap_rows(t, bc, POST, ks, "random"), "pre_rand": gap_rows(t, bc, PRE, ks, "random"),
         "D": gap_rows(t, bd, POST, ks)}
    per = {h: gap_rows(t, pairs_of(t, h, a), POST, ks) for h, a in PERIODS}
    num = {"post": [], "pre": []}
    den, cl = [], []
    for line in Path(args.polish).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        p, h = str(r["id"]), f"{r['orig_id']}:h"
        if p not in t.lp or h not in t.lp:
            continue
        e, n = 1.0 - float(r["edit_sim"]), min(t.n(p), t.n(h))
        for name, pair in (("post", POST), ("pre", PRE)):
            num[name].append([e * (t.delta(p, n, pair, k) - t.delta(h, n, pair, k)) for k in ks])
        den.append(e * e)
        cl.append(r["orig_id"])
    num = {k_: np.array(v) for k_, v in num.items()}
    den = np.array(den)
    _, cl = np.unique(np.array(cl), return_inverse=True)
    print(f"polished versions: {len(den)} from {cl.max() + 1} papers")

    def stats(wbc=None, wbd=None, wper=None, wpol=None) -> dict[str, np.ndarray]:
        def mean(rows, w):
            return rows.mean(0) if w is None else (w @ rows) / w.sum()
        o = {"gap_post": mean(R["post"], wbc), "gap_pre": mean(R["pre"], wbc),
             "gap_post_rand": mean(R["post_rand"], wbc), "gap_pre_rand": mean(R["pre_rand"], wbc),
             "D_gap": mean(R["D"], wbd)}
        o["DiD"] = o["gap_post"] - o["gap_pre"]
        o["DiD_rand"] = o["gap_post_rand"] - o["gap_pre_rand"]
        for h, _ in PERIODS:
            o[f"gap_{h}"] = mean(per[h], None if wper is None else wper[h])
        o["U"] = o["gap_H22"] - o["gap_H24"]
        o["U_B"] = o["gap_H22"] - o["gap_B"]
        wp = np.ones(len(den)) if wpol is None else wpol[cl]
        for name in ("post", "pre"):
            o[f"s_{name}"] = (wp @ num[name]) / (wp @ den)
        for q in list(o):  # change from k = 0, same resample
            o[f"{q}_minus_k0"] = o[q] - o[q][0]
        return o

    point = stats()
    boots = []
    for _ in range(args.n_boot):
        w_bc = np.bincount(rng.integers(0, len(bc), len(bc)), minlength=len(bc)).astype(float)
        w_bd = np.bincount(rng.integers(0, len(bd), len(bd)), minlength=len(bd)).astype(float)
        w_per = {h: np.bincount(rng.integers(0, len(per[h]), len(per[h])), minlength=len(per[h])).astype(float)
                 for h, _ in PERIODS}
        K = cl.max() + 1
        w_pol = np.bincount(rng.integers(0, K, K), minlength=K).astype(float)
        boots.append(stats(w_bc, w_bd, w_per, w_pol))
    ci = {q: np.percentile(np.array([b[q] for b in boots]), [2.5, 97.5], axis=0) for q in point}

    def row(q, fmt="{:+.3f}"):
        return "  ".join(f"{fmt.format(point[q][j])} [{ci[q][0][j]:+.3f}, {ci[q][1][j]:+.3f}]" for j in range(len(ks)))

    print("\n=== Delta-based results by k (95% bootstrap CI over pairs) ===")
    print(f"{'quantity':<16}" + "  ".join(f"{'k=' + format(k, 'g'):<25}" for k in ks))
    for q, lab in [("DiD", "DiD"), ("gap_post", "gap_post"), ("gap_pre", "gap_pre (placebo)"),
                   ("DiD_rand", "DiD random"), ("gap_pre_rand", "gap_pre random"), ("D_gap", "D' gap"),
                   ("gap_H19", "gap H19"), ("gap_H22", "gap H22"), ("gap_H24", "gap H24"), ("gap_B", "gap B"),
                   ("U", "U (polish_24)"), ("U_B", "U_B"), ("s_post", "s_post"), ("s_pre", "s_pre")]:
        print(f"{lab:<16}" + row(q))
    print("\nchange from k = 0 (same resample)")
    for q in ["DiD", "gap_pre", "D_gap", "U", "s_post"]:
        print(f"{q:<16}" + row(f"{q}_minus_k0"))

    print("\n=== check at k = 0 against 00/01/02/05 (same texts, same lengths) ===")
    for q, ref in [("gap_post", 0.263), ("gap_pre", -0.024), ("DiD", 0.287), ("D_gap", -0.137), ("gap_H19", 0.380),
                   ("gap_H22", 0.356), ("gap_H24", 0.281), ("U", 0.074), ("U_B", 0.093), ("s_post", 0.216),
                   ("s_pre", -0.013)]:
        print(f"  {q:<10} {point[q][0]:+.3f}   (reference {ref:+.3f})")

    # ---- detection (exploratory) ----
    y = np.r_[np.zeros(len(bc)), np.ones(len(bc))]
    det = {}
    print("\n=== detection C vs B (exploratory): AUROC (TPR@5%FPR) ===")
    print(f"{'pair / removal':<22}" + "".join(f"{'k=' + format(k, 'g'):>16}" for k in ks))
    for pname, pair in (("2024-2021", POST), ("2024-2022", POST22)):
        for mode in ("filter", "random"):
            vals = []
            for k in ks:
                x = np.r_[[t.delta(h, n, pair, k, mode) for _, h, _, n in bc],
                          [t.delta(a, n, pair, k, mode) for _, _, a, n in bc]]
                vals.append({"auroc": auroc(y, x), "tpr5": tpr_at_fpr(y, x, 0.05)})
            det[f"{pname}_{mode}"] = vals
            print(f"{pname + ' ' + mode:<22}" + "".join(f"{v['auroc']:>9.3f} ({v['tpr5']:.2f})" for v in vals))

    # ---- removed-token types (only when type_k is one of the swept k) ----
    types, tops = {}, {}
    if args.type_k in ks and args.type_k > 0:
        types, tops = removed_types(t, bc, bd, args.type_k, args.tokenizer)

    # ---- hypotheses ----
    j05, j10, j20 = ki.get(0.05), ki.get(0.1), ki.get(0.2)
    verdict = {}
    if None not in (j05, j10, j20):
        verdict["H1"] = all(ci["DiD"][0][j] > 0 and point["DiD"][j] >= 0.5 * point["DiD"][0] for j in (j05, j10, j20))
        verdict["H2"] = bool(np.all(np.abs(point["gap_pre"]) <= 0.05))
        shrink = 1 - point["D_gap"][j10] / point["D_gap"][0]
        lo, hi = ci["D_gap_minus_k0"][0][j10], ci["D_gap_minus_k0"][1][j10]
        verdict["H3a"] = bool(shrink >= 0.25 and (lo > 0 or hi < 0))
        b = types["B"]
        verdict["H3b"] = bool(b["capitalized"]["enrich"] > 1 and b["digit"]["enrich"] > 1
                              and b["llm_term"]["enrich"] > 1 and b["function_word"]["enrich"] < 1)
        verdict["H4"] = bool(ci["U"][0][j10] > 0 and ci["s_post"][0][j10] > 0)
        print("\n=== hypotheses ===")
        print(f"D' gap shrink at k = 0.1: {shrink:.0%}")
        for h_, v in verdict.items():
            print(f"{h_}: {'holds' if v else 'fails'}")

    if args.json:
        out = {"ks": ks, "point": {q: v.tolist() for q, v in point.items()},
               "ci": {q: v.tolist() for q, v in ci.items()}, "detection": det, "types": types,
               "top_removed": tops, "verdict": verdict}
        Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
