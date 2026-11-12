"""Which tokens make the AI-human gap in Delta? Reading aid for experiment 00 (and the k-sweep later).

For each token type v, its share of the gap is
    contrib(v) = sum_{AI tokens = v} d_t / N_AI  -  sum_{human tokens = v} d_t / N_human,   d_t = lp_new - lp_old
so the contributions add up to mean_t(d | AI) - mean_t(d | human), the token-weighted Delta gap.

Also prints the texts with the highest and lowest Delta, to read them.

Example:
  python scripts/token_contrib.py --scores outputs/k0.jsonl --pairs data/pairs.jsonl --top 30
  python scripts/token_contrib.py --scores outputs/pilot_k0.jsonl --pairs data/pilot_pairs.jsonl \
      --where generator=gpt-4 domain=xsum
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(scores: str, pairs: str | None, tokenizer: str, where: list[str], max_tokens: int):
    recs = [json.loads(l) for l in Path(scores).read_text(encoding="utf-8").splitlines() if l.strip()]
    for cond in where:
        k, v = cond.split("=", 1)
        recs = [r for r in recs if str(r.get(k)) == v]
    texts = {}
    if pairs:
        for l in Path(pairs).read_text(encoding="utf-8").splitlines():
            if l.strip():
                p = json.loads(l)
                texts[(p["id"], p["generator"], 0)] = p["human"]
                texts[(p["id"], p["generator"], 1)] = p["ai"]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer)
    for r in recs:
        if "ids" not in r:  # older score files: re-tokenize exactly as score_pairs.py did
            ids = tok(texts[(r["id"], r["generator"], r["label"])], add_special_tokens=True)["input_ids"][:max_tokens]
            r["ids"] = ids[1:]
        if len(r["ids"]) != len(r["lp_new"]):
            raise SystemExit(f"token/log-prob length mismatch for {r['id']}")
        r["text"] = texts.get((r["id"], r["generator"], r["label"]), "")
    return recs, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--pairs", default=None, help="needed for old score files without ids, and to print texts")
    ap.add_argument("--tokenizer", default="datedgpt/datedgpt-2024-base")
    ap.add_argument("--where", nargs="*", default=[], help="field=value filters, e.g. generator=gpt-4")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()

    recs, tok = load(args.scores, args.pairs, args.tokenizer, args.where, args.max_tokens)
    tot = {0: 0, 1: 0}
    s = {0: defaultdict(float), 1: defaultdict(float)}
    c = {0: defaultdict(int), 1: defaultdict(int)}
    for r in recs:
        d = np.array(r["lp_new"]) - np.array(r["lp_old"])
        lab = r["label"]
        tot[lab] += len(d)
        for t, v in zip(r["ids"], d):
            s[lab][t] += v
            c[lab][t] += 1

    vocab = set(s[0]) | set(s[1])
    contrib = {v: s[1][v] / tot[1] - s[0][v] / tot[0] for v in vocab}
    gap = sum(contrib.values())
    print(f"texts={len(recs)} tokens AI={tot[1]} human={tot[0]}")
    print(f"token-weighted gap mean(d|AI) - mean(d|human) = {gap:+.4f}  (sum of contributions below)")

    def row(v):
        name = tok.convert_ids_to_tokens(int(v))
        ma = s[1][v] / c[1][v] if c[1][v] else float("nan")
        mh = s[0][v] / c[0][v] if c[0][v] else float("nan")
        return (f"{name!r:<18}{contrib[v]:>+9.4f}{c[1][v]:>8}{c[0][v]:>8}"
                f"{c[1][v] / tot[1] * 1000:>8.2f}{c[0][v] / tot[0] * 1000:>8.2f}{ma:>+8.3f}{mh:>+8.3f}")

    head = f"{'token':<18}{'contrib':>9}{'n_AI':>8}{'n_hum':>8}{'AI/1k':>8}{'hum/1k':>8}{'d_AI':>8}{'d_hum':>8}"
    common = [v for v in vocab if c[0][v] + c[1][v] >= args.min_count]
    for title, order in (("tokens pushing Delta up for AI", sorted(common, key=lambda v: -contrib[v])),
                         ("tokens pushing Delta up for human", sorted(common, key=lambda v: contrib[v]))):
        print(f"\n{title} (count >= {args.min_count})\n{head}")
        for v in order[: args.top]:
            print(row(v))

    # how concentrated is the gap? share explained by the top-n tokens by |contribution|
    mags = np.sort(np.abs(np.array(list(contrib.values()))))[::-1]
    for n in (10, 50, 200):
        print(f"top-{n} |contrib| tokens hold {mags[:n].sum() / mags.sum():.1%} of total |contrib|")

    if args.examples and any(r["text"] for r in recs):
        delta = [(float(np.mean(np.array(r["lp_new"]) - np.array(r["lp_old"]))), r) for r in recs]
        delta.sort(key=lambda x: x[0])
        for title, sel in (("highest Delta", delta[::-1][: args.examples]), ("lowest Delta", delta[: args.examples])):
            print(f"\n=== {title}")
            for dv, r in sel:
                lab = "AI" if r["label"] else "human"
                print(f"[{dv:+.3f} {lab} {r['generator']} {r['id']}] {r['text'][:400]}")


if __name__ == "__main__":
    main()
