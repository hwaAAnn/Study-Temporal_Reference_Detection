"""Experiment 12 scoring input: the human abstracts plus every kept generation, as text rows.

Each generation goes through 01's filters (generate_ai.py): clean_generation, the refusal pattern and at least
--min-tokens DatedGPT tokens (101 = 100 scored tokens). A reasoning model's output with no final answer is
dropped as "no_final". Humans keep 08's ids ({pair_id}:h, groups H19 and B); AI rows are {pair_id}:{tag} in
groups N19.<tag> and NB.<tag>. Every generation, with its reason for being dropped, also goes to --report so
the samples can be read.

Example (pod):
  python scripts/build_12_input.py --titles data/12_titles.jsonl --gens outputs/g12_*.jsonl \
      --out data/12_input.jsonl --report outputs/g12_report.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from generate_ai import REFUSAL, clean_generation  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--titles", required=True)
    ap.add_argument("--gens", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--tokenizer", default="datedgpt/datedgpt-2024-base")
    ap.add_argument("--tokenizer-revision", default=None)
    ap.add_argument("--min-tokens", type=int, default=101)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)
    n_tok = lambda s: len(tok(s)["input_ids"])  # noqa: E731

    titles = {}
    rows = []
    for line in Path(args.titles).read_text(encoding="utf-8").splitlines():
        if line.strip():
            t = json.loads(line)
            titles[t["pair_id"]] = t
            rows.append({"id": f"{t['pair_id']}:h", "pair_id": t["pair_id"], "group": t["group"], "label": 0,
                         "text": t["human"], "generator": "", "domain": "arxiv_cs"})
    stats: dict[str, Counter] = {}
    report = []
    for path in args.gens:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            g = json.loads(line)
            t = titles[g["pair_id"]]
            st = stats.setdefault(g["tag"], Counter())
            st["generated"] += 1
            reason, ai = "", None
            if g["final"] is None:
                reason = "no_final"
            else:
                ai = clean_generation(g["final"], t["title"])
                if REFUSAL.search(ai):
                    reason = "refusal"
                elif n_tok(ai) < args.min_tokens:
                    reason = "short"
            st[reason or "kept"] += 1
            report.append({"pair_id": g["pair_id"], "group": g["group"], "tag": g["tag"], "title": t["title"],
                           "finish_reason": g["finish_reason"], "n_out_tokens": g["n_out_tokens"],
                           "dropped": reason, "clean": ai, "raw": g["raw"]})
            if not reason:
                ag = ("N19." if t["group"] == "H19" else "NB.") + g["tag"]
                rows.append({"id": f"{g['pair_id']}:{g['tag']}", "pair_id": g["pair_id"], "group": ag, "label": 1,
                             "text": ai, "generator": g["tag"], "domain": "arxiv_cs"})
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    Path(args.report).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in report), encoding="utf-8")
    for tag, st in sorted(stats.items()):
        print(f"{tag}: {dict(st)}")
    print(f"wrote {len(rows)} rows ({sum(r['label'] == 0 for r in rows)} human) to {args.out}")


if __name__ == "__main__":
    main()
