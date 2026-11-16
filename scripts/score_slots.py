"""Experiment 08: Fast-DetectGPT and Binoculars for every (reference r, scorer s) pair of DatedGPT vintages.

Both detectors set how predictable the text is under the scoring model s against how predictable s finds the
next-token distribution of a reference model r (the same functions as baseline_scores.py, experiment 04):
  Fast-DetectGPT  D(x) = sum_t [log s(x_t) - mu_t] / sqrt(sum_t var_t),   mu_t = sum_v r(v) log s(v)
                  (r = sampling model, s = scoring model; higher = more AI)
  Binoculars      B(x) = mean_t -log s(x_t) / mean_t sum_v r(v) (-log s(v))
                  (r = observer, s = performer; lower = more AI)
Cell "r_s" = reference vintage r, scoring vintage s. The diagonal r = s is the usual single-model setting.

Two 1.35B models together nearly fill the 6 GB GPU and spill into shared memory (about 1 s per text), so texts
are taken in chunks: each vintage in turn scores the chunk, its fp16 logits stay in RAM, and every r x s cell
is then computed on the GPU from the kept logits. The vintage order alternates between chunks so the last
model of one chunk is reused for the next. Tokenization is that of score_texts.py: the tokenizer's own special
tokens, first --max-tokens tokens, no truncation to the pair's shorter length (as in 04 and 07).

Output, one line per text, appended per chunk (a rerun skips ids already written):
  {"id", "pair_id", "group", "label", "generator", "n_tokens",
   "cells": {"r_s": {"fdgpt", "ll", "bino", "ppl", "xppl"}}}      ll = mean log s(x_t), ppl = -ll

Example:
  python scripts/score_slots.py --input data/01_all.jsonl --out outputs/s08_grid.jsonl
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

VINTAGES = ["2013", "2019", "2022", "2023", "2024"]
GROUPS = ["H19", "C19", "H24", "C24", "B", "C"]


def model_id(v: str) -> str:
    return f"datedgpt/datedgpt-{v}-base"


def forward_chunk(model, seqs: list[list[int]], pad_id: int, batch_size: int) -> list:
    """fp16 logits (L_i, V) on the CPU for each sequence, in input order."""
    import torch
    out = [None] * len(seqs)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    with torch.no_grad():
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            L = max(len(seqs[i]) for i in idx)
            ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
            mask = torch.zeros((len(idx), L), dtype=torch.long)
            for j, i in enumerate(idx):
                ids[j, :len(seqs[i])] = torch.tensor(seqs[i])
                mask[j, :len(seqs[i])] = 1
            logits = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).logits
            if not torch.isfinite(logits[mask.cuda().bool()]).all():
                raise FloatingPointError("non-finite logits in fp16")
            for j, i in enumerate(idx):
                out[i] = logits[j, :len(seqs[i])].to(torch.float16).cpu()
            del logits
    return out


def cell_terms(lg_r, lg_s, ids, mask) -> dict:
    """Both detectors for one (reference, scorer) pair of logits (1, L, V)."""
    from baseline_scores import binoculars_terms, fdgpt_terms
    disc, ll = fdgpt_terms(lg_r, lg_s, ids, mask)
    ppl, xppl = binoculars_terms(lg_r, lg_s, ids, mask)
    return {"fdgpt": float(disc[0]), "ll": float(ll[0]), "bino": float(ppl[0] / xppl[0]),
            "ppl": float(ppl[0]), "xppl": float(xppl[0])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", nargs="+", default=GROUPS)
    ap.add_argument("--vintages", nargs="+", default=VINTAGES)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=64, help="texts whose logits for every vintage are kept in RAM")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N texts only (pipeline check)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from trd.data import normalise_rows

    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    keep = set(args.groups)
    items = [it for it in normalise_rows(rows) if it["group"] in keep]
    if args.limit:
        items = items[:args.limit]
    tok = AutoTokenizer.from_pretrained(model_id(args.vintages[-1]))
    for v in args.vintages[:-1]:
        if AutoTokenizer.from_pretrained(model_id(v)).get_vocab() != tok.get_vocab():
            raise SystemExit(f"{model_id(v)} does not share the vocabulary")
    ids_all = [tok(it["text"], add_special_tokens=True)["input_ids"][:args.max_tokens] for it in items]
    if min(len(x) for x in ids_all) < 2:
        raise SystemExit("a text has fewer than 2 tokens")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)

    out = Path(args.out)
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [i for i in range(len(items)) if items[i]["id"] not in done]
    print(f"{len(items)} texts, {len(done)} already scored, {len(todo)} to go; vintages {args.vintages}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)

    current, model = None, None
    t0 = time.time()
    for c, c0 in enumerate(range(0, len(todo), args.chunk)):
        chunk = todo[c0:c0 + args.chunk]
        seqs = [ids_all[i] for i in chunk]
        order = args.vintages if c % 2 == 0 else args.vintages[::-1]
        logits = {}
        for v in order:
            if current != v:
                del model
                gc.collect()
                torch.cuda.empty_cache()
                model = AutoModelForCausalLM.from_pretrained(model_id(v), dtype=torch.float16).to("cuda").eval()
                current = v
            logits[v] = forward_chunk(model, seqs, pad_id, args.batch_size)
        lines = []
        with torch.no_grad():
            for k, i in enumerate(chunk):
                ids = torch.tensor(ids_all[i]).unsqueeze(0).cuda()
                mask = torch.ones_like(ids)
                lg = {v: logits[v][k].cuda().unsqueeze(0) for v in args.vintages}
                cells = {f"{r}_{s}": cell_terms(lg[r], lg[s], ids, mask) for r in args.vintages for s in args.vintages}
                it = items[i]
                lines.append(json.dumps({"id": it["id"], "pair_id": it["pair_id"], "group": it["group"],
                                         "label": it["label"], "generator": it["generator"],
                                         "n_tokens": len(ids_all[i]), "cells": cells}))
                del lg
        del logits
        with out.open("a", encoding="utf-8") as f:
            f.write("".join(line + "\n" for line in lines))
        n = min(c0 + args.chunk, len(todo))
        el = time.time() - t0
        print(f"chunk {c + 1}: {n}/{len(todo)} texts, {el / 60:.1f} min, eta {el / n * (len(todo) - n) / 60:.0f} min",
              flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
