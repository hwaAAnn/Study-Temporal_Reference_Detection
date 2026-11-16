"""Experiments 10 and 11: the 08 slot grid (score_slots.py) for model families other than DatedGPT.

  --family tic   TiC-LM Oracle checkpoints (open_lm, trd.tic), fp16 safetensors made as in 07
                 (models/tic/tic1b_<cutoff>.safetensors); a model label is a cutoff such as 202301.
                 Tokenization is 07's: <|endoftext|> (id 0) first, at most --max-tokens ids including it.
  --family hf    Hugging Face causal LMs, each given as label=repo@revision (e.g. base=tiiuae/falcon-7b@ec89...),
                 loaded with the library's own model class (no remote code) in --dtype. Tokenization is the
                 tokenizer's default (add_special_tokens=True), the first --max-tokens ids.

The computation and the output are those of score_slots.py (cell_terms is imported from it): texts go in
chunks; each model in turn scores the chunk and its logits stay in RAM as fp16; every r x s cell is then
computed on the GPU. The model order alternates between chunks so one model is reused across chunks. A cell
"r_s" has reference r and scorer s. A rerun appends to --out and skips ids already there.

Examples:
  python scripts/score_slots_family.py --family tic --models 201305 201901 202101 202301 202407 \
      --input data/01_all.jsonl --groups H19 C19 H24 C24 B C --out outputs/s10_A.jsonl --chunk 32
  python scripts/score_slots_family.py --family hf --dtype bfloat16 \
      --models base=tiiuae/falcon-7b@ec89142b67d748a1865ea4451372db8313ada0d8 \
               inst=tiiuae/falcon-7b-instruct@8782b5c5d8c9290412416618f36a133653e85285 \
      --input data/11_input.jsonl --out outputs/s11_grid.jsonl --chunk 128 --batch-size 8
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

DTYPES = ("float16", "bfloat16", "float32")


class TicFamily:
    """TiC-LM Oracle 1B (07). Right-padded batches need no attention mask: attention is causal and rotary
    positions start at 0 in every row (trd.tic.token_logprobs)."""

    def __init__(self, labels: list[str], weights_dir: str, dtype: str = "float16"):
        from trd.tic import CUTOFFS, EOT
        bad = [c for c in labels if c not in CUTOFFS]
        if bad:
            raise SystemExit(f"unknown TiC cutoffs {bad}")
        self.labels, self.dir, self.dtype, self.pad_id = labels, Path(weights_dir), dtype, EOT
        for c in labels:
            if not self.path(c).exists():
                raise SystemExit(f"missing {self.path(c)} (made by tic_check.py convert)")

    def path(self, c: str) -> Path:
        return self.dir / f"tic1b_{c}.safetensors"

    def encode(self, texts: list[str], max_tokens: int) -> list[list[int]]:
        from trd.tic import encode
        return encode(texts, max_tokens)

    def load(self, label: str):
        from trd.tic import load_model
        return load_model(str(self.path(label)), "1b", self.dtype, "cuda")

    @staticmethod
    def logits(model, ids, mask):
        return model(input_ids=ids)[0]


class HfFamily:
    def __init__(self, specs: list[str], dtype: str):
        from transformers import AutoTokenizer
        self.repo, self.rev, self.labels = {}, {}, []
        for spec in specs:
            label, _, ref = spec.partition("=")
            repo, _, rev = ref.partition("@")
            if not (label and repo and rev):
                raise SystemExit(f"--models entries are label=repo@revision, got {spec!r}")
            self.labels.append(label)
            self.repo[label], self.rev[label] = repo, rev
        self.dtype = dtype
        first = self.labels[0]
        self.tok = AutoTokenizer.from_pretrained(self.repo[first], revision=self.rev[first])
        for lab in self.labels[1:]:
            other = AutoTokenizer.from_pretrained(self.repo[lab], revision=self.rev[lab])
            if other.get_vocab() != self.tok.get_vocab():
                raise SystemExit(f"{self.repo[lab]} does not share the vocabulary of {self.repo[first]}")
        t = self.tok
        self.pad_id = t.pad_token_id if t.pad_token_id is not None else (t.eos_token_id or 0)

    def encode(self, texts: list[str], max_tokens: int) -> list[list[int]]:
        return [self.tok(x, add_special_tokens=True)["input_ids"][:max_tokens] for x in texts]

    def load(self, label: str):
        import torch
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            self.repo[label], revision=self.rev[label], dtype=getattr(torch, self.dtype),
            trust_remote_code=False).to("cuda").eval()

    @staticmethod
    def logits(model, ids, mask):
        return model(input_ids=ids, attention_mask=mask).logits


def make_family(args):
    if args.family == "tic":
        return TicFamily(args.models, args.weights_dir, args.dtype)
    return HfFamily(args.models, args.dtype)


def forward_chunk(fam, model, seqs: list[list[int]], batch_size: int) -> list:
    """fp16 logits (L_i, V) on the CPU for each sequence, in input order (score_slots.forward_chunk with the
    family's forward call)."""
    import torch
    out = [None] * len(seqs)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    with torch.no_grad():
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            L = max(len(seqs[i]) for i in idx)
            ids = torch.full((len(idx), L), fam.pad_id, dtype=torch.long)
            mask = torch.zeros((len(idx), L), dtype=torch.long)
            for j, i in enumerate(idx):
                ids[j, :len(seqs[i])] = torch.tensor(seqs[i])
                mask[j, :len(seqs[i])] = 1
            logits = fam.logits(model, ids.cuda(), mask.cuda())
            real = mask.cuda().bool()
            if not torch.isfinite(logits[real]).all():
                raise FloatingPointError("non-finite logits")
            for j, i in enumerate(idx):
                h = logits[j, :len(seqs[i])].to(torch.float16)
                if not torch.isfinite(h).all():
                    raise FloatingPointError("logits overflow fp16")
                out[i] = h.cpu()
            del logits
    return out


def score(fam, items: list[dict], ids_all: list[list[int]], out: Path, chunk: int, batch_size: int) -> None:
    import torch
    from score_slots import cell_terms

    labels = fam.labels
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [i for i in range(len(items)) if items[i]["id"] not in done]
    print(f"{len(items)} texts, {len(done)} already scored, {len(todo)} to go; models {labels}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    current, model = None, None
    t0 = time.time()
    for c, c0 in enumerate(range(0, len(todo), chunk)):
        part = todo[c0:c0 + chunk]
        seqs = [ids_all[i] for i in part]
        order = labels if c % 2 == 0 else labels[::-1]
        logits = {}
        for lab in order:
            if current != lab:
                del model
                gc.collect()
                torch.cuda.empty_cache()
                model = fam.load(lab)
                current = lab
            logits[lab] = forward_chunk(fam, model, seqs, batch_size)
        lines = []
        with torch.no_grad():
            for k, i in enumerate(part):
                ids = torch.tensor(ids_all[i]).unsqueeze(0).cuda()
                mask = torch.ones_like(ids)
                lg = {lab: logits[lab][k].cuda().unsqueeze(0) for lab in labels}
                cells = {f"{r}_{s}": cell_terms(lg[r], lg[s], ids, mask) for r in labels for s in labels}
                it = items[i]
                lines.append(json.dumps({"id": it["id"], "pair_id": it["pair_id"], "group": it["group"],
                                         "label": it["label"], "generator": it["generator"],
                                         "n_tokens": len(ids_all[i]), "cells": cells}))
                del lg
        del logits
        with out.open("a", encoding="utf-8") as f:
            f.write("".join(line + "\n" for line in lines))
        n = min(c0 + chunk, len(todo))
        el = time.time() - t0
        print(f"chunk {c + 1}: {n}/{len(todo)} texts, {el / 60:.1f} min, eta {el / n * (len(todo) - n) / 60:.0f} min",
              flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"wrote {out}", flush=True)


def load_items(path: str, groups: list[str] | None, limit: int) -> list[dict]:
    from trd.data import normalise_rows
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    items = normalise_rows(rows)
    if groups:
        keep = set(groups)
        items = [it for it in items if it["group"] in keep]
    return items[:limit] if limit else items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["tic", "hf"])
    ap.add_argument("--models", nargs="+", required=True, help="tic: cutoffs; hf: label=repo@revision")
    ap.add_argument("--weights-dir", default="models/tic", help="tic only")
    ap.add_argument("--dtype", default="float16", choices=DTYPES)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--groups", nargs="+", default=None, help="default: every group in --input")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=32, help="texts whose logits for every model are kept in RAM")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N texts only (pipeline check)")
    args = ap.parse_args()

    fam = make_family(args)
    items = load_items(args.input, args.groups, args.limit)
    ids_all = fam.encode([it["text"] for it in items], args.max_tokens)
    if min(len(x) for x in ids_all) < 2:
        raise SystemExit("a text has fewer than 2 tokens")
    score(fam, items, ids_all, Path(args.out), args.chunk, args.batch_size)


if __name__ == "__main__":
    main()
