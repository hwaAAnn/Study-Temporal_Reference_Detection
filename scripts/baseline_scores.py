"""Zero-shot baseline detectors for experiment 04, sized for a 6 GB GPU (fp16, one or two small models).

Binoculars (Hans et al., ICML 2024): B(x) = log-PPL_performer(x) / X-PPL(x), where
  log-PPL_performer = mean_t -log q(x_t | x_<t),  X-PPL = mean_t sum_v p(v | x_<t) * -log q(v | x_<t),
  p = observer (base model), q = performer (instruct model). Lower = more AI-like.
  The official code averages X-PPL over every non-pad position, one more than the log-PPL; here both
  average over the same predicted positions t = 1..n-1 (difference: one position).
Fast-DetectGPT, analytic sampling discrepancy (Bao et al., ICLR 2024):
  D(x) = (sum_t log s(x_t) - sum_t mu_t) / sqrt(sum_t var_t), mu_t = sum_v r(v) log s(v),
  var_t = sum_v r(v) log s(v)^2 - mu_t^2; r = reference (sampling) model, s = scoring model. Higher = more AI.

Tokenization is that of score_texts.py: the tokenizer's own special tokens, first --max-tokens tokens.
Output, one line per text: {"id", <method fields>, "n_tokens"}.
  binoculars:    "binoculars", "ppl" (performer mean NLL), "xppl"
  fastdetectgpt: "fdgpt", "ll" (scoring model mean log p)

Examples:
  python scripts/baseline_scores.py --input data/04_all.jsonl --out outputs/b04_binoculars.jsonl \
      --method binoculars --observer meta-llama/Llama-3.2-1B --performer meta-llama/Llama-3.2-1B-Instruct
  python scripts/baseline_scores.py --input data/04_all.jsonl --out outputs/b04_fdgpt.jsonl \
      --method fastdetectgpt --scorer EleutherAI/gpt-neo-2.7B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def binoculars_terms(obs_logits, perf_logits, ids, mask):
    """Per-sequence (performer mean NLL, cross-perplexity) from logits (B, L, V), ids and mask (B, L)."""
    import torch
    m = mask[:, 1:].float()
    tgt = ids[:, 1:]
    q_lp = torch.log_softmax(perf_logits[:, :-1].float(), dim=-1)
    p = torch.softmax(obs_logits[:, :-1].float(), dim=-1)
    nll = -q_lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    xent = -(p * q_lp).sum(-1)
    den = m.sum(1)
    return (nll * m).sum(1) / den, (xent * m).sum(1) / den


def fdgpt_terms(ref_logits, score_logits, ids, mask):
    """Per-sequence (analytic sampling discrepancy, mean log p under the scoring model)."""
    import torch
    m = mask[:, 1:].float()
    tgt = ids[:, 1:]
    s_lp = torch.log_softmax(score_logits[:, :-1].float(), dim=-1)
    r = torch.softmax(ref_logits[:, :-1].float(), dim=-1)
    ll = s_lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    mu = (r * s_lp).sum(-1)
    var = (r * s_lp.square()).sum(-1) - mu.square()
    disc = ((ll - mu) * m).sum(1) / ((var * m).sum(1)).sqrt()
    return disc, (ll * m).sum(1) / m.sum(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["binoculars", "fastdetectgpt"], required=True)
    ap.add_argument("--observer", help="binoculars: base model")
    ap.add_argument("--performer", help="binoculars: instruct model (same tokenizer)")
    ap.add_argument("--scorer", help="fastdetectgpt: scoring model")
    ap.add_argument("--reference", help="fastdetectgpt: sampling model (default: the scoring model)")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from trd.data import normalise_rows

    rows_in = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    items = normalise_rows(rows_in)  # same rows and ids as score_texts.py
    if args.method == "binoculars":
        if not (args.observer and args.performer):
            raise SystemExit("binoculars needs --observer and --performer")
        names = [args.observer, args.performer]
    else:
        if not args.scorer:
            raise SystemExit("fastdetectgpt needs --scorer")
        names = [args.reference or args.scorer, args.scorer]
    tok = AutoTokenizer.from_pretrained(names[1])
    if AutoTokenizer.from_pretrained(names[0]).get_vocab() != tok.get_vocab():
        raise SystemExit(f"{names[0]} and {names[1]} do not share a vocabulary")
    load = lambda n: AutoModelForCausalLM.from_pretrained(n, dtype=torch.float16).to("cuda").eval()  # noqa: E731
    models = [load(names[0])] if names[0] == names[1] else [load(names[0]), load(names[1])]

    ids_all = [tok(it["text"], add_special_tokens=True)["input_ids"][:args.max_tokens] for it in items]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    order = sorted(range(len(items)), key=lambda i: len(ids_all[i]))
    out: list[dict | None] = [None] * len(items)
    with torch.no_grad():
        for b in tqdm(range(0, len(order), args.batch_size), desc=args.method, leave=False):
            idx = order[b:b + args.batch_size]
            L = max(len(ids_all[i]) for i in idx)
            ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
            mask = torch.zeros((len(idx), L), dtype=torch.long)
            for j, i in enumerate(idx):
                ids[j, :len(ids_all[i])] = torch.tensor(ids_all[i])
                mask[j, :len(ids_all[i])] = 1
            ids, mask = ids.cuda(), mask.cuda()
            logits = [mdl(input_ids=ids, attention_mask=mask).logits for mdl in models]
            first, second = (logits[0], logits[0]) if len(logits) == 1 else logits
            for lg in logits:
                if not torch.isfinite(lg[mask.bool()]).all():
                    raise FloatingPointError("non-finite logits in fp16")
            if args.method == "binoculars":
                ppl, xppl = binoculars_terms(first, second, ids, mask)
                rows = [{"binoculars": float(a / x), "ppl": float(a), "xppl": float(x)} for a, x in zip(ppl, xppl)]
            else:
                disc, ll = fdgpt_terms(first, second, ids, mask)
                rows = [{"fdgpt": float(d), "ll": float(v)} for d, v in zip(disc, ll)]
            for j, i in enumerate(idx):
                out[i] = {"id": items[i]["id"], **rows[j], "n_tokens": len(ids_all[i])}
            del logits, first, second
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8")
    print(f"wrote {len(out)} {args.method} scores to {args.out} (models: {', '.join(dict.fromkeys(names))})")


if __name__ == "__main__":
    main()
