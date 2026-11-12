"""Score texts under one or more time-locked models (one model in memory at a time).

Input: JSONL rows, either
  pairs  {"id", "human", "ai", "generator", "domain"}            -> groups B (human) and C (AI)
  texts  {"id", "text", "group": "A|B|C|D", "pair_id"?, ...}     -> any group
Output: one JSONL line per text with per-token log-probs under every model, keyed by vintage year.

Examples:
  # 00 experiment (2021 vs 2024) + 2018 for the difference-in-differences check
  python scripts/score_texts.py --input data/pairs.jsonl --out outputs/k0.jsonl \
      --models datedgpt/datedgpt-2018-base datedgpt/datedgpt-2021-base datedgpt/datedgpt-2024-base
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trd.data import model_key, normalise_rows
from trd.scoring import token_logprobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"],
                    help="GTX 1060 (Pascal): float16 (verify with check_models.py first). Cloud: bfloat16.")
    ap.add_argument("--batch-size", type=int, default=1, help="start at 1 on a 6GB GPU")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.input).read_text(encoding="utf-8").splitlines() if l.strip()]
    items = normalise_rows(rows)
    texts = [it["text"] for it in items]

    lps, ref_ids, ref_model = {}, None, None
    for mid in args.models:
        key = model_key(mid)
        if key in lps:
            raise SystemExit(f"duplicate model key {key!r}; give models distinct vintages")
        lp, ids = token_logprobs(texts, mid, args.max_tokens, args.dtype, batch_size=args.batch_size)
        if ref_ids is None:
            ref_ids, ref_model = ids, mid
        elif ids != ref_ids:
            raise SystemExit(f"tokenization of {mid} differs from {ref_model}; models must share a tokenizer")
        lps[key] = lp
        print(f"scored {len(texts)} texts with {mid} (key {key})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i, it in enumerate(items):
            rec = {k: it[k] for k in ("id", "pair_id", "group", "label", "generator", "domain")}
            rec["n_tokens"] = int(len(lps[model_key(args.models[0])][i]))
            rec["lp"] = {k: v[i].round(5).tolist() for k, v in lps.items()}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} records to {out} (models: {', '.join(lps)})")


if __name__ == "__main__":
    main()
