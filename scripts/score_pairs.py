"""Score human/AI pairs with an old and a new time-locked model.

Writes one JSON line per text with token log-probs summarised, so evaluate.py (and the
k-sweep later) can run without reloading models.

Example:
  python scripts/score_pairs.py --pairs data/pairs.jsonl \
      --old datedgpt/datedgpt-2021-base --new datedgpt/datedgpt-2024-base --out outputs/k0.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trd.scoring import token_logprobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"],
                    help="GTX 1060 (Pascal): float16. Cloud (Ampere+): bfloat16.")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--no-pair-truncate", action="store_true",
                    help="keep each text's own length instead of cutting both texts of a pair to the shorter one")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.pairs).read_text(encoding="utf-8").splitlines() if l.strip()]
    items = []
    for r in rows:
        for label, key in ((0, "human"), (1, "ai")):
            items.append({"id": r["id"], "label": label, "generator": r.get("generator", ""),
                          "domain": r.get("domain", ""), "text": r[key]})
    texts = [it["text"] for it in items]

    lp_old, ids_old = token_logprobs(texts, args.old, args.max_tokens, args.dtype, batch_size=args.batch_size)
    lp_new, ids_new = token_logprobs(texts, args.new, args.max_tokens, args.dtype, batch_size=args.batch_size)
    if ids_old != ids_new:
        raise SystemExit("Tokenizations differ between models; pick a pair that shares a tokenizer.")

    if not args.no_pair_truncate:
        # Length control: both texts of a pair keep only as many tokens as the shorter one. The models are
        # causal, so the log-probs of the first L tokens are exactly those of the text cut to L tokens.
        n_pair = {}
        for it, lp in zip(items, lp_old):
            key = (it["id"], it["generator"])
            n_pair[key] = min(n_pair.get(key, len(lp)), len(lp))
        for i, it in enumerate(items):
            n = n_pair[(it["id"], it["generator"])]
            lp_old[i], lp_new[i], ids_old[i] = lp_old[i][:n], lp_new[i][:n], ids_old[i][:n + 1]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for it, a, b, ids in zip(items, lp_old, lp_new, ids_old):
            rec = {k: it[k] for k in ("id", "label", "generator", "domain")}
            rec.update({"old_model": args.old, "new_model": args.new, "dtype": args.dtype,
                        "pair_truncate": not args.no_pair_truncate,
                        "ids": ids[1:],  # target tokens, aligned with lp_*
                        "lp_old": a.round(5).tolist(), "lp_new": b.round(5).tolist()})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} scored texts to {out}")


if __name__ == "__main__":
    main()
