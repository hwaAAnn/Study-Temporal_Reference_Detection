"""Score texts under one TiC-LM Oracle model (open_lm); same output format as score_texts.py.

The record's "lp" key is the cutoff ("201305" ... "202407"), so per-cutoff files merge by text id in
tic_curve.py. Length control is the same as score_texts.py (--no-pair-truncate keeps each text's length).

Example (07):
  python scripts/score_tic.py --input data/01_all.jsonl --weights models/tic/tic1b_202301.safetensors \
      --cutoff 202301 --out outputs/tic1b_202301.jsonl --no-pair-truncate --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# run with python -I (it reads downloaded checkpoints), so put this checkout's src/ first explicitly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trd.data import common_lengths, normalise_rows  # noqa: E402
from trd.tic import CUTOFFS, SIZES, token_logprobs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--weights", required=True, help="fp16 .safetensors made by tic_check.py convert (or a .pt)")
    ap.add_argument("--cutoff", required=True, choices=CUTOFFS)
    ap.add_argument("--size", default="1b", choices=sorted(SIZES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None, help="default: cuda if available")
    ap.add_argument("--no-pair-truncate", action="store_true")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.input).read_text(encoding="utf-8").splitlines() if l.strip()]
    items = normalise_rows(rows)
    lp, ids = token_logprobs([it["text"] for it in items], args.weights, args.size, args.max_tokens,
                             args.dtype, device=args.device, batch_size=args.batch_size,
                             desc=f"tic{args.size}_{args.cutoff}")
    n_full = [len(x) - 1 for x in ids]
    n_keep = n_full if args.no_pair_truncate else common_lengths([it["pair_id"] for it in items], n_full)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for i, it in enumerate(items):
            n = n_keep[i]
            rec = {k: it[k] for k in ("id", "pair_id", "group", "label", "generator", "domain")}
            rec.update({"n_tokens": n, "n_tokens_full": n_full[i], "pair_truncate": not args.no_pair_truncate,
                        "dtype": args.dtype, "model": f"tic-oracle-{args.size}-{args.cutoff}", "ids": ids[i][1:n + 1]})
            rec["lp"] = {args.cutoff: lp[i][:n].round(5).tolist()}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(out)
    print(f"wrote {len(items)} records to {out} (TiC-LM {args.size} {args.cutoff}, {args.dtype})")


if __name__ == "__main__":
    main()
