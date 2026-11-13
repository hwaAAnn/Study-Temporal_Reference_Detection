"""AI anchors for documents without a title (experiment 02-b): continue the document's opening.

Like Fast-DetectGPT's data builder: the chat model gets the first --prefix-words words and writes an
article starting exactly with them, so the human and AI texts share topic, opening and knowledge era.
Both sides are flattened to one paragraph (whitespace collapsed) so line breaks cannot differ between
them (the pilot's human texts had their line breaks stripped while the AI texts kept theirs).

Example:
  python scripts/generate_continuation.py --human data/fw_2022.jsonl --out data/pairs_fw2022.jsonl \
      --per-generator 200 --human-group W22 --ai-group CW22
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import generate_ai  # scripts/ is on sys.path when this file is run as a script

PROMPT = "Please write an article with about 250 words starting exactly with: {prefix}"


def flat(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", required=True, help="rows with id and text (sample_fineweb.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--generators", nargs="+", default=list(generate_ai.GENERATORS))
    ap.add_argument("--per-generator", type=int, default=200)
    ap.add_argument("--prefix-words", type=int, default=30)
    ap.add_argument("--min-tokens", type=int, default=101, help="DatedGPT tokens incl. BOS")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--tokenizer", default="datedgpt/datedgpt-2024-base")
    ap.add_argument("--human-group", required=True)
    ap.add_argument("--ai-group", required=True)
    ap.add_argument("--domain", default="fineweb_edu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    dtok = AutoTokenizer.from_pretrained(args.tokenizer)
    n_tok = lambda s: len(dtok(s)["input_ids"])

    docs = [json.loads(l) for l in Path(args.human).read_text(encoding="utf-8").splitlines() if l.strip()]
    docs = [dict(d, text=flat(d["text"])) for d in docs]
    docs = [d for d in docs if n_tok(d["text"]) >= args.min_tokens]
    random.Random(args.seed).shuffle(docs)
    if args.per_generator * len(args.generators) > len(docs):
        raise SystemExit(f"need {args.per_generator * len(args.generators)} documents, have {len(docs)}")

    rows, raw_rows = [], []
    for gi, tag in enumerate(args.generators):
        chunk = docs[gi * args.per_generator:(gi + 1) * args.per_generator]
        prefixes = [" ".join(d["text"].split()[: args.prefix_words]) for d in chunk]
        gens = generate_ai.generate_prompts(tag, [PROMPT.format(prefix=p) for p in prefixes],
                                            args.max_new_tokens, args.seed + gi)
        kept, short = 0, 0
        for d, p, g in zip(chunk, prefixes, gens):
            body = flat(generate_ai.clean_generation(g, ""))
            ai = body if body.startswith(p[:20]) else f"{p} {body}"
            raw_rows.append({"id": d["id"], "generator": tag, "raw": g, "clean": ai})
            if n_tok(ai) < args.min_tokens:
                short += 1
                continue
            rows.append({"id": d["id"], "human": d["text"], "ai": ai, "generator": tag, "domain": args.domain,
                         "url": d.get("url"), "dump": d.get("dump"),
                         "human_group": args.human_group, "ai_group": args.ai_group})
            kept += 1
        print(f"{tag}: generated={len(gens)} kept={kept} short={short}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    out.with_suffix(".raw_generations.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in raw_rows), encoding="utf-8")
    print(f"wrote {len(rows)} pairs to {out}")


if __name__ == "__main__":
    main()
