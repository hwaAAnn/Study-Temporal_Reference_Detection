"""Write AI abstracts for the titles of human arXiv abstracts and assemble data/pairs.jsonl.

Each generator gets its own disjoint slice of human papers, so no human text is counted twice in
AUROC. Decoding uses each model's own generation_config (as a user would get it). A pair is kept only
when both texts reach --min-tokens DatedGPT tokens; score_pairs.py then cuts both texts of a pair to
the shorter one (at most 200 tokens), so the two sides are always scored on the same number of tokens.

Example:
  python scripts/generate_ai.py --human data/human_arxiv.jsonl --out data/pairs.jsonl --per-generator 200
  # re-assemble pairs from saved generations without running the generators again
  python scripts/generate_ai.py --from-raw data/pairs.raw_generations.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

GENERATORS = {  # tag: (model id, batch size)
    "qwen2.5-1.5b-instruct": ("Qwen/Qwen2.5-1.5B-Instruct", 16),
    "llama3.2-1b-instruct": ("meta-llama/Llama-3.2-1B-Instruct", 16),
}
PROMPT = ('Write the abstract of a computer science research paper titled "{title}". '
          "Respond with only the abstract text, as a single paragraph of about 200 words.")
REFUSAL = re.compile(r"\b(I can(?:'|no)t|I'm sorry|as an AI|I am unable)\b", re.I)


def clean_generation(text: str, title: str) -> str:
    """Strip chat-style wrappers ('Here is the abstract:', 'Abstract:', markdown, title echo)."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    body = []
    for l in lines:
        plain = re.sub(r"[*#_`]+", "", l).strip()
        low = plain.lower().rstrip(":")
        if not body and (low.startswith("here is") or low.startswith("here's") or low in {"abstract", "title"}
                         or low.startswith("title:") or plain.strip('"').lower() == title.lower()):
            continue
        body.append(plain)
    s = " ".join(body)
    s = re.sub(r"^\s*abstract\s*[:.\-]\s*", "", s, flags=re.I)
    return re.sub(r"\s+", " ", s).strip()


def generate(tag: str, titles: list[str], max_new_tokens: int, seed: int) -> list[str]:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id, bs = GENERATORS[tag]
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to("cuda").eval()
    prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(title=t)}],
                                       tokenize=False, add_generation_prompt=True) for t in titles]
    outs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), bs), desc=tag):
            enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
            outs.extend(tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True))
    del model
    torch.cuda.empty_cache()
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", default="data/human_arxiv.jsonl")
    ap.add_argument("--out", default="data/pairs.jsonl")
    ap.add_argument("--generators", nargs="+", default=list(GENERATORS))
    ap.add_argument("--per-generator", type=int, default=200)
    ap.add_argument("--reserve", type=float, default=0.35, help="extra fraction generated to cover rejects")
    ap.add_argument("--min-tokens", type=int, default=101, help="DatedGPT tokens incl. BOS (101 = 100 scored tokens)")
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--tokenizer", default="datedgpt/datedgpt-2024-base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-raw", default=None, help="reuse generations saved by an earlier run (no GPU needed)")
    ap.add_argument("--human-group", default="B", help="group name of the human side (02 uses H19/H22/H24)")
    ap.add_argument("--ai-group", default="C", help="group name of the AI side (02 uses C19/C22/C24)")
    args = ap.parse_args()

    saved = {}
    if args.from_raw:
        for l in Path(args.from_raw).read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                saved[(r["id"], r["generator"])] = r["raw"]

    from transformers import AutoTokenizer
    dtok = AutoTokenizer.from_pretrained(args.tokenizer)
    n_tok = lambda s: len(dtok(s)["input_ids"])

    humans = [json.loads(l) for l in Path(args.human).read_text(encoding="utf-8").splitlines() if l.strip()]
    humans = [h for h in humans if n_tok(h["abstract"]) >= args.min_tokens]
    random.Random(args.seed).shuffle(humans)
    n_slice = int(args.per_generator * (1 + args.reserve))
    if n_slice * len(args.generators) > len(humans):
        raise SystemExit(f"need {n_slice * len(args.generators)} human abstracts, have {len(humans)} long enough")

    out_rows, raw_rows = [], []
    for gi, tag in enumerate(args.generators):
        chunk = humans[gi * n_slice:(gi + 1) * n_slice]
        if args.from_raw:
            missing = [h["id"] for h in chunk if (h["id"], tag) not in saved]
            if missing:
                raise SystemExit(f"{len(missing)} papers of {tag} have no saved generation, e.g. {missing[:3]}")
            gens = [saved[(h["id"], tag)] for h in chunk]
        else:
            gens = generate(tag, [h["title"] for h in chunk], args.max_new_tokens, args.seed + gi)
        kept, why = 0, {"short": 0, "refusal": 0}
        for h, g in zip(chunk, gens):
            ai = clean_generation(g, h["title"])
            raw_rows.append({"id": h["id"], "generator": tag, "raw": g, "clean": ai})
            if REFUSAL.search(ai):
                why["refusal"] += 1
                continue
            if n_tok(ai) < args.min_tokens:
                why["short"] += 1
                continue
            if kept < args.per_generator:
                out_rows.append({"id": h["id"], "human": h["abstract"], "ai": ai, "generator": tag,
                                 "domain": "arxiv_cs", "title": h["title"], "published": h["published"],
                                 "primary_category": h["primary_category"],
                                 "human_group": args.human_group, "ai_group": args.ai_group})
                kept += 1
        print(f"{tag}: generated={len(gens)} kept={kept} rejected={why}", flush=True)
        if kept < args.per_generator:
            print(f"  WARNING: only {kept} pairs for {tag}; raise --reserve or relax --min-tokens")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8")
    out.with_suffix(".raw_generations.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in raw_rows), encoding="utf-8")
    print(f"wrote {len(out_rows)} pairs to {out}")


if __name__ == "__main__":
    main()
