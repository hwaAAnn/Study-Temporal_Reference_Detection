"""Group D' (pre-ChatGPT generation AI text): an old base LM writes the abstract for each title.

The model sees the paper title in arXiv LaTeX form and continues it,
    \\title{<title>}\\n\\begin{abstract}\\n
so \\end{abstract} gives a natural stopping point. GPT-Neo-1.3B (2021, trained on the Pile, which
includes arXiv) writes far more abstract-like text than GPT-2, whose WebText has little arXiv.

Decoding is close to group C's defaults (T=0.7, top-p 0.9, repetition penalty 1.1), so C and D'
differ in the model generation rather than in the sampling setup. The same filters as the human
abstracts apply (at most 4 '$', at least --min-tokens DatedGPT tokens).

Output rows (data/extra.jsonl): {"id", "text", "group": "D", "pair_id", "generator", "domain"}

Example:
  python scripts/generate_old_lm.py --pairs data/pairs.jsonl --out data/extra.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROMPTS = {
    "latex": "\\title{{{title}}}\n\\begin{{abstract}}\n",
    "title_abstract": "Title: {title}\n\nAbstract: ",
    "heading": "{title}\n\nAbstract\n\n",
    "pubmed": "{title}\n",
}
STOP = re.compile(r"\\end\{abstract\}|\\section|\\maketitle|\\keywords|\\begin\{|\n\s*\n\s*\n"
                  r"|\n\s*(?:Keywords?|Key words|Index Terms|(?:1\.?|I\.)?\s*Introduction|INTRODUCTION|References"
                  r"|Background|Title:|Abstract:)\b", re.I)


def clean_latex(text: str) -> str:
    """Cut at the end of the abstract and strip LaTeX markup that arXiv's plain-text abstracts lack."""
    s = STOP.split(text, maxsplit=1)[0]
    s = re.sub(r"(?m)%.*$", "", s)                                        # comments
    s = re.sub(r"~?\\(?:cite[tp]?|ref|eqref|label|footnote)\*?\{[^}]*\}", "", s)
    s = re.sub(r"\\(?:emph|textit|textbf|texttt|textsc|mathrm|text)\{([^}]*)\}", r"\1", s)
    s = s.replace("~", " ").replace("``", '"').replace("''", '"')
    return re.sub(r"\s+", " ", s).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/pairs.jsonl", help="rows with id and title (from generate_ai.py)")
    ap.add_argument("--out", default="data/extra.jsonl")
    ap.add_argument("--model", default="EleutherAI/gpt-neo-1.3B")
    ap.add_argument("--tag", default="gpt-neo-1.3b")
    ap.add_argument("--prompt", default="heading", choices=list(PROMPTS))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--min-tokens", type=int, default=101, help="DatedGPT tokens incl. BOS")
    ap.add_argument("--tokenizer", default="datedgpt/datedgpt-2024-base")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(l) for l in Path(args.pairs).read_text(encoding="utf-8").splitlines() if l.strip()]
    seen, papers = set(), []
    for r in rows:
        if r["id"] not in seen:
            seen.add(r["id"])
            papers.append(r)

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16).to("cuda").eval()
    prompts = [PROMPTS[args.prompt].format(title=p["title"]) for p in papers]
    raw = []
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), args.batch_size), desc=args.tag):
            enc = tok(prompts[i:i + args.batch_size], return_tensors="pt", padding=True).to("cuda")
            gen = model.generate(**enc, do_sample=True, temperature=0.7, top_p=0.9, top_k=0,
                                 repetition_penalty=1.1, max_new_tokens=args.max_new_tokens,
                                 pad_token_id=tok.pad_token_id)
            raw.extend(tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True))
    del model
    torch.cuda.empty_cache()

    dtok = AutoTokenizer.from_pretrained(args.tokenizer)
    out_rows, why = [], {"short": 0, "math": 0}
    for p, g in zip(papers, raw):
        text = clean_latex(g)
        if text.count("$") > 4:
            why["math"] += 1
            continue
        if len(dtok(text)["input_ids"]) < args.min_tokens:
            why["short"] += 1
            continue
        out_rows.append({"id": f"{p['id']}:d-{args.tag}", "text": text, "group": "D", "pair_id": str(p["id"]),
                         "generator": args.tag, "domain": p.get("domain", "")})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8")
    out.with_suffix(".raw_generations.jsonl").write_text(
        "".join(json.dumps({"pair_id": str(p["id"]), "raw": g}, ensure_ascii=False) + "\n" for p, g in zip(papers, raw)),
        encoding="utf-8")
    print(f"{args.tag}: generated={len(raw)} kept={len(out_rows)} rejected={why} -> {out}")


if __name__ == "__main__":
    main()
