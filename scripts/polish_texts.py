"""LLM-polished versions of human abstracts (experiment 05).

A human text is revised by a chat model at three intensities (LEVELS: proofread, polish, rewrite).
Two input modes:
  --pairs   02 pair rows (human, same-title AI anchor, generator). Each abstract is polished by the
            model that wrote its anchor, so the generator mix matches the anchors.
  --texts   rows {"id", "text"} without an anchor (e.g. APT-Eval originals). Every generator polishes
            every text.

Output: text rows {"id", "text", "group", "pair_id", "label", "generator", "level", "orig_id",
"edit_sim", ...} for score_texts.py, and <out>.raw_generations.jsonl with the raw outputs. Score with
--no-pair-truncate: polish_calibration.py cuts each comparison (polished vs anchor, polished vs
original) to a common length itself, which is exact for causal LMs.

edit_sim is difflib's ratio between the lower-cased word sequences of the original and the polished
text (1 = unchanged). Decoding uses each model's own generation_config, as for the anchors.

Examples:
  python scripts/polish_texts.py --pairs data/pairs_2022.jsonl --period 22 --out data/polish_2022.jsonl
  python scripts/polish_texts.py --texts data/apt_originals.jsonl --period apt --out data/polish_apt.jsonl
  # re-clean saved generations without the GPU
  python scripts/polish_texts.py --pairs data/pairs_2022.jsonl --period 22 --out data/polish_2022.jsonl \
      --from-raw data/polish_2022.raw_generations.jsonl
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path

from generate_ai import GENERATORS, REFUSAL, generate_prompts

LEVELS = {
    "proof": ("Proofread the following abstract. Correct only errors in grammar, spelling and punctuation, "
              "and keep the original wording wherever possible. Respond with only the corrected abstract.\n\n{text}"),
    "polish": ("Polish the following abstract to improve its clarity, fluency and academic tone, keeping all "
               "of its content. Respond with only the polished abstract.\n\n{text}"),
    "rewrite": ("Rewrite the following abstract in your own words, keeping all of its content. "
                "Respond with only the rewritten abstract.\n\n{text}"),
}

LEAD = re.compile(r"^(here(?:'s| is| are)|below is|sure|certainly|of course|okay|ok)\b", re.I)
HEADER = re.compile(r"^(?:the\s+)?(?:polished|corrected|revised|rewritten|proofread|improved|edited|refined)"
                    r"(?:\s+(?:version|abstract|text))*(?:\s+of\s+the\s+(?:abstract|text))?\s*[:\-]\s*", re.I)
TAIL = re.compile(r"^(?:notes?|changes(?: made)?|corrections(?: made)?|revisions|edits|explanation|summary of"
                  r"|key (?:changes|improvements)|i (?:made|have made|changed|corrected|revised|kept|also|did)"
                  r"|i've|the (?:main |key |following )?(?:changes|corrections|revisions|improvements|edits)"
                  r"|let me know|i hope|feel free|this (?:revised|polished|rewritten|corrected) (?:version|abstract)"
                  r"|(?:main |key )?(?:changes|improvements) (?:made|include))\b", re.I)
BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def clean_polish(raw: str) -> str:
    """Keep the revised text; drop 'Here is the polished abstract:' lines, headers and change notes."""
    body = []
    for line in raw.strip().splitlines():
        plain = re.sub(r"[*#_`]+", "", line).strip()
        if not plain:
            continue
        if not body:
            if LEAD.match(plain) and ":" in plain[:150]:
                plain = plain.split(":", 1)[1].strip()  # 'Here is the polished abstract: <text>'
            plain = HEADER.sub("", plain, count=1)
            plain = re.sub(r"^\s*abstract\s*[:.\-]\s*", "", plain, flags=re.I)
            if not plain or plain.lower().rstrip(":") in {"abstract", "title"}:
                continue
        elif TAIL.match(plain) or BULLET.match(plain):
            break  # commentary after the revised text
        body.append(plain)
    s = re.sub(r"\s+", " ", " ".join(body)).strip()
    s = re.sub(r"\s*\((?:note|i |changes|corrections?)[^)]*\)\s*$", "", s, flags=re.I)
    if len(s) > 1 and s[0] == s[-1] == '"':
        s = s[1:-1].strip()
    return s


def edit_sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower().split(), b.lower().split(), autojunk=False).ratio()


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pairs", help="02 pair rows: polish 'human' with the row's own generator")
    src.add_argument("--texts", help="rows {id, text}: every generator polishes every text")
    ap.add_argument("--period", required=True, help="tag for group names, e.g. 22 -> P22.polish")
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", nargs="+", default=list(LEVELS), choices=list(LEVELS))
    ap.add_argument("--generators", nargs="+", default=list(GENERATORS))
    ap.add_argument("--limit", type=int, default=0, help="first N texts per generator (quick check)")
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--min-tokens", type=int, default=101, help="DatedGPT tokens incl. BOS, as in 02")
    ap.add_argument("--ratio", type=float, nargs=2, default=[0.5, 2.0], help="allowed polished/original word ratio")
    ap.add_argument("--tokenizer", default="datedgpt/datedgpt-2024-base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-raw", default=None, help="reuse generations saved by an earlier run")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.pairs or args.texts).read_text(encoding="utf-8").splitlines() if l.strip()]
    jobs = []  # (row, generator)
    for gi, tag in enumerate(args.generators):
        mine = [r for r in rows if r.get("generator") == tag] if args.pairs else rows
        jobs.extend((r, tag) for r in (mine[: args.limit] if args.limit else mine))
    text_of = (lambda r: r["human"]) if args.pairs else (lambda r: r["text"])

    saved = {}
    if args.from_raw:
        for l in Path(args.from_raw).read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                saved[(r["orig_id"], r["generator"], r["level"])] = r["raw"]

    raws = {}
    for gi, tag in enumerate(args.generators):
        todo = [(r, lv) for r, g in jobs if g == tag for lv in args.levels]
        if args.from_raw:
            for r, lv in todo:
                raws[(str(r["id"]), tag, lv)] = saved[(str(r["id"]), tag, lv)]
            continue
        msgs = [LEVELS[lv].format(text=text_of(r)) for r, lv in todo]
        order = sorted(range(len(msgs)), key=lambda i: len(msgs[i]))  # similar lengths per batch
        outs = generate_prompts(tag, [msgs[i] for i in order], args.max_new_tokens, args.seed + gi)
        for i, o in zip(order, outs):
            r, lv = todo[i]
            raws[(str(r["id"]), tag, lv)] = o

    from transformers import AutoTokenizer
    dtok = AutoTokenizer.from_pretrained(args.tokenizer)
    n_tok = lambda s: len(dtok(s)["input_ids"])

    out_rows, raw_rows, why = [], [], {}
    for r, tag in jobs:
        oid, orig = str(r["id"]), text_of(r)
        for lv in args.levels:
            raw = raws[(oid, tag, lv)]
            p = clean_polish(raw)
            ratio = len(p.split()) / max(1, len(orig.split()))
            reason = ("refusal" if REFUSAL.search(p) else "short" if n_tok(p) < args.min_tokens
                      else "length_ratio" if not args.ratio[0] <= ratio <= args.ratio[1] else "")
            sim = edit_sim(orig, p)
            raw_rows.append({"orig_id": oid, "generator": tag, "level": lv, "raw": raw, "clean": p,
                             "kept": not reason, "why": reason, "edit_sim": round(sim, 4)})
            key = (tag, lv)
            why.setdefault(key, {"kept": 0, "refusal": 0, "short": 0, "length_ratio": 0})
            why[key][reason or "kept"] += 1
            if reason:
                continue
            if args.pairs:
                rid, group, pair_id = f"{oid}~{lv}:p", f"P{args.period}.{lv}", f"{oid}~{lv}"
            else:
                rid, group, pair_id = f"{oid}~{tag}~{lv}", f"{args.period}.{tag}.{lv}", oid
            out_rows.append({"id": rid, "text": p, "group": group, "pair_id": pair_id, "label": 1,
                             "generator": tag, "domain": r.get("domain", ""), "level": lv, "orig_id": oid,
                             "polisher": tag, "edit_sim": round(sim, 4), "word_ratio": round(ratio, 3)})
    for (tag, lv), c in why.items():
        sims = [x["edit_sim"] for x in raw_rows if x["generator"] == tag and x["level"] == lv and x["kept"]]
        print(f"{tag:<24}{lv:<8} {c}  mean edit_sim {sum(sims) / max(1, len(sims)):.3f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows), encoding="utf-8")
    out.with_suffix(".raw_generations.jsonl").write_text(  # with --from-raw: same raw text, fresh 'clean'
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in raw_rows), encoding="utf-8")
    print(f"wrote {len(out_rows)} polished texts to {out}")


if __name__ == "__main__":
    main()
