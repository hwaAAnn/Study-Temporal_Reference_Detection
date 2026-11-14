"""APT-Eval paper abstracts -> score_texts.py rows (experiment 05-b).

APT-Eval (Saha & Feizi, Findings of ACL 2025; MIT licence; huggingface.co/datasets/smksaha/apt-eval)
polished 300 human texts taken from MixSet with five LLMs (GPT-4o, Llama3.1-70B, Llama3-8B, Llama2-7B,
DeepSeek-V3), by degree (extreme_minor, minor, slight_major, major) and by the share of words the
polisher was asked to change (1-75%). 05-b used only the paper_abstract domain (51 originals); 04 uses all six
domains (--domain all) and the degree-based versions (--polish-type degree-based).

A polished row does not store its original's id, so each is matched to the original with the highest
word-sequence similarity (the match score is kept; a weak match is reported, not silently used).

Writes
  <out-dir>/apt_originals.jsonl  {"id": "apt<n>", "text", "group": "APT.H", "label": 0, ...}
  <out-dir>/apt_polished.jsonl   {"id", "text", "group": "APT.<polisher>.<level>", "pair_id": "apt<n>",
                                  "label": 1, "polisher", "polish_type", "level", "orig_id", "edit_sim", ...}
  level is the degree (extreme_minor ... major) or p<percent> (p1 ... p75).

Examples (CSV files downloaded from the dataset repo into their own folder):
  python scripts/convert_apt_eval.py --csv-dir downloads/apt-eval --out-dir data                       # 05-b
  python scripts/convert_apt_eval.py --csv-dir data/raw/apt-eval --out-dir data --domain all       --polish-type degree-based --prefix apt04                                                          # 04
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
from pathlib import Path


def words(s: str) -> list[str]:
    return s.lower().split()


def edit_sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, words(a), words(b), autojunk=False).ratio()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", required=True, help="folder with original.csv and merged_apt_eval_dataset.csv")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--domain", default="paper_abstract", help="one domain, a comma list, or all")
    ap.add_argument("--polish-type", default="", help="keep only this polish_type (degree-based / percentage-based)")
    ap.add_argument("--prefix", default="apt", help="output files <prefix>_originals.jsonl, <prefix>_polished.jsonl")
    ap.add_argument("--min-match", type=float, default=0.3, help="report polished rows matched below this")
    args = ap.parse_args()

    read = lambda name: list(csv.DictReader((Path(args.csv_dir) / name).open(encoding="utf-8", newline="")))
    all_orig, all_pol = read("original.csv"), read("merged_apt_eval_dataset.csv")
    domains = sorted({o["domain"] for o in all_orig}) if args.domain == "all" else args.domain.split(",")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    orig_rows, rows, weak = [], [], 0
    for dom in domains:
        originals = [r for r in all_orig if r["domain"] == dom]
        polished = [r for r in all_pol if r["domain"] == dom
                    and (not args.polish_type or r["polish_type"] == args.polish_type)]
        osets = [set(words(o["generation"])) for o in originals]
        orig_rows += [{"id": f"apt{o['id']}", "text": " ".join(o["generation"].split()), "group": "APT.H", "label": 0,
                       "pair_id": f"apt{o['id']}", "generator": "", "domain": f"apt_{dom}"} for o in originals]
        for p in polished:
            text = " ".join(p["generation"].split())
            pw = set(words(text))
            # shortlist by word-set overlap, then pick the best sequence match
            cand = sorted(range(len(originals)), key=lambda j: -len(pw & osets[j]) / max(1, len(pw | osets[j])))[:3]
            sims = {j: edit_sim(originals[j]["generation"], text) for j in cand}
            j = max(sims, key=sims.get)
            weak += sims[j] < args.min_match
            level = (p["polishing_degree"] if p["polish_type"] == "degree-based"
                     else f"p{int(float(p['polishing_percent']))}")
            oid = f"apt{originals[j]['id']}"
            rows.append({"id": f"{oid}~{p['polisher']}~{level}~{p['id']}", "text": text,
                         "group": f"APT.{p['polisher']}.{level}", "pair_id": oid, "label": 1,
                         "generator": p["polisher"], "polisher": p["polisher"], "domain": f"apt_{dom}",
                         "polish_type": p["polish_type"], "level": level, "orig_id": oid,
                         "edit_sim": round(sims[j], 4), "apt_levenshtein": float(p["levenshtein_distance"] or "nan"),
                         "apt_sem_similarity": float(p["sem_similarity"] or "nan"),
                         "word_ratio": round(len(text.split()) / max(1, len(originals[j]["generation"].split())), 3)})

    (out_dir / f"{args.prefix}_originals.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in orig_rows), encoding="utf-8")
    (out_dir / f"{args.prefix}_polished.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(orig_rows)} originals, {len(rows)} polished texts ({weak} matched below {args.min_match})")


if __name__ == "__main__":
    main()
