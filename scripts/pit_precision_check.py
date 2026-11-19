"""Experiment 06: is bfloat16 scoring close enough to float32 for PIT? (decided before any curve is seen)

Compares two score files of the same texts and vintage (run_pit.py --tag _chk32 / _chk16) and prints:
  token   max |lp_bf16 - lp_fp32| over every token
  text    mean and max |mean lp_bf16 - mean lp_fp32| over texts
  gap     bias of the AI - human gap: mean text difference over AI texts minus over human texts
Rule fixed in docs/experiments/06-pit-year-curve.md: bfloat16 is used when, for every checked file pair,
max text difference < 0.02 and |gap bias| < 0.002. Exit code 0 = bfloat16 passes, 3 = use float32.

Example:
  python scripts/pit_precision_check.py --pairs outputs/pit1b_2022_chk32.jsonl:outputs/pit1b_2022_chk16.jsonl \
      outputs/pit4b_2022_chk32.jsonl:outputs/pit4b_2022_chk16.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MAX_TEXT, MAX_GAP = 0.02, 0.002


def load(path: str) -> dict[str, tuple[int, np.ndarray]]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            (lp,) = r["lp"].values()
            out[r["id"]] = (r["label"], np.asarray(lp, dtype=np.float64))
    return out


def compare(ref_path: str, test_path: str) -> dict:
    ref, test = load(ref_path), load(test_path)
    if ref.keys() != test.keys():
        raise SystemExit(f"{ref_path} and {test_path} hold different texts")
    tok, text, lab = [], [], []
    for i, (label, a) in ref.items():
        b = test[i][1]
        if len(a) != len(b):
            raise SystemExit(f"{i}: token counts differ")
        if len(a) == 0:
            continue
        tok.append(np.abs(b - a).max())
        text.append(b.mean() - a.mean())
        lab.append(label)
    text, lab = np.asarray(text), np.asarray(lab)
    gap = float(text[lab == 1].mean() - text[lab == 0].mean())
    return {"ref": ref_path, "test": test_path, "n_texts": int(len(text)),
            "token_max": float(max(tok)), "text_mean_abs": float(np.abs(text).mean()),
            "text_max": float(np.abs(text).max()), "gap_bias": gap,
            "pass": bool(np.abs(text).max() < MAX_TEXT and abs(gap) < MAX_GAP)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True, help="fp32_file:bf16_file")
    ap.add_argument("--json")
    args = ap.parse_args()
    res = [compare(*p.split(":", 1)) for p in args.pairs]
    for r in res:
        print(f"{Path(r['test']).name}: n {r['n_texts']}, token max {r['token_max']:.4f}, "
              f"text mean |d| {r['text_mean_abs']:.5f}, text max {r['text_max']:.5f}, "
              f"gap bias {r['gap_bias']:+.5f} -> {'pass' if r['pass'] else 'FAIL'}")
    ok = all(r["pass"] for r in res)
    print("bfloat16" if ok else "float32")
    if args.json:
        Path(args.json).write_text(json.dumps({"results": res, "use": "bfloat16" if ok else "float32"}, indent=2))
    raise SystemExit(0 if ok else 3)


if __name__ == "__main__":
    main()
