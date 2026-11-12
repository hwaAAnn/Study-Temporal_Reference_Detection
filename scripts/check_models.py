"""Step 0 of experiment 00: check the model family before trusting any Delta.

  1. tokenizer identity  every vintage encodes the sample texts to identical ids and has the
                         same vocabulary
  2. fp16 numerics       on the GPU in float16, log-probs are finite and close to a CPU float32
                         reference (DatedGPT ships bf16; GTX 1060 has no native bf16)

Example:
  python scripts/check_models.py --models datedgpt/datedgpt-2018-base datedgpt/datedgpt-2021-base \
      datedgpt/datedgpt-2024-base --fp-check datedgpt/datedgpt-2021-base
"""

from __future__ import annotations

import argparse

import numpy as np

from trd.scoring import token_logprobs

SAMPLES = [
    "The committee met on Tuesday to discuss the proposed changes to the city budget.",
    "In this paper, we propose a simple yet effective method for improving robustness.",
    "Honestly I didn't expect the ending, but the second half dragged a bit lol.",
]


def check_tokenizers(models: list[str]) -> bool:
    from transformers import AutoTokenizer

    ref_vocab, ref_ids, ok = None, None, True
    for m in models:
        tok = AutoTokenizer.from_pretrained(m)
        vocab = tok.get_vocab()
        ids = [tok(s)["input_ids"] for s in SAMPLES]
        if ref_vocab is None:
            ref_vocab, ref_ids = vocab, ids
            print(f"[tok] {m}: vocab {len(vocab)} (reference)")
            continue
        same_vocab, same_ids = vocab == ref_vocab, ids == ref_ids
        ok &= same_vocab and same_ids
        print(f"[tok] {m}: vocab {len(vocab)}  same_vocab={same_vocab}  same_ids={same_ids}")
    return ok


def check_fp16(model: str, tol: float = 0.05) -> bool:
    lp16, _ = token_logprobs(SAMPLES, model, dtype="float16", device="cuda", batch_size=1)
    lp32, _ = token_logprobs(SAMPLES, model, dtype="float32", device="cpu", batch_size=1)
    ok = True
    for i, (a, b) in enumerate(zip(lp16, lp32)):
        finite = bool(np.isfinite(a).all())
        diff = float(np.abs(a - b).max())
        mean_diff = float(abs(a.mean() - b.mean()))
        ok &= finite and mean_diff < tol
        print(f"[fp16] sample {i}: finite={finite}  max|Δtoken|={diff:.4f}  |Δmean|={mean_diff:.4f}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--fp-check", default=None, help="one model to compare fp16 (GPU) vs fp32 (CPU)")
    args = ap.parse_args()

    tok_ok = check_tokenizers(args.models)
    fp_ok = check_fp16(args.fp_check) if args.fp_check else True
    print(f"\ntokenizers identical: {tok_ok}   fp16 usable: {fp_ok}")
    if not (tok_ok and fp_ok):
        raise SystemExit("check failed: see docs/experiments/00-k0-signal.md step 0")


if __name__ == "__main__":
    main()
