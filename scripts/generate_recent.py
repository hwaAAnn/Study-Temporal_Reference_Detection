"""Experiment 12 generation (pod, vLLM): one abstract per title with a 2025-2026 open-weight generator.

Prompt and post-processing follow generate_ai.py (01): the same PROMPT, each model's own chat template and
default sampling (vLLM reads generation_config.json), seed 0. Reasoning models keep only the final answer
(gpt-oss: the "final" channel; others: the text after the last </think>); outputs with no final answer within
--max-tokens are recorded and later dropped by build_12_input.py. Special tokens are kept in the raw text so
the channels can be split.

Example (pod):
  python scripts/generate_recent.py --tag qwen3i --titles data/12_titles.jsonl --out outputs/g12_qwen3i.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from generate_ai import PROMPT  # noqa: E402

GENERATORS = {  # tag: (repo, revision, kind)
    "gptoss120b": ("openai/gpt-oss-120b", "b5c939de8f754692c1647ca79fbf85e8c1e70f8a", "harmony"),
    "glm47flash": ("zai-org/GLM-4.7-Flash", "7dd20894a642a0aa287e9827cb1a1f7f91386b67", "think"),
    "qwen3i": ("Qwen/Qwen3-30B-A3B-Instruct-2507", "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe", "chat"),
    "qwen3t": ("Qwen/Qwen3-30B-A3B-Thinking-2507", "144afc2f379b542fdd4e85a1fcd5e1f79112d95d", "think"),
}
SPECIAL = re.compile(r"<\|[^|<>]{1,40}\|>")


def final_text(raw: str, kind: str) -> str | None:
    """The answer a user would see, or None when the model never reached it."""
    if kind == "harmony":
        if "<|channel|>final<|message|>" not in raw:
            return None
        t = raw.split("<|channel|>final<|message|>")[-1]
    elif "</think>" in raw:
        t = raw.split("</think>")[-1]
    elif kind == "think":
        return None
    else:
        t = raw
    t = SPECIAL.sub(" ", t).replace("<think>", " ").replace("</think>", " ")
    return t.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=sorted(GENERATORS))
    ap.add_argument("--titles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    args = ap.parse_args()

    from vllm import LLM

    repo, rev, kind = GENERATORS[args.tag]
    rows = [json.loads(x) for x in Path(args.titles).read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit:
        rows = rows[:args.limit]
    t0 = time.time()
    # download the pinned revision completely first (vLLM's own download left a shard missing once, 11/16),
    # without gpt-oss's original/ and metal/ copies
    from huggingface_hub import snapshot_download
    local = snapshot_download(repo, revision=rev, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt",
                                                                  "*.model", "*.tiktoken"],
                              ignore_patterns=["original/*", "metal/*", "coreml/*"])
    print(f"{args.tag}: downloaded {repo}@{rev[:12]} in {time.time() - t0:.0f} s", flush=True)
    llm = LLM(model=local, seed=args.seed, trust_remote_code=False,
              max_model_len=args.max_tokens + 512, gpu_memory_utilization=args.gpu_mem)
    sp = llm.get_default_sampling_params()
    sp.max_tokens = args.max_tokens
    sp.skip_special_tokens = False
    print(f"{args.tag}: loaded in {time.time() - t0:.0f} s; sampling {sp}", flush=True)
    msgs = [[{"role": "user", "content": PROMPT.format(title=r["title"])}] for r in rows]
    t1 = time.time()
    outs = llm.chat(msgs, sp, use_tqdm=False)   # vLLM 0.33 divides by tqdm's elapsed time, 0 under TQDM_DISABLE
    print(f"{args.tag}: {len(outs)} generations in {time.time() - t1:.0f} s", flush=True)
    n_final = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            c = o.outputs[0]
            fin = final_text(c.text, kind)
            n_final += fin is not None
            f.write(json.dumps({"pair_id": r["pair_id"], "group": r["group"], "tag": args.tag, "repo": repo,
                                "revision": rev, "raw": c.text, "final": fin, "finish_reason": c.finish_reason,
                                "n_out_tokens": len(c.token_ids)}, ensure_ascii=False) + "\n")
    print(f"{args.tag}: {n_final}/{len(rows)} with a final answer; wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
