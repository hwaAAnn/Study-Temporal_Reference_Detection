"""Token log-probs under a causal LM, and the temporal-difference score Delta.

Delta(x) = mean over kept tokens of [log p_new(x_t | x_<t) - log p_old(x_t | x_<t)]

Both models must share a tokenizer (true for the DatedGPT family). Models are loaded one at a
time so that two 1.3B models fit on a 6GB GPU.
"""

from __future__ import annotations

import gc
from typing import Iterable

import numpy as np


def token_logprobs(texts: Iterable[str], model_id: str, max_tokens: int = 200,
                   dtype: str = "float16", device: str | None = None, batch_size: int = 4) -> tuple[list[np.ndarray], list[list[int]]]:
    """Return per-text arrays of log p(x_t | x_<t) for t >= 1, plus the token ids used."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype).to(device).eval()

    texts = list(texts)
    ids_all = [tok(t, add_special_tokens=True)["input_ids"][:max_tokens] for t in texts]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(ids_all), batch_size):
            chunk = ids_all[i:i + batch_size]
            L = max(len(c) for c in chunk)
            inp = torch.full((len(chunk), L), pad_id, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for j, c in enumerate(chunk):
                inp[j, :len(c)] = torch.tensor(c)
                att[j, :len(c)] = 1
            inp, att = inp.to(device), att.to(device)
            logits = model(input_ids=inp, attention_mask=att).logits.float()
            lp = torch.log_softmax(logits[:, :-1], dim=-1)
            tgt = inp[:, 1:]
            got = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).cpu().numpy()
            for j, c in enumerate(chunk):
                out.append(got[j, :len(c) - 1].astype(np.float64))

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return out, ids_all


def keep_mask(lp_old: np.ndarray, k: float = 0.0) -> np.ndarray:
    """Mask that drops the top-k fraction of tokens the OLD model is most surprised by.

    k=0 keeps every token. These dropped tokens are treated as 'unknown knowledge'
    (new entities, post-cutoff terms) rather than style.
    """
    n = len(lp_old)
    if n == 0 or k <= 0:
        return np.ones(n, dtype=bool)
    n_drop = int(np.floor(k * n))
    if n_drop == 0:
        return np.ones(n, dtype=bool)
    order = np.argsort(lp_old)  # most negative (most surprising) first
    mask = np.ones(n, dtype=bool)
    mask[order[:n_drop]] = False
    return mask


def delta_score(lp_new: np.ndarray, lp_old: np.ndarray, k: float = 0.0) -> float:
    """Mean of (lp_new - lp_old) over tokens kept by the surprisal filter."""
    if len(lp_new) != len(lp_old):
        raise ValueError("log-prob arrays differ in length; models must share a tokenizer")
    if len(lp_new) == 0:
        return float("nan")
    m = keep_mask(lp_old, k)
    return float(np.mean(lp_new[m] - lp_old[m]))
