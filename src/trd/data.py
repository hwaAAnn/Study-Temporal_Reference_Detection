"""Input normalisation and model naming.

Groups (see docs/plan.md):
  A  human, written before 2021 (seen by every vintage >= 2021)
  B  human, written after the newest model's cutoff (2025+), low-AI-risk source
  C  AI, current chat/instruct LLMs, same topics as B
  D  AI, pre-ChatGPT generation LMs (GPT-2, GPT-Neo), same topics as B
"""

from __future__ import annotations

import re

AI_GROUPS = {"C", "D"}
HUMAN_GROUPS = {"A", "B"}


def normalise_rows(rows: list[dict]) -> list[dict]:
    """Accept pair rows {"id","human","ai",...} or text rows {"id","text","group",...}."""
    items = []
    for r in rows:
        meta = {"generator": r.get("generator", ""), "domain": r.get("domain", "")}
        if "human" in r and "ai" in r:
            pid = str(r["id"])
            items.append({"id": f"{pid}:h", "pair_id": pid, "group": r.get("human_group", "B"),
                          "label": 0, "text": r["human"], **meta})
            items.append({"id": f"{pid}:a", "pair_id": pid, "group": r.get("ai_group", "C"),
                          "label": 1, "text": r["ai"], **meta})
        elif "text" in r and "group" in r:
            g = str(r["group"])
            if "label" in r:
                label = int(r["label"])
            elif g in AI_GROUPS:
                label = 1
            elif g in HUMAN_GROUPS:
                label = 0
            else:
                raise ValueError(f"row {r.get('id')}: unknown group {g!r} and no label")
            items.append({"id": str(r["id"]), "pair_id": str(r.get("pair_id", r["id"])),
                          "group": g, "label": label, "text": r["text"], **meta})
        else:
            raise ValueError(f"row {r.get('id')}: need (human, ai) or (text, group)")
    return items


def common_lengths(pair_ids: list[str], lengths: list[int]) -> list[int]:
    """For each item, the shortest length among the items that share its pair_id."""
    shortest: dict[str, int] = {}
    for p, n in zip(pair_ids, lengths):
        shortest[p] = min(shortest.get(p, n), n)
    return [shortest[p] for p in pair_ids]


def model_key(model_id: str) -> str:
    """'datedgpt/datedgpt-2021-base' -> '2021'; falls back to the repo name."""
    m = re.search(r"(19|20)\d{2}", model_id.split("/")[-1])
    return m.group(0) if m else model_id.split("/")[-1]
