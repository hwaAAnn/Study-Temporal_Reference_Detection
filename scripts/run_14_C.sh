#!/usr/bin/env bash
# Experiment 14 part C: 09's main pools (RAID.H, RAID chat sampling, APT.H, APT strong) rescored with the five
# 08 vintages (2013 2019 2022 2023 2024 -> 25 cells), so the 2023 reference exists on commercial and large
# generators. Same settings as run_09_slots.sh (200 tokens, chunk 64). Resumable (score_slots.py skips ids
# already written).
# usage (from this worktree): scripts/run_14_C.sh <main checkout> [--limit N --out-name NAME]   # pipeline check
set -eo pipefail
ROOT=$(cd "$1" && pwd -W); shift
HERE=$(cd "$(dirname "$0")/.." && pwd -W)
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1 TQDM_DISABLE=1
PY="$ROOT/.venv/Scripts/python.exe"
IN="$ROOT/data/04_new.jsonl"
NAME=s14_C.jsonl; LIMIT=()
while [ $# -gt 0 ]; do
  case "$1" in --limit) LIMIT=(--limit "$2"); shift 2;; --out-name) NAME="$2"; shift 2;; *) echo "unknown $1"; exit 1;; esac
done
ALL=$(grep -o '"group": "[^"]*"' "$IN" | cut -d'"' -f4 | sort -u)
MAIN=$(echo "$ALL" | grep -E '^(RAID\.H|APT\.H|RAID\.(chatgpt|gpt4|cohere-chat|llama-chat|mistral-chat|mpt-chat)\.sampling|APT\..+\.(slight_major|major))$')
echo "groups: $(echo "$MAIN" | wc -l) main"; date
"$PY" "$HERE/scripts/score_slots.py" --input "$IN" --out "$ROOT/outputs/$NAME" \
    --vintages 2013 2019 2022 2023 2024 --chunk 64 --groups $MAIN "${LIMIT[@]}" 2>&1 | grep -v "CUDACachingAllocator"
date; echo "done"
