#!/usr/bin/env bash
# Experiment 09 scoring chain: the 08 slots (score_slots.py) on the 04 test texts. Resumable: score_slots.py
# skips ids already written, so rerunning after an interruption continues where it stopped.
#   main groups (RAID.H, RAID chat sampling, APT.H, APT strong): vintages 2018 2021 2022 2024 -> 16 cells
#   other groups: vintages 2022 2024 -> 4 cells
# usage (from this worktree): scripts/run_09_slots.sh <main checkout>   # data/, outputs/ and .venv live there
set -eo pipefail
ROOT=$(cd "$1" && pwd -W)
HERE=$(cd "$(dirname "$0")/.." && pwd -W)   # Windows form, like run_07_tic.sh
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1 TQDM_DISABLE=1
PY="$ROOT/.venv/Scripts/python.exe"
IN="$ROOT/data/04_new.jsonl"
ALL=$(grep -o '"group": "[^"]*"' "$IN" | cut -d'"' -f4 | sort -u)
MAIN=$(echo "$ALL" | grep -E '^(RAID\.H|APT\.H|RAID\.(chatgpt|gpt4|cohere-chat|llama-chat|mistral-chat|mpt-chat)\.sampling|APT\..+\.(slight_major|major))$')
REST=$(echo "$ALL" | grep -vxF "$MAIN")
echo "groups: $(echo "$MAIN" | wc -l) main, $(echo "$REST" | wc -l) other"

date; echo "main: 2018 2021 2022 2024"
"$PY" "$HERE/scripts/score_slots.py" --input "$IN" --out "$ROOT/outputs/s09_main.jsonl" \
    --vintages 2018 2021 2022 2024 --chunk 64 --groups $MAIN 2>&1 | grep -v "CUDACachingAllocator"
date; echo "other: 2022 2024"
"$PY" "$HERE/scripts/score_slots.py" --input "$IN" --out "$ROOT/outputs/s09_rest.jsonl" \
    --vintages 2022 2024 --chunk 128 --groups $REST 2>&1 | grep -v "CUDACachingAllocator"
date; echo "done"
