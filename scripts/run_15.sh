#!/usr/bin/env bash
# Experiment 15 scoring chain, run from the 15 worktree with the main checkout as $1 (data/ and outputs/ live there).
#   check: 04's RAID abstracts scored again with 2022 and 2024 must match yc04 (check_15.py)
#   score: data/15_input.jsonl with all twelve DatedGPT vintages, one file per vintage (as 01)
# Usage: bash scripts/run_15.sh /c/Users/OFFICE/Study-Temporal_Reference_Detection [check|score]
set -euo pipefail
MAIN="$1"
STEP="${2:-score}"
PY="$MAIN/.venv/Scripts/python.exe"
D="$MAIN/data"
O="$MAIN/outputs"

if [ "$STEP" = "check" ]; then
  "$PY" scripts/check_15.py subset --input "$D/15_input.jsonl" --ref "$O/yc04_2022.jsonl" --out "$D/15_check.jsonl"
  "$PY" scripts/score_texts.py --input "$D/15_check.jsonl" --out "$O/s15_check.jsonl" \
      --models datedgpt/datedgpt-2022-base datedgpt/datedgpt-2024-base --batch-size 8 --no-pair-truncate
  "$PY" scripts/check_15.py compare --new "$O/s15_check.jsonl" --ref "$O/yc04_2022.jsonl" "$O/yc04_2024.jsonl"
  exit 0
fi

for y in 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024; do
  if [ -s "$O/yc15_$y.jsonl" ]; then echo "skip $y (exists)"; continue; fi
  "$PY" scripts/score_texts.py --input "$D/15_input.jsonl" --out "$O/yc15_$y.jsonl" \
      --models datedgpt/datedgpt-$y-base --batch-size 8 --no-pair-truncate
  date
done
echo done
