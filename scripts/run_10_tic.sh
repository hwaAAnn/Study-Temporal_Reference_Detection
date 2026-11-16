#!/usr/bin/env bash
# Experiment 10 scoring chain: 08's slot grid on TiC-LM Oracle 1B, local GPU (fp16), resumable.
#   1) the five checkpoints: download .pt -> fp16 safetensors (07's tic_check.py convert) -> SHA-256 must equal
#      07's outputs/tic_checkpoints.tsv -> delete .pt (a safetensors file already present is reused)
#   2) pipeline check on 12 texts (slots_check_family.py); a failure stops the chain before any grid score
#   3) part A: 08's 2,398 abstracts, five cutoffs, 25 cells      -> outputs/s10_A.jsonl
#   4) part B: 09's main pools (5,596 texts), three cutoffs, 9 cells -> outputs/s10_B.jsonl
#   5) delete the safetensors copies
# usage (from this worktree): bash scripts/run_10_tic.sh /c/Users/OFFICE/Study-Temporal_Reference_Detection
set -eo pipefail
MAIN=${1:?path to the main checkout (data/, outputs/, models/, .venv/)}
ROOT=$(cd "$MAIN" && pwd -W)
HERE=$(cd "$(dirname "$0")/.." && pwd -W)   # Windows form (python -c strings are not path-converted)
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1 TQDM_DISABLE=1
PY="$ROOT/.venv/Scripts/python.exe"
OUT="$ROOT/outputs"; MOD="$ROOT/models/tic"; mkdir -p "$OUT" "$MOD"
CUTS="201305 201901 202101 202301 202407"
A_IN="$ROOT/data/01_all.jsonl"; A_GROUPS="H19 C19 H24 C24 B C"
B_IN="$ROOT/data/04_new.jsonl"
B_GROUPS=$(grep -o '"group": "[^"]*"' "$B_IN" | cut -d'"' -f4 | sort -u \
  | grep -E '^(RAID\.H|APT\.H|RAID\.(chatgpt|gpt4|cohere-chat|llama-chat|mistral-chat|mpt-chat)\.sampling|APT\..+\.(slight_major|major))$' \
  | tr '\n' ' ')
echo "part B groups: $B_GROUPS"

for c in $CUTS; do
  st="$MOD/tic1b_$c.safetensors"
  if [ -s "$st" ]; then echo "have $st"; continue; fi
  date; echo "== download TiC-LM 1B $c"
  url=$("$PY" -I -c "import sys; sys.path.insert(0, r'$HERE/src'); from trd.tic import checkpoint_url; print(checkpoint_url('1b', '$c'))")
  pt="$MOD/tic1b_$c.pt"
  curl -sSL --retry 5 -C - -A "research-client" -o "$pt" "$url"
  meta=$("$PY" -I "$HERE/scripts/tic_check.py" convert --pt "$pt" --out "$st")
  sha=$(echo "$meta" | "$PY" -I -c "import json,sys; print(json.load(sys.stdin)['sha256'])")
  want=$(awk -F'\t' -v c="$c" '$1 == c {print $3}' "$OUT/tic_checkpoints.tsv")
  if [ "$sha" != "$want" ]; then echo "SHA-256 mismatch for $c: $sha vs 07's $want"; rm -f "$st"; exit 1; fi
  echo "sha256 ok ($sha)"
  rm -f "$pt"
done

date; echo "== pipeline check (12 texts)"
rm -f "$OUT/s10_check.jsonl"
"$PY" -I "$HERE/scripts/score_slots_family.py" --family tic --weights-dir "$MOD" --models $CUTS \
    --input "$A_IN" --groups $A_GROUPS --out "$OUT/s10_check.jsonl" --limit 12 --chunk 12
"$PY" -I "$HERE/scripts/slots_check_family.py" --family tic --weights-dir "$MOD" --models $CUTS \
    --input "$A_IN" --groups $A_GROUPS --check "$OUT/s10_check.jsonl" --cross 202301_202407 202407_202301 \
    --tic-dir "$OUT" --out "$OUT/s10_check.json"

date; echo "== part A: five cutoffs, 25 cells"
"$PY" -I "$HERE/scripts/score_slots_family.py" --family tic --weights-dir "$MOD" --models $CUTS \
    --input "$A_IN" --groups $A_GROUPS --out "$OUT/s10_A.jsonl" --chunk 32 2>&1 | grep -v "CUDACachingAllocator"

date; echo "== part B: 202101 202301 202407, 9 cells"
"$PY" -I "$HERE/scripts/score_slots_family.py" --family tic --weights-dir "$MOD" --models 202101 202301 202407 \
    --input "$B_IN" --groups $B_GROUPS --out "$OUT/s10_B.jsonl" --chunk 48 2>&1 | grep -v "CUDACachingAllocator"

for c in $CUTS; do rm -f "$MOD/tic1b_$c.safetensors"; done
date; echo "done"
