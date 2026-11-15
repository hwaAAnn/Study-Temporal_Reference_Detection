#!/usr/bin/env bash
# Experiment 07 scoring chain: TiC-LM Oracle 1B, seven cutoffs, local GPU (fp16). Resumable: a cutoff whose
# score file and fp16 check both exist is skipped. Per cutoff: download .pt -> SHA-256 + fp16 safetensors ->
# CPU fp32 reference for 3 check texts -> delete .pt -> GPU scoring -> fp16 rule -> delete safetensors.
#
# Usage (from the 07 worktree; data and outputs live in the main checkout, which git ignores):
#   bash scripts/run_07_tic.sh /c/Users/OFFICE/Study-Temporal_Reference_Detection
set -euo pipefail
MAIN=${1:?path to the main checkout (data/, outputs/, .venv/)}
HERE=$(cd "$(dirname "$0")/.." && pwd -W)   # Windows form: it is also embedded in python -c strings
PY="$MAIN/.venv/Scripts/python.exe -I"
export PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1 TQDM_DISABLE=1
IN="$MAIN/data/01_all.jsonl"
OUT="$MAIN/outputs"
MOD="$MAIN/models/tic"          # one checkpoint at a time; git ignores *.pt and *.safetensors
mkdir -p "$OUT" "$MOD"
LOG="$OUT/tic_checkpoints.tsv"
[ -s "$LOG" ] || printf "cutoff\tfile\tsha256\tbytes\n" > "$LOG"

for c in 201305 201501 201701 201901 202101 202301 202407; do
  score="$OUT/tic1b_$c.jsonl"; chk="$OUT/tic1b_${c}_fp16check.json"
  if [ -s "$score" ] && grep -q '"pass": true' "$chk" 2>/dev/null; then echo "skip $c"; continue; fi
  date; echo "== TiC-LM 1B $c"
  url=$($PY -c "import sys; sys.path.insert(0, r'$HERE/src'); from trd.tic import checkpoint_url; print(checkpoint_url('1b', '$c'))")
  pt="$MOD/tic1b_$c.pt"; st="$MOD/tic1b_$c.safetensors"; ref="$OUT/tic1b_${c}_ref.json"
  if [ ! -s "$st" ]; then
    curl -sSL --retry 5 -C - -A "research-client" -o "$pt" "$url"
    meta=$($PY "$HERE/scripts/tic_check.py" convert --pt "$pt" --out "$st")
    echo "$meta"
    sha=$(echo "$meta" | $PY -c "import json,sys; print(json.load(sys.stdin)['sha256'])")
    printf "%s\t%s\t%s\t%s\n" "$c" "$(basename "$url")" "$sha" "$(stat -c %s "$pt")" >> "$LOG"
    $PY "$HERE/scripts/tic_check.py" ref --pt "$pt" --input "$IN" --out "$ref"
    rm -f "$pt"
  fi
  $PY "$HERE/scripts/score_tic.py" --input "$IN" --weights "$st" --cutoff "$c" --out "$score" \
      --no-pair-truncate --batch-size 8
  $PY "$HERE/scripts/tic_check.py" compare --ref "$ref" --scores "$score" --out "$chk"
  rm -f "$st"
done
date; echo "all cutoffs scored"
