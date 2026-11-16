#!/bin/bash
# 12 on a Runpod GPU pod (A100 80GB): vLLM generation with four 2025-2026 generators, input assembly with 01's
# filters, then DatedGPT 5x5 (fp16) and Falcon 2x2 (bf16) scoring of the humans and every kept generation.
# Each scoring step starts with a 12-text pipeline check and stops on failure.
# Run from the unpacked repo root:  mkdir -p outputs; nohup bash scripts/run_12_pod.sh > outputs/run_12.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs
export HF_HOME=/root/hf TQDM_DISABLE=1 PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_PROGRESS_BARS=1
export PIP_BREAK_SYSTEM_PACKAGES=1   # Ubuntu 24.04 image marks its Python as externally managed (PEP 668)

# Safety net (as 11): stop the pod after MAX_HOURS. Do not kill the sleep to cancel it: the subshell then runs
# the stop at once.
if [ -r /proc/1/environ ]; then
  eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^(RUNPOD_POD_ID|RUNPOD_API_KEY)=' | sed 's/^/export /')" || true
fi
( sleep $(( ${MAX_HOURS:-3} * 3600 )); runpodctl pod stop "${RUNPOD_POD_ID:-}" || runpodctl stop pod "${RUNPOD_POD_ID:-}" ) > outputs/watchdog.log 2>&1 &
echo "watchdog: stop after ${MAX_HOURS:-3} h (pid $!)"

DG="2013=datedgpt/datedgpt-2013-base@855538883fd62ae8138789c4858e1dcb708187dc
2019=datedgpt/datedgpt-2019-base@8fe891bd59e31c3666112dd00139c4ed7cee9dda
2022=datedgpt/datedgpt-2022-base@1e2f7b5d6a019e0eafffcac6bb3023c3662736dd
2023=datedgpt/datedgpt-2023-base@ef6cd57356392e44f66f3246a83f71ff7d457520
2024=datedgpt/datedgpt-2024-base@ed8abac3e81ba4f964fc92cf9e9b412123f681f4"
FA="base=tiiuae/falcon-7b@ec89142b67d748a1865ea4451372db8313ada0d8
inst=tiiuae/falcon-7b-instruct@8782b5c5d8c9290412416618f36a133653e85285"
TAGS="${TAGS:-qwen3i qwen3t glm47flash gptoss120b}"

date
python -m pip install -q "vllm==0.33.0" && python -m pip install -q --no-deps -e . || { echo "pip install failed"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch, transformers, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__, '| vllm', vllm.__version__, '|', torch.cuda.get_device_name())"
free -g | head -2; df -h / | tail -1

for t in $TAGS; do
  date; echo "== generate $t"
  if [ -s "outputs/g12_$t.jsonl" ]; then echo "have outputs/g12_$t.jsonl"; continue; fi
  python scripts/generate_recent.py --tag "$t" --titles data/12_titles.jsonl --out "outputs/g12_$t.jsonl" \
    || { echo "GENERATION FAILED: $t"; rm -f "outputs/g12_$t.jsonl"; }
  rm -rf "$HF_HOME/hub" "$HF_HOME/xet"   # one generator at a time; hub 2.x shares blobs across repos (hub/blobs)
  df -h / | tail -1
done
ls outputs/g12_*.jsonl >/dev/null 2>&1 || { echo "no generations"; exit 1; }

date; echo "== build input"
python scripts/build_12_input.py --titles data/12_titles.jsonl --gens outputs/g12_*.jsonl --out data/12_input.jsonl \
    --report outputs/g12_report.jsonl --tokenizer-revision ed8abac3e81ba4f964fc92cf9e9b412123f681f4 || exit 1

score() {  # name dtype models...
  local name=$1 dt=$2; shift 2
  date; echo "== $name: pipeline check (12 texts)"
  rm -f "outputs/s12_${name}_check.jsonl"
  python scripts/score_slots_family.py --family hf --dtype "$dt" --models "$@" --input data/12_input.jsonl \
      --out "outputs/s12_${name}_check.jsonl" --limit 12 --chunk 12 || return 1
  python scripts/slots_check_family.py --family hf --dtype "$dt" --models "$@" --input data/12_input.jsonl \
      --check "outputs/s12_${name}_check.jsonl" --cross "${CROSS[@]}" --tol-f "$TOLF" --tol-b "$TOLB" \
      --out "outputs/s12_${name}_check.json" || { echo "$name pipeline check FAILED"; return 1; }
  date; echo "== $name: grid"
  python scripts/score_slots_family.py --family hf --dtype "$dt" --models "$@" --input data/12_input.jsonl \
      --out "outputs/s12_${name}.jsonl" --chunk 128 --batch-size 16 || return 1
}
CROSS=(2022_2024 2024_2022); TOLF=0.02; TOLB=0.02
score dg float16 $DG || exit 1
CROSS=(base_inst inst_base); TOLF=0.05; TOLB=0.01
score fa bfloat16 $FA || exit 1
date
wc -l outputs/s12_dg.jsonl outputs/s12_fa.jsonl
touch outputs/ALL_DONE
echo "ALL_DONE"
