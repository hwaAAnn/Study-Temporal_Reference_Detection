#!/bin/bash
# 06 on a Runpod GPU pod: install, precision check (decides bf16 vs fp32 by the rule in the 06 doc), then all
# 24 PIT vintages. Run from the unpacked repo root:  nohup bash scripts/run_06_pod.sh > outputs/run_06.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs
export HF_HOME=/root/hf TQDM_DISABLE=1 PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_PROGRESS_BARS=1

# Safety net: stop the pod after MAX_HOURS even if nobody is watching. Stop ends GPU billing, and the pod
# has no volume disk, so nothing keeps billing. (Pod env vars live in PID 1's environment.)
if [ -r /proc/1/environ ]; then
  eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^(RUNPOD_POD_ID|RUNPOD_API_KEY)=' | sed 's/^/export /')" || true
fi
( sleep $(( ${MAX_HOURS:-5} * 3600 )); runpodctl pod stop "${RUNPOD_POD_ID:-}" || runpodctl stop pod "${RUNPOD_POD_ID:-}" ) > outputs/watchdog.log 2>&1 &
echo "watchdog: stop after ${MAX_HOURS:-5} h (pid $!)"

date
python -m pip install -q "transformers==5.22.2" "huggingface_hub==2.4.0" "safetensors>=0.8" "numpy<3" tqdm \
  && python -m pip install -q --no-deps -e . || { echo "pip install failed"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch, transformers, huggingface_hub as h; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__, '| hub', h.__version__, '|', torch.cuda.get_device_name(), 'bf16', torch.cuda.is_bf16_supported())"
df -h / | tail -1

# 1) precision check on the 2022 vintages (first 800 texts), snapshots kept for the main run
date
python scripts/run_pit.py --years 2022 --limit 800 --dtype float32 --tag _chk32 --keep || exit 1
python scripts/run_pit.py --years 2022 --limit 800 --dtype bfloat16 --tag _chk16 --keep || exit 1
python scripts/pit_precision_check.py --json outputs/pit_precision.json --pairs \
  outputs/pit1b_2022_chk32.jsonl:outputs/pit1b_2022_chk16.jsonl \
  outputs/pit4b_2022_chk32.jsonl:outputs/pit4b_2022_chk16.jsonl
case $? in
  0) DT=bfloat16 ;;
  3) DT=float32 ;;
  *) echo "precision check crashed"; exit 1 ;;
esac
echo "dtype for the main run: $DT"

# 2) all vintages, 1B first (fast, early sanity check), then 4B
date
python scripts/run_pit.py --sizes 1B 4B --dtype "$DT" || exit 1
date
touch outputs/ALL_DONE
echo "ALL_DONE"
