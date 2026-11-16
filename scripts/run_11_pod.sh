#!/bin/bash
# 11 on a Runpod GPU pod (RTX 4090 24GB, bf16): install, download the two pinned Falcon models (safetensors,
# configs and tokenizer only), pipeline check on 12 texts, then the 2x2 grid on data/11_input.jsonl (7,994 texts).
# Run from the unpacked repo root:  nohup bash scripts/run_11_pod.sh > outputs/run_11.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p outputs
export HF_HOME=/root/hf TQDM_DISABLE=1 PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_PROGRESS_BARS=1

# Safety net (as 06): stop the pod after MAX_HOURS even if nobody is watching. Stopping ends GPU billing, and
# the pod has no volume disk. (Pod env vars live in PID 1's environment.)
if [ -r /proc/1/environ ]; then
  eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^(RUNPOD_POD_ID|RUNPOD_API_KEY)=' | sed 's/^/export /')" || true
fi
( sleep $(( ${MAX_HOURS:-3} * 3600 )); runpodctl pod stop "${RUNPOD_POD_ID:-}" || runpodctl stop pod "${RUNPOD_POD_ID:-}" ) > outputs/watchdog.log 2>&1 &
echo "watchdog: stop after ${MAX_HOURS:-3} h (pid $!)"

BASE="base=tiiuae/falcon-7b@ec89142b67d748a1865ea4451372db8313ada0d8"
INST="inst=tiiuae/falcon-7b-instruct@8782b5c5d8c9290412416618f36a133653e85285"

date
python -m pip install -q "transformers==5.22.2" "huggingface_hub==2.4.0" "safetensors>=0.8" "numpy<3" tqdm \
  && python -m pip install -q --no-deps -e . || { echo "pip install failed"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch, transformers, huggingface_hub as h; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__, '| hub', h.__version__, '|', torch.cuda.get_device_name(), 'bf16', torch.cuda.is_bf16_supported())"
free -g | head -2; df -h / | tail -1

date; echo "== download"
python - <<'EOF' || { echo "download failed"; exit 1; }
from huggingface_hub import snapshot_download
for repo, rev in [("tiiuae/falcon-7b", "ec89142b67d748a1865ea4451372db8313ada0d8"),
                  ("tiiuae/falcon-7b-instruct", "8782b5c5d8c9290412416618f36a133653e85285")]:
    p = snapshot_download(repo, revision=rev, allow_patterns=["*.safetensors", "*.json"], ignore_patterns=["coreml/*"])
    print(repo, rev, p)
EOF
export HF_HUB_OFFLINE=1
df -h / | tail -1

date; echo "== pipeline check (12 texts)"
rm -f outputs/s11_check.jsonl
python scripts/score_slots_family.py --family hf --dtype bfloat16 --models "$BASE" "$INST" \
    --input data/11_input.jsonl --out outputs/s11_check.jsonl --limit 12 --chunk 12 || exit 1
python scripts/slots_check_family.py --family hf --dtype bfloat16 --models "$BASE" "$INST" \
    --input data/11_input.jsonl --check outputs/s11_check.jsonl --tol-f 0.05 --tol-b 0.01 \
    --out outputs/s11_check.json || { echo "pipeline check FAILED: no grid run"; exit 1; }

date; echo "== grid: 2 x 2 cells"
python scripts/score_slots_family.py --family hf --dtype bfloat16 --models "$BASE" "$INST" \
    --input data/11_input.jsonl --out outputs/s11_grid.jsonl --chunk 128 --batch-size 8 || exit 1
date
wc -l outputs/s11_grid.jsonl
touch outputs/ALL_DONE
echo "ALL_DONE"
