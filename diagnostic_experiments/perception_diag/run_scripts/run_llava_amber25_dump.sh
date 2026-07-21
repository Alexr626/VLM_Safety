#!/usr/bin/env bash
# LLaVA-1.5 × AMBER-25 steered capture dump (all steering settings, max_new_tokens=512)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
source /data/romanus/miniconda3/etc/profile.d/conda.sh
conda activate vlm_hallucination_mitigation
export HF_HOME="${HF_HOME:-/data/romanus/huggingface}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

AUG=data/amber/augmented_amber25.jsonl
test -f "$AUG"

CUDA_VISIBLE_DEVICES="$GPU" python diagnostic_experiments/perception_diag/run_dump.py \
  --model llava-hf/llava-1.5-7b-hf \
  --augmented_jsonl "$AUG" \
  --cells all \
  --run_tag amber25_all_steering_settings \
  --max_new_tokens 512 \
  --device_map cuda:0

echo "Done: data/amber/dumps/llava-1.5-7b-hf/amber25_all_steering_settings"
