#!/usr/bin/env bash
# Qwen2.5-VL × AMBER-25 steered capture dump (default subset, max_new_tokens=128)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
source /data/romanus/miniconda3/etc/profile.d/conda.sh
conda activate vlm_hallucination_mitigation
export HF_HOME="${HF_HOME:-/data/romanus/huggingface}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

AUG=data/amber/augmented_amber25.jsonl
test -f "$AUG"

echo "[dump] Qwen2.5-VL AMBER-25 on GPU=$GPU"
CUDA_VISIBLE_DEVICES="$GPU" python diagnostic_experiments/perception_diag/run_dump.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --augmented_jsonl "$AUG" \
  --cells default_subset \
  --run_tag amber25_steering_settings \
  --max_new_tokens 128 \
  --max_pixels 1003520 \
  --device_map cuda:0

echo "Done: data/amber/dumps/qwen2.5-vl-7b-instruct/amber25_steering_settings"
