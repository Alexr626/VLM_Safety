#!/usr/bin/env bash
# Stage B: Qwen2.5-VL layer-windowed steering (POPE-30 then AMBER-100).
# Colocated with Stage A on the same A6000 (default GPU 0).
# Plan: implementation_plans/layer_windowed_steering_yes_no_probability_plan_2026-07-21.md
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
source /data/romanus/miniconda3/etc/profile.d/conda.sh
conda activate vlm_hallucination_mitigation
export HF_HOME="${HF_HOME:-/data/romanus/huggingface}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
LOG_DIR="$ROOT/diagnostic_experiments/perception_diag/logs"
mkdir -p "$LOG_DIR"

HF_ID="Qwen/Qwen2.5-VL-7B-Instruct"

run_dump () {
  local aug="$1" tag="$2"
  echo "[$(date -Is)] START tag=$tag gpu=$GPU model=qwen2.5-vl" >&2
  CUDA_VISIBLE_DEVICES="$GPU" python diagnostic_experiments/perception_diag/run_dump.py \
    --model "$HF_ID" \
    --augmented_jsonl "$aug" \
    --windowed_grid \
    --conditions gold_conditional \
    --run_tag "$tag" \
    --max_new_tokens 128 \
    --max_pixels 1003520 \
    --device_map cuda:0
  echo "[$(date -Is)] DONE tag=$tag model=qwen2.5-vl" >&2
}

build_summary () {
  local label="$1"
  echo "[$(date -Is)] aggregator build ($label)" >&2
  # Do not abort dumps if the aggregator fails (set -e).
  python diagnostic_experiments/perception_diag/build_windowed_steering_summary.py \
    || echo "[$(date -Is)] WARNING: aggregator failed ($label); continuing dumps" >&2
}

# B1 — POPE-30
run_dump data/pope/augmented_pope30.jsonl pope30_windowed_steering
build_summary "after_B1_qwen_pope30"

# B2 — AMBER-100
run_dump data/amber/augmented_amber100.jsonl amber100_windowed_steering
build_summary "after_B2_qwen_amber100"

echo "[$(date -Is)] Stage B complete" >&2
