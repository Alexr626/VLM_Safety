#!/usr/bin/env bash
# Stage A: LLaVA layer-windowed steering (POPE-30 then AMBER-100).
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

HF_ID="llava-hf/llava-1.5-7b-hf"

run_dump () {
  local aug="$1" tag="$2"
  echo "[$(date -Is)] START tag=$tag gpu=$GPU" >&2
  CUDA_VISIBLE_DEVICES="$GPU" python diagnostic_experiments/perception_diag/run_dump.py \
    --model "$HF_ID" \
    --augmented_jsonl "$aug" \
    --windowed_grid \
    --conditions gold_conditional \
    --run_tag "$tag" \
    --max_new_tokens 128 \
    --device_map cuda:0
  echo "[$(date -Is)] DONE tag=$tag" >&2
}

build_summary () {
  local label="$1"
  echo "[$(date -Is)] aggregator build ($label)" >&2
  # Do not abort dumps if the aggregator fails (set -e).
  python diagnostic_experiments/perception_diag/build_windowed_steering_summary.py \
    --models llava-1.5-7b-hf \
    || echo "[$(date -Is)] WARNING: aggregator failed ($label); continuing dumps" >&2
}

# A1 — POPE-30
run_dump data/pope/augmented_pope30.jsonl pope30_windowed_steering
build_summary "after_A1_pope30"

# A2 — AMBER-100
run_dump data/amber/augmented_amber100.jsonl amber100_windowed_steering
build_summary "after_A2_amber100"

echo "[$(date -Is)] Stage A complete" >&2
