#!/usr/bin/env bash
# Concurrent LLaVA POPE-30-yes + POPE-30-no windowed steering on one A6000.
# Plan: implementation_plans/7-21-26/pope_yes_no_windowed_steering_control_plan_2026-07-22.md
# Alex override (chat): run yes+no concurrently on GPU 0 (plan default was sequential).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"
source /data/romanus/miniconda3/etc/profile.d/conda.sh
conda activate vlm_hallucination_mitigation
export HF_HOME="${HF_HOME:-/data/romanus/huggingface}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
LOG_DIR="$ROOT/diagnostic_experiments/perception_diag/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
HF_ID="llava-hf/llava-1.5-7b-hf"

run_one () {
  local aug="$1" tag="$2" logfile="$3"
  echo "[$(date -Is)] START model=llava tag=$tag gpu=$GPU" | tee -a "$logfile"
  CUDA_VISIBLE_DEVICES="$GPU" python diagnostic_experiments/perception_diag/run_dump.py \
    --model "$HF_ID" \
    --augmented_jsonl "$aug" \
    --windowed_grid \
    --conditions gold_conditional \
    --run_tag "$tag" \
    --max_new_tokens 128 \
    --device_map cuda:0 \
    >>"$logfile" 2>&1
  local ec=$?
  echo "[$(date -Is)] DONE tag=$tag exit=$ec" | tee -a "$logfile"
  return $ec
}

LOG_YES="$LOG_DIR/llava_pope30_yes_windowed_steering_${STAMP}.log"
LOG_NO="$LOG_DIR/llava_pope30_no_windowed_steering_${STAMP}.log"

# Stagger load by ~90s so the first process can allocate weights before the second.
run_one data/pope/augmented_pope30_no.jsonl pope30_no_windowed_steering "$LOG_NO" &
PID_NO=$!
sleep 90
run_one data/pope/augmented_pope30_yes.jsonl pope30_yes_windowed_steering "$LOG_YES" &
PID_YES=$!

echo "[$(date -Is)] LLaVA concurrent PIDs: no=$PID_NO yes=$PID_YES" >&2
ec_no=0
ec_yes=0
wait "$PID_NO" || ec_no=$?
wait "$PID_YES" || ec_yes=$?
echo "[$(date -Is)] LLaVA finished no_exit=$ec_no yes_exit=$ec_yes" >&2

python diagnostic_experiments/perception_diag/check_pope30_yes_reproducibility.py \
  --model llava-1.5-7b-hf \
  --out "diagnostic_experiments/perception_diag/windowed_steering_summary/pope30_yes_reproducibility_llava.json" \
  || echo "[$(date -Is)] WARNING: LLaVA reproducibility check failed" >&2

python diagnostic_experiments/perception_diag/build_windowed_steering_summary.py \
  --models llava-1.5-7b-hf \
  || echo "[$(date -Is)] WARNING: aggregator failed after LLaVA" >&2

exit $(( ec_no != 0 || ec_yes != 0 ? 1 : 0 ))
