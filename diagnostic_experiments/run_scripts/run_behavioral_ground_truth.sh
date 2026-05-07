#!/bin/bash
# ============================================================
# Behavioral Ground Truth
# ============================================================
# Usage:
#   bash run_behavioral_ground_truth.sh
#   MODEL="Qwen/Qwen2-VL-7B-Instruct" bash run_behavioral_ground_truth.sh
#   bash run_behavioral_ground_truth.sh CLASSIFY_METHOD=keyword
# ============================================================
set -e

MODEL="${MODEL:-llava-hf/llava-1.5-7b-hf}"
CLASSIFY_METHOD="${CLASSIFY_METHOD:-llm_twoaxis}"
PROVIDER="${PROVIDER:-anthropic}"
CONDITIONS="${CONDITIONS:-vl,tt}"

for arg in "$@"; do eval "$arg"; done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIAGNOSTIC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "======================================================"
echo " Behavioral Ground Truth   model=$MODEL"
echo "======================================================"

echo "=== [1/4] Generate model responses (GPU) ==="
python "$DIAGNOSTIC_ROOT/experiment_scripts/generate_responses.py" \
    --model "$MODEL" --skip_if_exists

echo "=== [2/4] Classify responses (method=$CLASSIFY_METHOD) ==="
python "$DIAGNOSTIC_ROOT/experiment_scripts/classify_responses.py" \
    --model "$MODEL" --method "$CLASSIFY_METHOD" \
    --provider "$PROVIDER" --conditions "$CONDITIONS"

echo "=== [3/4] CatQA behavioral baseline ==="
python "$DIAGNOSTIC_ROOT/experiment_scripts/catqa_behavioral_baseline.py" \
    --model "$MODEL" --method "$CLASSIFY_METHOD" --provider "$PROVIDER" \
    --skip_if_exists

echo "=== [4/4] Plot behavioral ground truth ==="
python "$DIAGNOSTIC_ROOT/plotting_scripts/plot_behavioral_ground_truth.py" --model "$MODEL"

echo "======================================================"
echo " Done."
echo "======================================================"
