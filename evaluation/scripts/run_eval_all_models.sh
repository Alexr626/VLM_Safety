#!/usr/bin/env bash
# Run the evaluation for all five target models, sequentially.
set -euo pipefail

MODELS=(
    "llava-hf/llava-1.5-7b-hf"
    "Lin-Chen/ShareGPT4V-7B"
    "Qwen/Qwen-VL-Chat"
    "Qwen/Qwen2-VL-7B"
    "Qwen/Qwen2-VL-7B-Instruct"
)

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

for MODEL in "${MODELS[@]}"; do
    echo "======================================================"
    echo " Evaluating: $MODEL"
    echo "======================================================"
    MODEL="$MODEL" bash "$PROJECT_ROOT/evaluation/scripts/run_eval.sh"
done
