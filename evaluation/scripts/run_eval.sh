#!/usr/bin/env bash
# Single-model evaluation launcher.
#
# Usage:
#   bash evaluation/scripts/run_eval.sh
#   MODEL=Qwen/Qwen2-VL-7B bash evaluation/scripts/run_eval.sh
#   MODEL=... INTERVENTIONS="vanilla comp_safety_shift" bash evaluation/scripts/run_eval.sh
#   LIMIT=20 bash evaluation/scripts/run_eval.sh   # quick smoke test

set -euo pipefail

MODEL="${MODEL:-llava-hf/llava-1.5-7b-hf}"
BENCHMARKS="${BENCHMARKS:-mm_safetybench figstep mssbench}"
INTERVENTIONS="${INTERVENTIONS:-vanilla adashield_s comp_safety_shift}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation/results}"
LIMIT="${LIMIT:-}"

LIMIT_FLAG=""
if [[ -n "$LIMIT" ]]; then
    LIMIT_FLAG="--limit $LIMIT"
fi

# Activate the conda env if running outside of it.
if [[ -z "${CONDA_DEFAULT_ENV:-}" || "$CONDA_DEFAULT_ENV" != "vlm_safety" ]]; then
    # shellcheck disable=SC1091
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        conda activate vlm_safety
    fi
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# shellcheck disable=SC2086
python evaluation/run_eval.py \
    --model "$MODEL" \
    --interventions $INTERVENTIONS \
    --benchmarks $BENCHMARKS \
    --output_dir "$OUTPUT_DIR" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --skip_if_exists \
    $LIMIT_FLAG
