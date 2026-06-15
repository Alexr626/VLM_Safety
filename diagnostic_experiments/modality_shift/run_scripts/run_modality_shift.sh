#!/usr/bin/env bash
# Run modality-shift analysis for one model.
set -euo pipefail
MODEL="${MODEL:-llava-hf/llava-1.5-7b-hf}"
BENCHMARK="${BENCHMARK:-pope}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXP_DIR="$(cd "$(dirname "$0")/.." && pwd)"

python "$EXP_DIR/compute_modality_shift.py" --model "$MODEL" --benchmark "$BENCHMARK" "$@"
python "$EXP_DIR/plot_modality_shift.py" --model "$MODEL"
