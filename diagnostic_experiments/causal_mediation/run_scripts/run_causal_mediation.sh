#!/usr/bin/env bash
set -euo pipefail
MODEL="${MODEL:-llava-hf/llava-1.5-7b-hf}"
BENCHMARK="${BENCHMARK:-pope}"
LIMIT="${LIMIT:-20}"
EXP_DIR="$(cd "$(dirname "$0")/.." && pwd)"

python "$EXP_DIR/run_mediation.py" --model "$MODEL" --benchmark "$BENCHMARK" --limit "$LIMIT"
python "$EXP_DIR/plot_recovery_rates.py" --model "$MODEL"
