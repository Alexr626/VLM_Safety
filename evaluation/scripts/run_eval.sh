#!/usr/bin/env bash
# Single-model hallucination evaluation launcher.
set -euo pipefail
MODEL="${MODEL:-llava-hf/llava-1.5-7b-hf}"
BENCHMARKS="${BENCHMARKS:-pope}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

python "$ROOT/evaluation/run_eval.py" \
    --model "$MODEL" \
    --benchmarks $BENCHMARKS \
    --interventions no_intervention \
    "$@"
