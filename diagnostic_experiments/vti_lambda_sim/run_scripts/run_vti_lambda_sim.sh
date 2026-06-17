#!/usr/bin/env bash
# Run gated_rotation lambda_sim diagnostic for one model.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-llava-hf/llava-1.5-7b-hf}"
HOOK_SITE="${HOOK_SITE:-mlp}"
POPE_SPLIT="${POPE_SPLIT:-random}"
LIMIT="${LIMIT:-200}"

python diagnostic_experiments/vti_lambda_sim/compute_lambda_sim.py \
  --model "$MODEL" \
  --hook_site "$HOOK_SITE" \
  --pope_split "$POPE_SPLIT" \
  --limit "$LIMIT"

python diagnostic_experiments/vti_lambda_sim/plot_lambda_sim.py \
  --model "$MODEL" \
  --hook_site "$HOOK_SITE" \
  --pope_split "$POPE_SPLIT"
