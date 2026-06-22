#!/usr/bin/env bash
# VTI rotation-strength experiment — layer-site rotation beta sweep.
#
# Sweeps the textual steering coefficient beta for the layer-hook rotation
# variants across N POPE samples and reports, per beta: POPE accuracy /
# yes_ratio / mean_len / decision-flip direction vs the no-hook baseline, plus
# decode-only and skip-position-0 mitigation probes at the strongest beta.
#
# Results: evaluation/vti_rotation_strength/results/{run_date}/{model}/
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash evaluation/vti_rotation_strength/run_scripts/run_rotation_strength_sweep.sh
#   NUM_SAMPLES=200 MODELS="llava-hf/llava-1.5-7b-hf" bash .../run_rotation_strength_sweep.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

export HF_HOME="${HF_HOME:-/data/romanus/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
POPE_SPLIT="${POPE_SPLIT:-random}"
RUN_DATE="${RUN_DATE:-$(date +%Y-%m-%d)}"
# Fine grid in the live zone; override with BETAS="...".
BETAS="${BETAS:-0.6 0.5 0.45 0.4 0.35 0.3 0.25 0.2 0.1}"

# gated_rotation is numerically identical to uniform_rotation (RESEARCH_LOG
# 2026-06-18); default to uniform only. Override with VARIANTS="uniform_rotation gated_rotation".
read -r -a MODELS <<< "${MODELS:-llava-hf/llava-1.5-7b-hf Qwen/Qwen-VL-Chat Qwen/Qwen2-VL-7B-Instruct Qwen/Qwen2.5-VL-7B-Instruct}"
read -r -a VARIANTS <<< "${VARIANTS:-uniform_rotation}"
read -r -a BETA_ARR <<< "$BETAS"

echo "=== VTI rotation-strength sweep ==="
echo "  models   : ${MODELS[*]}"
echo "  variants : ${VARIANTS[*]}"
echo "  n        : $NUM_SAMPLES   split: $POPE_SPLIT   gpu: $CUDA_VISIBLE_DEVICES"
echo "  betas    : ${BETAS}"
echo "  date     : $RUN_DATE"

for MODEL in "${MODELS[@]}"; do
  for VARIANT in "${VARIANTS[@]}"; do
    echo ""
    echo "--- $MODEL | $VARIANT @ layer | n=$NUM_SAMPLES ---"
    python evaluation/vti_rotation_strength/rotation_strength.py \
      --model "$MODEL" \
      --variant "$VARIANT" \
      --num_samples "$NUM_SAMPLES" \
      --pope_split "$POPE_SPLIT" \
      --betas "${BETA_ARR[@]}" \
      --run_date "$RUN_DATE" \
      || echo "[warn] sweep failed for $MODEL / $VARIANT (continuing)"
  done
done

echo ""
echo "=== rotation-strength sweep complete ==="
