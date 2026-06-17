#!/usr/bin/env bash
# POPE evaluation: all six textual VTI variants on LLaVA-1.5-7B.
#
# Runs each POPE split (random, popular, adversarial) with responses
# accumulating in the same output dir per intervention.
#
# Usage:
#   bash evaluation/run_scripts/run_vti_pope_llava.sh
#   LIMIT=10 CUDA_VISIBLE_DEVICES=2 bash evaluation/run_scripts/run_vti_pope_llava.sh
#   SKIP_IF_EXISTS=0 bash evaluation/run_scripts/run_vti_pope_llava.sh   # force re-run
#
# Prerequisites:
#   conda activate vlm_hallucination_mitigation
#   python data_scripts/download_chair.py --with-train2014   # VTI direction demos
#   nvidia-smi  # pick a free GPU via CUDA_VISIBLE_DEVICES
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-llava-hf/llava-1.5-7b-hf}"
LIMIT="${LIMIT:-200}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation/results}"
SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-1}"

IVS=(
  vti_textual_additive_mlp
  vti_textual_additive_layer
  vti_textual_uniform_rotation_mlp
  vti_textual_uniform_rotation_layer
  vti_textual_gated_rotation_mlp
  vti_textual_gated_rotation_layer
)

SKIP_ARGS=()
if [[ "$SKIP_IF_EXISTS" == "1" ]]; then
  SKIP_ARGS=(--skip_if_exists)
fi

echo "=== VTI POPE eval (all variants) ==="
echo "  model          : $MODEL"
echo "  limit/split    : $LIMIT"
echo "  output_dir     : $OUTPUT_DIR"
echo "  skip_if_exists : $SKIP_IF_EXISTS"
echo "  interventions  : ${IVS[*]}"

for split in random popular adversarial; do
  echo ""
  echo "--- POPE split: $split ---"
  python evaluation/run_eval.py \
    --model "$MODEL" \
    --benchmarks pope \
    --interventions "${IVS[@]}" \
    --pope_split "$split" \
    --limit "$LIMIT" \
    --output_dir "$OUTPUT_DIR" \
    "${SKIP_ARGS[@]}"
done

echo ""
python - <<PY
from evaluation.runners import print_comparison_table
from src.model import _normalize_model_name
print_comparison_table(_normalize_model_name("$MODEL"), "$OUTPUT_DIR")
PY
