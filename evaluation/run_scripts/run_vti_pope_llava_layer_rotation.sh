#!/usr/bin/env bash
# POPE evaluation: layer-hook rotation variants only (debug empty-response issue).
#
# Same layout as run_vti_pope_llava.sh, but restricted to the two interventions
# that currently produce empty model responses on LLaVA:
#   - vti_textual_uniform_rotation_layer
#   - vti_textual_gated_rotation_layer
#
# (additive_layer and all *_mlp variants generate non-empty text in smoke tests.)
#
# Usage:
#   bash evaluation/run_scripts/run_vti_pope_llava_layer_rotation.sh
#   LIMIT=10 SKIP_IF_EXISTS=0 bash evaluation/run_scripts/run_vti_pope_llava_layer_rotation.sh
#   FORCE_CLEAN=1 LIMIT=10 bash evaluation/run_scripts/run_vti_pope_llava_layer_rotation.sh
#
# FORCE_CLEAN=1 removes prior responses/metrics for these two interventions before
# running (useful after a failed or smoke-test run left empty checkpoints).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-llava-hf/llava-1.5-7b-hf}"
LIMIT="${LIMIT:-200}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation/results}"
SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-1}"
FORCE_CLEAN="${FORCE_CLEAN:-0}"

IVS=(
  vti_textual_uniform_rotation_layer
  vti_textual_gated_rotation_layer
)

SKIP_ARGS=()
if [[ "$SKIP_IF_EXISTS" == "1" ]]; then
  SKIP_ARGS=(--skip_if_exists)
fi

MODEL_SHORT="$(python -c "from src.model import _normalize_model_name; print(_normalize_model_name('$MODEL'))")"

if [[ "$FORCE_CLEAN" == "1" ]]; then
  for iv in "${IVS[@]}"; do
    rm -rf "$OUTPUT_DIR/$MODEL_SHORT/pope/$iv"
    echo "Removed $OUTPUT_DIR/$MODEL_SHORT/pope/$iv"
  done
fi

echo "=== VTI POPE eval (layer rotation debug) ==="
echo "  model          : $MODEL ($MODEL_SHORT)"
echo "  limit/split    : $LIMIT"
echo "  output_dir     : $OUTPUT_DIR"
echo "  skip_if_exists : $SKIP_IF_EXISTS"
echo "  force_clean    : $FORCE_CLEAN"
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
