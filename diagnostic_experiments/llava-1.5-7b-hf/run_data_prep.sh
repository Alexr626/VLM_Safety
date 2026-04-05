#!/bin/bash
# ============================================================
# Data Preparation for Augmented Diagnostic Experiments
# ============================================================
# Generates cohesive text and extracts CT activations.
# Must be run AFTER the base ShiftDC pipeline (run_shiftdc.sh).
#
# Prerequisites:
#   - run_shiftdc.sh completed (captions, VL/TT activations exist)
#
# Cohesive text generation uses the Anthropic API by default.
# Set PROVIDER=openai or PROVIDER=local to change.
# GPU required only for CT extraction (step 2) and PROVIDER=local.
#
# Usage:
#   bash diagnostic_experiments/llava-1.5-7b-hf/run_data_prep.sh
#   bash diagnostic_experiments/llava-1.5-7b-hf/run_data_prep.sh PROVIDER=openai
#   bash diagnostic_experiments/llava-1.5-7b-hf/run_data_prep.sh PROVIDER=local
# ============================================================
set -e

# ── Configuration ────────────────────────────────────────────
MODEL="llava-hf/llava-1.5-7b-hf"
DATASET="holisafe"
PROVIDER="anthropic"

for arg in "$@"; do
    eval "$arg"
done

# ── Resolve directories ─────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_SHORT="${MODEL##*/}"
CAPTIONS_DIR="$PROJECT_ROOT/data/captions"
HOLISAFE_ACT="$PROJECT_ROOT/data/holisafe-bench/activations/$MODEL_SHORT"

echo ""
echo "======================================================"
echo " Data Preparation: Augmented Diagnostics"
echo "   model    : $MODEL"
echo "   dataset  : $DATASET"
echo "   provider : $PROVIDER"
echo "======================================================"

# ── Verify prerequisites ────────────────────────────────────
if [ ! -f "$CAPTIONS_DIR/$DATASET.json" ]; then
    echo "ERROR: Captions not found at $CAPTIONS_DIR/$DATASET.json"
    echo "Run the ShiftDC pipeline first:"
    echo "  bash diagnostic_experiments/llava-1.5-7b-hf/shift_dc/run_shiftdc.sh"
    exit 1
fi

# ── Step 1: Generate cohesive text ───────────────────────────
echo ""
echo "=== [1/2] Generate cohesive text (provider=$PROVIDER) ==="
python "$PROJECT_ROOT/data_scripts/generate_cohesive_text.py" \
    --provider "$PROVIDER" \
    --captions_dir "$CAPTIONS_DIR" \
    --output_dir "$CAPTIONS_DIR" \
    --skip_if_exists

# ── Step 2: Extract CT activations (GPU) ─────────────────────
echo ""
echo "=== [2/2] Extract CT activations ==="
python "$PROJECT_ROOT/data_scripts/extract_ct.py" \
    --model "$MODEL" --dataset "$DATASET" \
    --output_dir "$HOLISAFE_ACT" \
    --cohesive_path "$CAPTIONS_DIR/holisafe_cohesive.json" \
    --skip_if_exists

echo ""
echo "======================================================"
echo " Data preparation complete."
echo "   Cohesive text : $CAPTIONS_DIR/holisafe_cohesive.json"
echo "   CT activations: $HOLISAFE_ACT/"
echo "======================================================"
