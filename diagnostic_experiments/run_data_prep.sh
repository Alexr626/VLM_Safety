#!/bin/bash
# ============================================================
# Data Prep: Cohesive Text + CT Activations
# ============================================================
set -e

MODEL="${MODEL:-llava-hf/llava-1.5-7b-hf}"
DATASET="${DATASET:-holisafe}"
PROVIDER="${PROVIDER:-anthropic}"

for arg in "$@"; do eval "$arg"; done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CAPTIONS_DIR="$PROJECT_ROOT/data/captions"

echo "======================================================"
echo " Data Prep   model=$MODEL  provider=$PROVIDER"
echo "======================================================"

if [ ! -f "$CAPTIONS_DIR/$DATASET.json" ]; then
    echo "ERROR: Captions not found. Run run_shiftdc.sh first."
    exit 1
fi

echo "=== [1/2] Generate cohesive text ==="
python "$PROJECT_ROOT/data_scripts/generate_cohesive_text.py" \
    --provider "$PROVIDER" --captions_dir "$CAPTIONS_DIR" \
    --output_dir "$CAPTIONS_DIR" --skip_if_exists

echo "=== [2/2] Extract CT activations ==="
python "$PROJECT_ROOT/data_scripts/extract_ct.py" \
    --model "$MODEL" --dataset "$DATASET" \
    --cohesive_path "$CAPTIONS_DIR/holisafe_cohesive.json" --skip_if_exists

echo "======================================================"
echo " Done."
echo "======================================================"
