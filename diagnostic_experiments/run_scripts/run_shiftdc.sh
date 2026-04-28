#!/bin/bash
# ============================================================
# ShiftDC Full Pipeline
# ============================================================
# Usage:
#   bash run_shiftdc.sh
#   MODEL="OpenGVLab/InternVL2-8B" bash run_shiftdc.sh
#   bash run_shiftdc.sh MODEL=OpenGVLab/InternVL2-8B DATASET=holisafe
# ============================================================
set -e

MODEL="${MODEL:-llava-hf/llava-1.5-7b-hf}"
DATASET="${DATASET:-holisafe}"
SAFE_REF="${SAFE_REF:-catqa-harmless}"
UNSAFE_REF="${UNSAFE_REF:-catqa-harmful}"
REF_SAMPLES="${REF_SAMPLES:-550}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"

for arg in "$@"; do eval "$arg"; done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIAGNOSTIC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$DIAGNOSTIC_ROOT/.." && pwd)"
CAPTIONS_DIR="$PROJECT_ROOT/data/captions"

echo ""
echo "======================================================"
echo " ShiftDC Pipeline   model=$MODEL  dataset=$DATASET"
echo "======================================================"

echo "=== [1/6] Generate captions: $DATASET ==="
python "$PROJECT_ROOT/data_scripts/generate_captions.py" \
    --model "$MODEL" --dataset "$DATASET" \
    --output_dir "$CAPTIONS_DIR" \
    --batch_size $BATCH_SIZE --max_new_tokens $MAX_NEW_TOKENS \
    --skip_if_exists

echo "=== [2/6] Extract VL activations ==="
python "$PROJECT_ROOT/data_scripts/extract_vl.py" \
    --model "$MODEL" --dataset "$DATASET" --skip_extraction

echo "=== [3/6] Extract TT activations ==="
python "$PROJECT_ROOT/data_scripts/extract_tt.py" \
    --model "$MODEL" --dataset "$DATASET" \
    --captions_dir "$CAPTIONS_DIR" --skip_extraction

echo "=== [3b/6] Extract CT activations (if cohesive text exists) ==="
CT_PATH="$CAPTIONS_DIR/holisafe_cohesive.json"
if [ -f "$CT_PATH" ]; then
    python "$PROJECT_ROOT/data_scripts/extract_ct.py" \
        --model "$MODEL" --dataset "$DATASET" \
        --cohesive_path "$CT_PATH" --skip_if_exists
else
    echo "  Skipping — cohesive text not found at $CT_PATH"
fi

echo "=== [4/6] Generate captions: $SAFE_REF (if needs captions) ==="
SAFE_REF_TEXT_ONLY=$(cd "$PROJECT_ROOT" && python -c "from src.dataset import REFERENCE_REGISTRY; print(REFERENCE_REGISTRY.get('$SAFE_REF', {}).get('text_only', False))")
if [ "$SAFE_REF_TEXT_ONLY" = "True" ]; then
    echo "  Skipping — '$SAFE_REF' is text-only."
else
    python "$PROJECT_ROOT/data_scripts/generate_captions.py" \
        --model "$MODEL" --dataset "$SAFE_REF" --output_dir "$CAPTIONS_DIR" \
        --batch_size $BATCH_SIZE --max_new_tokens $MAX_NEW_TOKENS --skip_if_exists
fi

echo "=== [4b/6] Generate captions: $UNSAFE_REF (if needs captions) ==="
UNSAFE_REF_TEXT_ONLY=$(cd "$PROJECT_ROOT" && python -c "from src.dataset import REFERENCE_REGISTRY; print(REFERENCE_REGISTRY.get('$UNSAFE_REF', {}).get('text_only', False))")
if [ "$UNSAFE_REF_TEXT_ONLY" = "True" ]; then
    echo "  Skipping — '$UNSAFE_REF' is text-only."
else
    python "$PROJECT_ROOT/data_scripts/generate_captions.py" \
        --model "$MODEL" --dataset "$UNSAFE_REF" --output_dir "$CAPTIONS_DIR" \
        --batch_size $BATCH_SIZE --max_new_tokens $MAX_NEW_TOKENS --skip_if_exists
fi

echo "=== [5/6] Extract reference activations ==="
python "$PROJECT_ROOT/data_scripts/extract_ref_activations.py" \
    --model "$MODEL" --captions_dir "$CAPTIONS_DIR" \
    --safe_ref "$SAFE_REF" --unsafe_ref "$UNSAFE_REF" \
    --ref_samples $REF_SAMPLES --skip_if_exists

echo "=== [6/6] ShiftDC analysis ==="
python "$DIAGNOSTIC_ROOT/experiment_scripts/vl_activation_shift.py" \
    --model "$MODEL" --dataset "$DATASET" \
    --safe_ref "$SAFE_REF" --unsafe_ref "$UNSAFE_REF" --skip_safety_dir --compositional_safety_dir
python "$DIAGNOSTIC_ROOT/experiment_scripts/sanity_check_tt_baseline.py" --model "$MODEL" --compositional_safety_dir
python "$DIAGNOSTIC_ROOT/plotting_scripts/plot_vl_activation_shift_projections.py" --model "$MODEL"
python "$DIAGNOSTIC_ROOT/plotting_scripts/plot_tt_baseline_projections.py" --model "$MODEL"

echo "======================================================"
echo " Done."
echo "======================================================"
