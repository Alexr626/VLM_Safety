#!/bin/bash
# ============================================================
# ShiftDC Full Pipeline
# ============================================================
# Usage:
#   bash run_shiftdc.sh                        # default settings
#   bash run_shiftdc.sh DATASET=holisafe        # override dataset
#
# All GPU steps run sequentially. Already-completed steps are
# skipped automatically via --skip_if_exists / --skip_extraction.
# ============================================================
set -e   # stop immediately on any error

# ── Configuration ────────────────────────────────────────────
MODEL="llava-hf/llava-1.5-7b-hf"
DATASET="holisafe"
SAFE_REF="catqa-harmless"
UNSAFE_REF="catqa-harmful"
REF_SAMPLES=550
BATCH_SIZE=4
MAX_NEW_TOKENS=100

# Allow overriding any variable from the command line:
#   bash run_shiftdc.sh MODEL=other-org/other-model DATASET=mydata
for arg in "$@"; do
    eval "$arg"
done

# ── Resolve directories ─────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUTPUTS_DIR="$SCRIPT_DIR/outputs"
MODEL_SHORT="${MODEL##*/}"
CAPTIONS_DIR="$PROJECT_ROOT/data/captions"
HOLISAFE_ACT="$PROJECT_ROOT/data/holisafe-bench/activations/$MODEL_SHORT"
REF_ACT="$PROJECT_ROOT/data/catqa-contrastive/activations/$MODEL_SHORT"

echo ""
echo "======================================================"
echo " ShiftDC Pipeline"
echo "   model      : $MODEL"
echo "   dataset    : $DATASET"
echo "   safe_ref   : $SAFE_REF"
echo "   unsafe_ref : $UNSAFE_REF"
echo "   ref_samples: $REF_SAMPLES"
echo "   results    : $OUTPUTS_DIR/results/"
echo "======================================================"
echo ""

# ── Phase 1: GPU extraction (sequential) ─────────────────────

echo "=== [1/7] Generate captions: $DATASET ==="
python "$PROJECT_ROOT/data_scripts/generate_captions.py" \
    --model "$MODEL" --dataset "$DATASET" \
    --output_dir "$CAPTIONS_DIR" \
    --batch_size $BATCH_SIZE --max_new_tokens $MAX_NEW_TOKENS \
    --skip_if_exists

echo ""
echo "=== [2/7] Extract VL activations: $DATASET ==="
python "$PROJECT_ROOT/data_scripts/extract_vl.py" \
    --model "$MODEL" --dataset "$DATASET" \
    --output_dir "$HOLISAFE_ACT" \
    --skip_extraction

echo ""
echo "=== [3/7] Extract TT activations: $DATASET ==="
python "$PROJECT_ROOT/data_scripts/extract_tt.py" \
    --model "$MODEL" --dataset "$DATASET" \
    --output_dir "$HOLISAFE_ACT" \
    --captions_dir "$CAPTIONS_DIR" \
    --skip_extraction

echo ""
echo "=== [4/7] Generate captions: $SAFE_REF ==="
SAFE_REF_TEXT_ONLY=$(cd "$PROJECT_ROOT" && python -c "from src.dataset import REFERENCE_REGISTRY; print(REFERENCE_REGISTRY.get('$SAFE_REF', {}).get('text_only', False))")
if [ "$SAFE_REF_TEXT_ONLY" = "True" ]; then
    echo "  Skipping — '$SAFE_REF' is text-only (no images to caption)."
else
    python "$PROJECT_ROOT/data_scripts/generate_captions.py" \
        --model "$MODEL" --dataset "$SAFE_REF" \
        --output_dir "$CAPTIONS_DIR" \
        --batch_size $BATCH_SIZE --max_new_tokens $MAX_NEW_TOKENS \
        --skip_if_exists
fi

echo ""
echo "=== [5/7] Generate captions: $UNSAFE_REF ==="
UNSAFE_REF_TEXT_ONLY=$(cd "$PROJECT_ROOT" && python -c "from src.dataset import REFERENCE_REGISTRY; print(REFERENCE_REGISTRY.get('$UNSAFE_REF', {}).get('text_only', False))")
if [ "$UNSAFE_REF_TEXT_ONLY" = "True" ]; then
    echo "  Skipping — '$UNSAFE_REF' is text-only (no images to caption)."
else
    python "$PROJECT_ROOT/data_scripts/generate_captions.py" \
        --model "$MODEL" --dataset "$UNSAFE_REF" \
        --output_dir "$CAPTIONS_DIR" \
        --batch_size $BATCH_SIZE --max_new_tokens $MAX_NEW_TOKENS \
        --skip_if_exists
fi

echo ""
echo "=== [6/7] Extract reference activations ==="
python "$PROJECT_ROOT/data_scripts/extract_ref_activations.py" \
    --model "$MODEL" \
    --output_dir "$REF_ACT" \
    --captions_dir "$CAPTIONS_DIR" \
    --safe_ref "$SAFE_REF" --unsafe_ref "$UNSAFE_REF" \
    --ref_samples $REF_SAMPLES \
    --skip_if_exists

# ── Phase 2: CPU analysis ─────────────────────────────────────

echo ""
echo "=== [7/7] ShiftDC analysis ==="
python "$SCRIPT_DIR/experiment_scripts/vl_activation_shift.py" \
    --model "$MODEL" --dataset "$DATASET" \
    --output_dir "$OUTPUTS_DIR" \
    --safe_ref "$SAFE_REF" --unsafe_ref "$UNSAFE_REF" \
    --skip_safety_dir

echo ""
echo "======================================================"
echo " Done. Results in: $OUTPUTS_DIR/results/"
echo "======================================================"
