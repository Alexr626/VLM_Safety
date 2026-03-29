#!/bin/bash
# ============================================================
# ShiftDC Full Pipeline
# ============================================================
# Usage:
#   bash run_shiftdc.sh                        # default settings
#   bash run_shiftdc.sh --dataset holisafe     # override dataset
#
# All GPU steps run sequentially. Already-completed steps are
# skipped automatically via --skip_if_exists / --skip_extraction.
# ============================================================
set -e   # stop immediately on any error

# ── Configuration ────────────────────────────────────────────
MODEL="llava-hf/llava-1.5-7b-hf"
DATASET="holisafe"
SAFE_REF="llava-instruct"
UNSAFE_REF="mm-safetybench"
BATCH_SIZE=4
MAX_NEW_TOKENS=100

# Allow overriding any variable from the command line:
#   bash run_shiftdc.sh MODEL=other-org/other-model DATASET=mydata
for arg in "$@"; do
    eval "$arg"
done

# ── Run from project root ────────────────────────────────────
cd "$(dirname "$0")"

echo ""
echo "======================================================"
echo " ShiftDC Pipeline"
echo "   model      : $MODEL"
echo "   dataset    : $DATASET"
echo "   safe_ref   : $SAFE_REF"
echo "   unsafe_ref : $UNSAFE_REF"
echo "======================================================"
echo ""

# ── Phase 1: GPU extraction (sequential) ─────────────────────

echo "=== [1/6] Generate captions: $DATASET ==="
python extraction/generate_captions.py \
    --model "$MODEL" --dataset "$DATASET" \
    --batch_size $BATCH_SIZE --max_new_tokens $MAX_NEW_TOKENS \
    --skip_if_exists

echo ""
echo "=== [2/6] Extract VL activations: $DATASET ==="
python extraction/extract_vl.py \
    --model "$MODEL" --dataset "$DATASET" \
    --skip_extraction

echo ""
echo "=== [3/6] Extract TT activations: $DATASET ==="
python extraction/extract_tt.py \
    --model "$MODEL" --dataset "$DATASET" \
    --skip_extraction

echo ""
echo "=== [4/6] Generate captions: $SAFE_REF ==="
python extraction/generate_captions.py \
    --model "$MODEL" --dataset "$SAFE_REF" \
    --batch_size $BATCH_SIZE --max_new_tokens $MAX_NEW_TOKENS \
    --skip_if_exists

echo ""
echo "=== [5/6] Generate captions: $UNSAFE_REF ==="
python extraction/generate_captions.py \
    --model "$MODEL" --dataset "$UNSAFE_REF" \
    --batch_size $BATCH_SIZE --max_new_tokens $MAX_NEW_TOKENS \
    --skip_if_exists

echo ""
echo "=== [6/6] Extract reference activations ==="
python extraction/extract_ref_activations.py \
    --model "$MODEL" \
    --safe_ref "$SAFE_REF" --unsafe_ref "$UNSAFE_REF" \
    --skip_if_exists

# ── Phase 2: CPU analysis ─────────────────────────────────────

echo ""
echo "=== [7/7] ShiftDC analysis ==="
python methods/shift_dc/method2_shiftdc_analysis.py \
    --model "$MODEL" --dataset "$DATASET" \
    --safe_ref "$SAFE_REF" --unsafe_ref "$UNSAFE_REF" \
    --skip_safety_dir

echo ""
echo "======================================================"
echo " Done. Results in: outputs/$(basename $MODEL)/"
echo "======================================================"
