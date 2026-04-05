#!/bin/bash
# ============================================================
# Experiment 3: Behavioral Ground Truth
# ============================================================
# Generates model responses under VL/TT/CT conditions,
# classifies them as refusal or compliance, and plots results.
#
# Prerequisites:
#   - run_shiftdc.sh completed (captions, VL/TT activations)
#   - run_data_prep.sh completed (cohesive text, CT activations)
#
# GPU required: Yes (response generation)
#
# Usage:
#   bash diagnostic_experiments/llava-1.5-7b-hf/run_behavioral_ground_truth.sh
#   bash ... CLASSIFY_METHOD=llm   # use LLM-based classification
# ============================================================
set -e

# ── Configuration ────────────────────────────────────────────
MODEL="llava-hf/llava-1.5-7b-hf"
CLASSIFY_METHOD="llm"

for arg in "$@"; do
    eval "$arg"
done

# ── Resolve directories ─────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXP_DIR="$SCRIPT_DIR/behavioral_ground_truth"

echo ""
echo "======================================================"
echo " Experiment 3: Behavioral Ground Truth"
echo "   model            : $MODEL"
echo "   classify method  : $CLASSIFY_METHOD"
echo "======================================================"

# ── Step 1: Generate responses (GPU) ─────────────────────────
echo ""
echo "=== [1/3] Generate model responses ==="
python "$EXP_DIR/experiment_scripts/generate_responses.py" \
    --model "$MODEL" \
    --skip_if_exists

# ── Step 2: Classify responses (CPU) ─────────────────────────
echo ""
echo "=== [2/3] Classify responses (method=$CLASSIFY_METHOD) ==="
python "$EXP_DIR/experiment_scripts/classify_responses.py" \
    --method "$CLASSIFY_METHOD"

# ── Step 3: Plot results ─────────────────────────────────────
echo ""
echo "=== [3/3] Plot behavioral ground truth ==="
python "$EXP_DIR/plotting_scripts/plot_behavioral_ground_truth.py"

echo ""
echo "======================================================"
echo " Experiment 3 complete. Results in:"
echo "   $EXP_DIR/outputs/results/"
echo "   $EXP_DIR/outputs/results/plots/"
echo "======================================================"
