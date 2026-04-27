#!/bin/bash
# ============================================================
# Run the combinatorial safety experiment on every supported model
# ============================================================
# For each MODEL listed below, runs:
#   - combinatorial_direction.py  (+ plot_direction_comparison.py)
#   - safety_probes.py            (+ plot_probe_results.py)
#
# Use this to regenerate the combinatorial-safety results across
# all models — e.g. after changes to the CatQA train/eval split or
# the content probe.
#
# Prerequisite: each model must have CatQA reference activations
# and a safety_direction_vectors.npz (i.e. run_shiftdc.sh has
# completed for that model).
#
# CPU-only (no GPU required).
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIAGNOSTIC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$DIAGNOSTIC_ROOT/.." && pwd)"

# By default, abort on first failure. Set CONTINUE_ON_ERROR=1 to
# log per-model failures and keep going across the rest.
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

# Models to run. Comment out any you don't want to process.
MODELS=(
    "llava-hf/llava-1.5-7b-hf"
    # "llava-hf/llava-v1.6-vicuna-7b-hf"
    "Lin-Chen/ShareGPT4V-7B"
    "Vision-CAIR/MiniGPT-4"
    "Qwen/Qwen-VL-Chat"
    "Qwen/Qwen2-VL-7B"
    "Qwen/Qwen2-VL-7B-Instruct"
    "Qwen/Qwen2.5-VL-7B-Instruct"
    "OpenGVLab/InternVL2-8B"
    "OpenGVLab/InternVL2_5-8B-MPO"
)

failed=()

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  Combinatorial safety: $MODEL"
    echo "============================================================"
    if [ "$CONTINUE_ON_ERROR" = "1" ]; then
        if ! MODEL="$MODEL" bash "$SCRIPT_DIR/run_combinatorial_safety.sh"; then
            echo "  WARNING: failed for $MODEL — continuing."
            failed+=("$MODEL")
        fi
    else
        MODEL="$MODEL" bash "$SCRIPT_DIR/run_combinatorial_safety.sh"
    fi
done

echo ""
echo "============================================================"
if [ ${#failed[@]} -eq 0 ]; then
    echo "  Combinatorial safety run complete for all models."
else
    echo "  Combinatorial safety run complete with failures:"
    for m in "${failed[@]}"; do
        echo "    - $m"
    done
fi
echo "============================================================"
