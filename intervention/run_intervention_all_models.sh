#!/bin/bash
# ============================================================
# Run intervention pipeline (none / original / spherical) on
# all four models under investigation.
# ============================================================
# For each model, runs three passes:
#   1. --method none       (baseline, no intervention)
#   2. --method original   (ShiftDC, Zou et al. 2025)
#   3. --method spherical  (Spherical ShiftDC, proposed, t=1.0)
#
# Prerequisites (per model):
#   data/holisafe-bench/activations/{model}/sample_*_{vl,tt}.npz
#   experiment_artifacts/{model}/vl_activation_shift/safety_direction_vectors.npz
#
# Outputs land in:
#   intervention/{model}/outputs/results/{method}_summary.json
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN="$SCRIPT_DIR/experiment_scripts/run_intervention.py"

MODELS=(
    "llava-hf/llava-1.5-7b-hf"
    "Qwen/Qwen2.5-VL-7B-Instruct"
    "OpenGVLab/InternVL2-8B"
    "OpenGVLab/InternVL2_5-8B-MPO"
)

run_model() {
    local MODEL="$1"
    echo ""
    echo "============================================================"
    echo "  Intervention: $MODEL"
    echo "============================================================"

    echo "[1/3] baseline (none) ..."
    python "$RUN" --model "$MODEL" --method none --skip_if_exists

    echo "[2/3] original ShiftDC ..."
    python "$RUN" --model "$MODEL" --method original --skip_if_exists

    echo "[3/3] spherical ShiftDC (t=1.0) ..."
    python "$RUN" --model "$MODEL" --method spherical --t 1.0 --skip_if_exists

    echo "  Done: $MODEL"
}

for MODEL in "${MODELS[@]}"; do
    run_model "$MODEL"
done

echo ""
echo "============================================================"
echo "  All models complete."
echo "  Plot results with:"
echo "    python intervention/plotting_scripts/plot_intervention_results.py --model <model>"
echo "============================================================"
