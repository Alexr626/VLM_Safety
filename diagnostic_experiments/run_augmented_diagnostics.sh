#!/bin/bash
# ============================================================
# Augmented Baseline (CPU Analysis)
# ============================================================
# Usage:
#   bash run_augmented_diagnostics.sh
#   MODEL="OpenGVLab/InternVL2-8B" bash run_augmented_diagnostics.sh
# ============================================================
set -e

MODEL="${MODEL:-llava-hf/llava-1.5-7b-hf}"
for arg in "$@"; do eval "$arg"; done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_NAME=$(cd "$PROJECT_ROOT" && python -c "from src.model import _normalize_model_name; print(_normalize_model_name('$MODEL'))")
ARTIFACTS_DIR="$PROJECT_ROOT/experiment_artifacts/$MODEL_NAME"

echo "======================================================"
echo " Augmented Baseline   model=$MODEL"
echo "======================================================"

if [ ! -f "$ARTIFACTS_DIR/vl_activation_shift/safety_direction_vectors.npz" ]; then
    echo "ERROR: Safety direction vectors not found. Run run_shiftdc.sh first."
    exit 1
fi

python "$SCRIPT_DIR/experiment_scripts/augmented_baseline_projections.py" --model "$MODEL"
python "$SCRIPT_DIR/plotting_scripts/plot_augmented_baseline.py" --model "$MODEL"

echo "======================================================"
echo " Done."
echo "======================================================"
