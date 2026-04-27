#!/bin/bash
# ============================================================
# Compositional Safety Diagnostic (CPU Analysis)
# ============================================================
# Usage:
#   bash run_compositional_safety.sh
#   MODEL="Qwen/Qwen2-VL-7B-Instruct" bash run_compositional_safety.sh
# ============================================================
set -e

MODEL="${MODEL:-llava-hf/llava-1.5-7b-hf}"
for arg in "$@"; do eval "$arg"; done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIAGNOSTIC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$DIAGNOSTIC_ROOT/.." && pwd)"
MODEL_NAME=$(cd "$PROJECT_ROOT" && python -c "from src.model import _normalize_model_name; print(_normalize_model_name('$MODEL'))")
ARTIFACTS_DIR="$PROJECT_ROOT/experiment_artifacts/$MODEL_NAME"

echo "======================================================"
echo " Compositional Safety Diagnostic   model=$MODEL"
echo "======================================================"

if [ ! -f "$ARTIFACTS_DIR/vl_activation_shift/safety_direction_vectors.npz" ]; then
    echo "ERROR: Safety direction vectors not found at $ARTIFACTS_DIR/vl_activation_shift/."
    echo "Run run_shiftdc.sh first."
    exit 1
fi

echo "=== [1/2] Compositional safety direction ==="
python "$DIAGNOSTIC_ROOT/experiment_scripts/compositional_safety_direction.py" --model "$MODEL"
python "$DIAGNOSTIC_ROOT/plotting_scripts/plot_direction_comparison.py" --model "$MODEL"

echo "=== [2/2] Safety probes ==="
python "$DIAGNOSTIC_ROOT/experiment_scripts/safety_probes.py" --model "$MODEL"
python "$DIAGNOSTIC_ROOT/plotting_scripts/plot_probe_results.py" --model "$MODEL"

echo "======================================================"
echo " Done."
echo "======================================================"
