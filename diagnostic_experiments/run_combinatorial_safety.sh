#!/bin/bash
# ============================================================
# Combinatorial Safety Diagnostic (CPU Analysis)
# ============================================================
# Usage:
#   bash run_combinatorial_safety.sh
#   MODEL="Qwen/Qwen2.5-VL-7B-Instruct" bash run_combinatorial_safety.sh
# ============================================================
set -e

MODEL="${MODEL:-llava-hf/llava-1.5-7b-hf}"
for arg in "$@"; do eval "$arg"; done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_NAME=$(cd "$PROJECT_ROOT" && python -c "from src.model import _normalize_model_name; print(_normalize_model_name('$MODEL'))")
ARTIFACTS_DIR="$PROJECT_ROOT/experiment_artifacts/$MODEL_NAME"

echo "======================================================"
echo " Combinatorial Safety Diagnostic   model=$MODEL"
echo "======================================================"

if [ ! -f "$ARTIFACTS_DIR/vl_activation_shift/safety_direction_vectors.npz" ]; then
    echo "ERROR: Safety direction vectors not found at $ARTIFACTS_DIR/vl_activation_shift/."
    echo "Run run_shiftdc.sh first."
    exit 1
fi

echo "=== [1/2] Combinatorial safety direction ==="
python "$SCRIPT_DIR/experiment_scripts/combinatorial_direction.py" --model "$MODEL"
python "$SCRIPT_DIR/plotting_scripts/plot_direction_comparison.py" --model "$MODEL"

echo "=== [2/2] Safety probes ==="
python "$SCRIPT_DIR/experiment_scripts/safety_probes.py" --model "$MODEL"
python "$SCRIPT_DIR/plotting_scripts/plot_probe_results.py" --model "$MODEL"

echo "======================================================"
echo " Done."
echo "======================================================"
