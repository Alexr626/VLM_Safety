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

CAPTION_PROVIDER="${CAPTION_PROVIDER:-anthropic}"

# echo "=== [1/3] Compositional safety direction ==="
# python "$DIAGNOSTIC_ROOT/experiment_scripts/compositional_safety_direction.py" --model "$MODEL"
# python "$DIAGNOSTIC_ROOT/plotting_scripts/plot_direction_comparison.py" --model "$MODEL"

echo "=== [2/3] Compositional eval prep (data) ==="
# Append usu_/suu_/uuu_eval_ids to train_eval_split.json (idempotent).
python -m src.dataset --n_eval 175 --seed 42

# # Caption USU/SUU/UUU eval samples (resume-safe; merges into holisafe.json).
# python "$PROJECT_ROOT/data_scripts/generate_captions.py" \
#     --dataset holisafe \
#     --holisafe_eval_only \
#     --holisafe_subsets SUU USU UUU \
#     --provider "$CAPTION_PROVIDER"

# Extract VL + TT activations for the new eval samples (per-sample cache; no-op for cached).
python "$PROJECT_ROOT/data_scripts/extract_vl.py" \
    --model "$MODEL" --dataset holisafe \
    --holisafe_eval_only --holisafe_subsets SUU USU UUU
python "$PROJECT_ROOT/data_scripts/extract_tt.py" \
    --model "$MODEL" --dataset holisafe \
    --holisafe_eval_only --holisafe_subsets SUU USU UUU

echo "=== [3/3] Safety probes + plots ==="
python "$DIAGNOSTIC_ROOT/experiment_scripts/safety_probes.py" --model "$MODEL"
python "$DIAGNOSTIC_ROOT/plotting_scripts/plot_probe_results.py" --model "$MODEL"
python "$DIAGNOSTIC_ROOT/plotting_scripts/plot_compositional_eval.py" --model "$MODEL"

echo "======================================================"
echo " Done."
echo "======================================================"
