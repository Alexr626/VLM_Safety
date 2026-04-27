#!/bin/bash
# ============================================================
# Overnight: compute the compositional safety direction vector
# for each supported model, sequentially. For models whose
# upstream artifacts (HoliSafe SSS+SSU activations, CatQA ref
# activations, CatQA-derived safety direction) are missing,
# this also runs the upstream extraction via run_shiftdc.sh,
# which is GPU-bound. Models with cached artifacts skip
# extraction (handled by --skip_extraction / --skip_if_exists
# inside run_shiftdc.sh) and only re-run the SVD step (CPU,
# fast).
#
# Outputs (per model):
#   experiment_artifacts/{model_short}/vl_activation_shift/
#       safety_direction_vectors.npz                    (CatQA s^l)
#   experiment_artifacts/{model_short}/compositional_safety/
#       compositional_safety_direction_vectors.npz      (compositional c^l)
# ============================================================
# Recommended invocation:
#
#   mkdir -p logs
#   nohup bash diagnostic_experiments/run_scripts/run_overnight_comp_directions.sh \
#       > logs/comp_directions_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   echo $! > logs/comp_directions.pid
#
#   # Override the model list:
#   MODELS="Qwen/Qwen-VL-Chat" nohup bash ... &
#
# Tail:  tail -f logs/comp_directions_*.log
# Stop:  kill $(cat logs/comp_directions.pid)
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIAGNOSTIC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$DIAGNOSTIC_ROOT/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate conda env if not already active.
if [[ "${CONDA_DEFAULT_ENV:-}" != "vlm_safety" ]]; then
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
    conda activate vlm_safety
fi

# Default model list (override with: MODELS="model1 model2" bash ...).
DEFAULT_MODELS=(
    "llava-hf/llava-1.5-7b-hf"
    "Lin-Chen/ShareGPT4V-7B"
    "Qwen/Qwen-VL-Chat"
    "Qwen/Qwen2-VL-7B"
    "Qwen/Qwen2-VL-7B-Instruct"
)
read -r -a MODELS_ARR <<< "${MODELS:-${DEFAULT_MODELS[*]}}"

echo "============================================================"
echo " Overnight Comp-Direction Pipeline  ($(date))"
echo "============================================================"
echo "  models   : ${MODELS_ARR[*]}"
echo "  python   : $(which python)"
echo "  env      : ${CONDA_DEFAULT_ENV:-<none>}"
echo

# Track per-model timings + status.
declare -A STATUS

for M in "${MODELS_ARR[@]}"; do
    echo
    echo "============================================================"
    echo " Model: $M    $(date)"
    echo "============================================================"
    t0=$(date +%s)

    # Step 1: Ensure upstream artifacts (captions, VL/TT/ref activations,
    #   CatQA safety direction). run_shiftdc.sh has skip-if-exists semantics
    #   throughout, so cached artifacts are not recomputed.
    echo "--- [1/2] Upstream artifacts (run_shiftdc.sh) ---"
    if bash "$DIAGNOSTIC_ROOT/run_scripts/run_shiftdc.sh" MODEL="$M"; then
        :
    else
        echo "  [error] run_shiftdc.sh failed for $M; skipping comp_safety_direction."
        STATUS[$M]="upstream_failed"
        continue
    fi

    # Step 2: Compositional safety direction (CPU, fast).
    echo
    echo "--- [2/2] compositional_safety_direction.py ---"
    if python "$DIAGNOSTIC_ROOT/experiment_scripts/compositional_safety_direction.py" \
            --model "$M"; then
        STATUS[$M]="ok"
    else
        echo "  [error] compositional_safety_direction.py failed for $M."
        STATUS[$M]="comp_dir_failed"
    fi

    t1=$(date +%s)
    echo "  Elapsed: $((t1 - t0)) s"
done

echo
echo "============================================================"
echo " Summary  ($(date))"
echo "============================================================"
for M in "${MODELS_ARR[@]}"; do
    printf "  %-35s %s\n" "$M" "${STATUS[$M]:-not_run}"
done
echo "============================================================"
