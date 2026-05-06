#!/bin/bash
# ============================================================
# Causal Mediation on SIUO (text-swap variant)
# ============================================================
# For each model in MODELS, runs the SIUO mediation across each
# ablation in ABLATION_MODES, then plots a 3-panel RR curve per
# ablation plus an overlay across ablations.
#
# Defaults: LLaVA-1.5-7B + ShareGPT4V-7B; modes = none, random_unsafe, no_text.
#
# Usage:
#   bash run_siuo_mediation.sh
#   bash run_siuo_mediation.sh MODELS=llava-hf/llava-1.5-7b-hf
#   bash run_siuo_mediation.sh ABLATION_MODES=none,random_unsafe
#   bash run_siuo_mediation.sh LIMIT=2 ABLATION_MODES=none   # smoke test
# ============================================================
set -e

MODELS="${MODELS:-llava-hf/llava-1.5-7b-hf,Lin-Chen/ShareGPT4V-7B}"
ABLATION_MODES="${ABLATION_MODES:-none,random_unsafe,no_text}"
PREFLIGHT_N="${PREFLIGHT_N:-30}"
DTYPE="${DTYPE:-float16}"
LIMIT="${LIMIT:-}"
SKIP_PLOTS="${SKIP_PLOTS:-0}"
for arg in "$@"; do eval "$arg"; done

# Normalize comma-separated to space-separated.
MODELS_LIST="${MODELS//,/ }"
MODES_LIST="${ABLATION_MODES//,/ }"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIAGNOSTIC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$DIAGNOSTIC_ROOT/.." && pwd)"

LIMIT_FLAG=""
if [ -n "$LIMIT" ]; then
    LIMIT_FLAG="--limit $LIMIT"
fi

declare -A MODE_SUFFIX
MODE_SUFFIX[none]=""
MODE_SUFFIX[random_unsafe]="_random"
MODE_SUFFIX[no_text]="_no_text"

echo "======================================================"
echo " SIUO Causal Mediation"
echo "   models : $MODELS_LIST"
echo "   modes  : $MODES_LIST"
echo "   limit  : ${LIMIT:-<none>}    dtype: $DTYPE"
echo "======================================================"

for MODEL in $MODELS_LIST; do
    MODEL_NAME=$(cd "$PROJECT_ROOT" && python -c "from src.model import _normalize_model_name; print(_normalize_model_name('$MODEL'))")
    OUT_ROOT="$DIAGNOSTIC_ROOT/$MODEL_NAME/causal_mediation_siuo/outputs"
    echo ""
    echo "######################################################"
    echo " MODEL: $MODEL  (short=$MODEL_NAME)"
    echo "######################################################"

    RR_FILES=()
    for MODE in $MODES_LIST; do
        echo ""
        echo "=== SIUO mediation: model=$MODEL  ablation_mode=$MODE ==="
        python "$DIAGNOSTIC_ROOT/experiment_scripts/causal_mediation_siuo.py" \
            --model "$MODEL" \
            --ablation_mode "$MODE" \
            --preflight_n "$PREFLIGHT_N" \
            --torch_dtype "$DTYPE" \
            $LIMIT_FLAG
        SUFFIX="${MODE_SUFFIX[$MODE]:-}"
        RR_FILES+=("$OUT_ROOT/results/recovery_rates${SUFFIX}.json")
    done

    if [ "$SKIP_PLOTS" = "0" ]; then
        echo ""
        echo "=== Plotting recovery curves for $MODEL_NAME ==="
        python "$DIAGNOSTIC_ROOT/plotting_scripts/plot_causal_mediation.py" \
            --results "${RR_FILES[@]}" \
            --output_dir "$OUT_ROOT/results/plots"
    fi
done

echo ""
echo "======================================================"
echo " Done. Per-model outputs under:"
echo "   diagnostic_experiments/{model_short}/causal_mediation_siuo/outputs/"
echo "======================================================"
