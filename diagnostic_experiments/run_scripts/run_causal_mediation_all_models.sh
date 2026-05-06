#!/bin/bash
# ============================================================
# Causal Mediation: similarity once, then sweep across models
# ============================================================
# Order: LLaVA-1.5-7B  →  ShareGPT4V-7B  →  Qwen-VL-Chat
#
# DINOv2 similarity is model-independent and runs once up front.
# Each model then runs `run_causal_mediation.sh` with SKIP_SIMILARITY=1
# so the per-model loop only does the mediation sweep + RR plots.
#
# Usage:
#   bash run_causal_mediation_all_models.sh
#   bash run_causal_mediation_all_models.sh TIER_PCT=25
#   bash run_causal_mediation_all_models.sh TIERS=top,bottom
#   bash run_causal_mediation_all_models.sh MODELS=llava-hf/llava-1.5-7b-hf
# ============================================================
set -e

TIER_PCT="${TIER_PCT:-10}"
TIERS="${TIERS:-top,bottom,all}"       # comma-separated (no spaces)
CORRUPT_MODE="${CORRUPT_MODE:-paired_ssu}"  # paired_ssu | random_ssu | blank
PREFLIGHT_N="${PREFLIGHT_N:-30}"
DTYPE="${DTYPE:-float16}"
LIMIT="${LIMIT:-}"
MODELS="${MODELS:-llava-hf/llava-1.5-7b-hf,Lin-Chen/ShareGPT4V-7B,Qwen/Qwen-VL-Chat}"  # comma-separated
for arg in "$@"; do eval "$arg"; done

# Normalize comma-separated lists to space-separated for iteration.
TIERS="${TIERS//,/ }"
MODELS="${MODELS//,/ }"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIAGNOSTIC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$DIAGNOSTIC_ROOT/.." && pwd)"
SCORES_PATH="$PROJECT_ROOT/data/mssbench/image_similarity/dinov2_similarity_scores.json"

echo "======================================================"
echo " Causal Mediation — multi-model run"
echo " models : $MODELS"
echo " tiers  : $TIERS    tier_pct : $TIER_PCT"
echo " dtype  : $DTYPE    limit    : ${LIMIT:-<none>}"
echo "======================================================"

# 1. DINOv2 similarities + similarity plots — once, model-independent.
if [ ! -f "$SCORES_PATH" ]; then
    echo "=== Compute DINOv2 image similarity (one-shot) ==="
    python "$DIAGNOSTIC_ROOT/experiment_scripts/compute_image_similarity.py" \
        --dataset mssbench
else
    echo "=== Image similarity already at $SCORES_PATH (skipped) ==="
fi
python "$DIAGNOSTIC_ROOT/plotting_scripts/plot_image_similarity.py" \
    --scores "$SCORES_PATH"
python "$DIAGNOSTIC_ROOT/plotting_scripts/plot_image_similarity.py" \
    --scores "$SCORES_PATH" --train_only

# 2. Per-model mediation + RR plots.
for MODEL in $MODELS; do
    echo ""
    echo "######################################################"
    echo " MODEL: $MODEL"
    echo "######################################################"
    EXTRA_ARGS=()
    EXTRA_ARGS+=("MODEL=$MODEL")
    EXTRA_ARGS+=("TIER_PCT=$TIER_PCT")
    EXTRA_ARGS+=("TIERS=${TIERS// /,}")
    EXTRA_ARGS+=("PREFLIGHT_N=$PREFLIGHT_N")
    EXTRA_ARGS+=("DTYPE=$DTYPE")
    EXTRA_ARGS+=("CORRUPT_MODE=$CORRUPT_MODE")
    EXTRA_ARGS+=("SKIP_SIMILARITY=1")
    if [ -n "$LIMIT" ]; then
        EXTRA_ARGS+=("LIMIT=$LIMIT")
    fi
    bash "$SCRIPT_DIR/run_causal_mediation.sh" "${EXTRA_ARGS[@]}"
done

echo ""
echo "======================================================"
echo " All models done."
echo " Per-model outputs under:"
echo "   diagnostic_experiments/{model_short}/causal_mediation/outputs/"
echo "======================================================"
