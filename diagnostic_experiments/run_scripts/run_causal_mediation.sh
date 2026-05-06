#!/bin/bash
# ============================================================
# Causal Mediation Analysis on MSSBench paired SSS/SSU images
# ============================================================
# Pipeline:
#   1. DINOv2 image-pair similarity (model-independent; runs once).
#   2. Similarity distribution plots (for tier inspection).
#   3. Causal mediation sweep — runs separately on the top tier,
#      bottom tier, and all train-split stems by default.
#   4. 3-panel Recovery Rate plots + top/bottom overlay.
#
# Usage:
#   bash run_causal_mediation.sh
#   MODEL=Lin-Chen/ShareGPT4V-7B bash run_causal_mediation.sh
#   bash run_causal_mediation.sh MODEL=Qwen/Qwen-VL-Chat TIER_PCT=25
# ============================================================
set -e

MODEL="${MODEL:-Qwen/Qwen-VL-Chat}"
TIER_PCT="${TIER_PCT:-10}"
TIERS="${TIERS:-top,bottom,all}"       # comma-separated (no spaces)
CORRUPT_MODE="${CORRUPT_MODE:-paired_ssu}"  # paired_ssu | random_ssu | blank
PREFLIGHT_N="${PREFLIGHT_N:-30}"
SKIP_SIMILARITY="${SKIP_SIMILARITY:-0}"
SKIP_PLOTS="${SKIP_PLOTS:-0}"
DTYPE="${DTYPE:-float16}"
LIMIT="${LIMIT:-}"   # for smoke testing; pass LIMIT=2 to run only 2 pairs.
for arg in "$@"; do eval "$arg"; done

# Normalize comma-separated to space-separated for iteration.
TIERS="${TIERS//,/ }"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIAGNOSTIC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$DIAGNOSTIC_ROOT/.." && pwd)"
MODEL_NAME=$(cd "$PROJECT_ROOT" && python -c "from src.model import _normalize_model_name; print(_normalize_model_name('$MODEL'))")

OUT_ROOT="$DIAGNOSTIC_ROOT/$MODEL_NAME/causal_mediation/outputs"
SCORES_PATH="$PROJECT_ROOT/data/mssbench/image_similarity/dinov2_similarity_scores.json"

echo "======================================================"
echo " Causal Mediation   model=$MODEL  tier_pct=$TIER_PCT"
echo " tiers=$TIERS  corrupt_mode=$CORRUPT_MODE"
echo " out=$OUT_ROOT"
echo "======================================================"

# 1. DINOv2 similarities (model-independent; idempotent).
if [ "$SKIP_SIMILARITY" = "0" ] && [ ! -f "$SCORES_PATH" ]; then
    echo "=== [1/4] Compute DINOv2 image similarity ==="
    python "$DIAGNOSTIC_ROOT/experiment_scripts/compute_image_similarity.py" \
        --dataset mssbench \
        --skip_if_exists
else
    echo "=== [1/4] Image similarity already present at $SCORES_PATH (skipped) ==="
fi

# 2. Similarity plots (rerun cheap; useful for tier inspection).
if [ "$SKIP_PLOTS" = "0" ]; then
    echo "=== [2/4] Plot similarity distributions ==="
    python "$DIAGNOSTIC_ROOT/plotting_scripts/plot_image_similarity.py" \
        --scores "$SCORES_PATH"
    python "$DIAGNOSTIC_ROOT/plotting_scripts/plot_image_similarity.py" \
        --scores "$SCORES_PATH" --train_only
fi

# 3. Mediation sweeps.
LIMIT_FLAG=""
if [ -n "$LIMIT" ]; then
    LIMIT_FLAG="--limit $LIMIT"
fi

MODE_SUFFIX=""
if [ "$CORRUPT_MODE" = "random_ssu" ]; then
    MODE_SUFFIX="_random"
elif [ "$CORRUPT_MODE" = "blank" ]; then
    MODE_SUFFIX="_blank"
fi

RR_FILES=()
for TIER in $TIERS; do
    echo "=== [3/4] Causal mediation sweep — tier=$TIER corrupt_mode=$CORRUPT_MODE ==="
    EXTRA=""
    if [ "$TIER" != "all" ]; then
        EXTRA="--tier_pct $TIER_PCT"
    fi
    python "$DIAGNOSTIC_ROOT/experiment_scripts/causal_mediation_mssbench.py" \
        --model "$MODEL" \
        --similarity_scores "$SCORES_PATH" \
        --tier "$TIER" $EXTRA \
        --corrupt_mode "$CORRUPT_MODE" \
        --preflight_n "$PREFLIGHT_N" \
        --torch_dtype "$DTYPE" \
        $LIMIT_FLAG
    if [ "$TIER" = "all" ]; then
        RR_FILES+=("$OUT_ROOT/results/recovery_rates_all${MODE_SUFFIX}.json")
    else
        PCT_LBL=$(echo "$TIER_PCT" | tr '.' 'p')
        RR_FILES+=("$OUT_ROOT/results/recovery_rates_${TIER}${PCT_LBL}${MODE_SUFFIX}.json")
    fi
done

# 4. RR plots.
if [ "$SKIP_PLOTS" = "0" ]; then
    echo "=== [4/4] Plot recovery-rate curves ==="
    python "$DIAGNOSTIC_ROOT/plotting_scripts/plot_causal_mediation.py" \
        --results "${RR_FILES[@]}" \
        --output_dir "$OUT_ROOT/results/plots"
fi

echo "======================================================"
echo " Done. Outputs under $OUT_ROOT"
echo "======================================================"
