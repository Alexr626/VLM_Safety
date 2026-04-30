#!/bin/bash
# ============================================================
# Compositional Safety Diagnostic v2 — multi-source pipeline
# ============================================================
# For one MODEL, computes all four compositional direction vectors
# (HoliSafe TT/VL via joint PCA, MSSBench TT/VL via pairwise PCA),
# the cross-direction cosine matrix, and the expanded 5-probe ×
# 7-test cross-evaluation.
#
# Prerequisites
# -------------
#   1. run_shiftdc.sh has completed for $MODEL (writes the
#      pairwise CatQA s^l → safety_direction_vectors.npz).
#   2. MSSBench split exists (data/mssbench/train_eval_split.json).
#      Created automatically by step [1/4] if missing.
#   3. MSSBench captions + activations exist for $MODEL. Step [2/4]
#      populates them via the API captioner and the GPU extractors;
#      both are resume-safe (no-op when caches are warm).
#
# Usage
# -----
#   bash run_compositional_safety_v2.sh
#   MODEL="Qwen/Qwen2-VL-7B-Instruct" bash run_compositional_safety_v2.sh
#   CAPTION_PROVIDER=openai bash run_compositional_safety_v2.sh
#
# CPU-only after step [2/4] completes (the directions and probes
# are pure-numpy/sklearn).
# ============================================================
set -e

MODEL="${MODEL:-llava-hf/llava-1.5-7b-hf}"
CAPTION_PROVIDER="${CAPTION_PROVIDER:-anthropic}"
SKIP_DATA_PREP="${SKIP_DATA_PREP:-0}"
for arg in "$@"; do eval "$arg"; done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DIAGNOSTIC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$DIAGNOSTIC_ROOT/.." && pwd)"
MODEL_NAME=$(cd "$PROJECT_ROOT" && python -c "from src.model import _normalize_model_name; print(_normalize_model_name('$MODEL'))")
ARTIFACTS_DIR="$PROJECT_ROOT/experiment_artifacts/$MODEL_NAME"

echo "======================================================"
echo " Compositional Safety v2   model=$MODEL"
echo "======================================================"

if [ ! -f "$ARTIFACTS_DIR/vl_activation_shift/safety_direction_vectors.npz" ]; then
    echo "ERROR: Pairwise CatQA s^l not found at"
    echo "  $ARTIFACTS_DIR/vl_activation_shift/safety_direction_vectors.npz"
    echo "Run run_shiftdc.sh for $MODEL first."
    exit 1
fi

# ── [1/4] Splits ────────────────────────────────────────────
echo ""
echo "=== [1/4] HoliSafe + MSSBench splits ==="
python -m src.dataset --n_eval 175 --seed 42       # HoliSafe USU/SUU/UUU eval ids
python -m src.dataset --mssbench_split             # MSSBench 75/25 record-level

# ── [2/4] MSSBench data prep (GPU) ──────────────────────────
if [ "$SKIP_DATA_PREP" = "1" ]; then
    echo ""
    echo "=== [2/4] SKIPPED (SKIP_DATA_PREP=1) ==="
else
    echo ""
    echo "=== [2/4] MSSBench captions + VL/TT activations (GPU) ==="
    python "$PROJECT_ROOT/data_scripts/generate_captions.py" \
        --dataset mssbench --provider "$CAPTION_PROVIDER" --skip_if_exists
    python "$PROJECT_ROOT/data_scripts/extract_vl.py" \
        --model "$MODEL" --dataset mssbench
    python "$PROJECT_ROOT/data_scripts/extract_tt.py" \
        --model "$MODEL" --dataset mssbench

    # Also ensure HoliSafe USU/SUU/UUU activations exist for safety_probes.py
    # (needed by holisafe_eval_sss_vs_{usu,suu,uuu}_{tt,vl} test sets).
    python "$PROJECT_ROOT/data_scripts/extract_vl.py" \
        --model "$MODEL" --dataset holisafe \
        --holisafe_eval_only --holisafe_subsets SUU USU UUU
    python "$PROJECT_ROOT/data_scripts/extract_tt.py" \
        --model "$MODEL" --dataset holisafe \
        --holisafe_eval_only --holisafe_subsets SUU USU UUU
fi

# ── [3/4] Four compositional directions (CPU) ───────────────
echo ""
echo "=== [3/4] Compositional directions (4 sources, CPU) ==="
for SRC in holisafe mssbench; do
    for REP in tt vl; do
        echo "  --- source=$SRC representation=$REP ---"
        python "$DIAGNOSTIC_ROOT/experiment_scripts/compositional_safety_direction.py" \
            --model "$MODEL" --source "$SRC" --representation "$REP"
    done
done

echo ""
echo "  --- Cross-direction cosine matrix ---"
python "$DIAGNOSTIC_ROOT/experiment_scripts/compare_compositional_directions.py" \
    --model "$MODEL"

# ── [4/4] Probes + plots (CPU) ──────────────────────────────
echo ""
echo "=== [4/4] Probes + plots (CPU) ==="

# Behavioral refusal-label prep — populates the inputs that
# safety_probes.py needs for ssu_behavioral_{tt,vl} and
# mssbench_behavioral_{tt,vl}. Each step is a no-op (and gracefully skips)
# when its upstream source isn't available for this model:
#   - classify_responses.py needs holisafe_responses.json from the separate
#     behavioral_ground_truth/generate_responses.py run.
#   - build_mssbench_refusal_labels.py needs the eval pipeline's vanilla
#     MSSBench responses.json.
HOLISAFE_RESPONSES="$DIAGNOSTIC_ROOT/$MODEL_NAME/behavioral_ground_truth/outputs/results/holisafe_responses.json"
if [ -f "$HOLISAFE_RESPONSES" ]; then
    python "$DIAGNOSTIC_ROOT/experiment_scripts/classify_responses.py" \
        --model "$MODEL" --method keyword
else
    echo "  [skip classify_responses] $HOLISAFE_RESPONSES not found"
fi

MSSB_VANILLA_RESPONSES="$PROJECT_ROOT/evaluation/results/$MODEL_NAME/mssbench/vanilla/responses.json"
if [ -f "$MSSB_VANILLA_RESPONSES" ]; then
    python "$PROJECT_ROOT/helper_scripts/build_mssbench_refusal_labels.py" \
        --model "$MODEL"
else
    echo "  [skip build_mssbench_refusal_labels] $MSSB_VANILLA_RESPONSES not found"
fi

python "$DIAGNOSTIC_ROOT/experiment_scripts/safety_probes.py" --model "$MODEL"
python "$DIAGNOSTIC_ROOT/plotting_scripts/plot_probe_results.py" --model "$MODEL"
python "$DIAGNOSTIC_ROOT/plotting_scripts/plot_compositional_eval.py" --model "$MODEL"
python "$DIAGNOSTIC_ROOT/plotting_scripts/plot_behavioral_eval.py" --model "$MODEL"
python "$DIAGNOSTIC_ROOT/plotting_scripts/plot_direction_comparison.py" --model "$MODEL" || true

echo ""
echo "======================================================"
echo " Done. Outputs:"
echo "   - $ARTIFACTS_DIR/compositional_safety/{holisafe_tt,holisafe_vl,mssbench_tt,mssbench_vl}/"
echo "   - $DIAGNOSTIC_ROOT/$MODEL_NAME/compositional_safety/outputs/results/"
echo "======================================================"
