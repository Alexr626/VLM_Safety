#!/bin/bash
# ============================================================
# Rebuild every Qwen-VL-Chat activation + direction artifact
# after the chat-template fix in QwenVLWrapper._prepare_vl/text.
# ============================================================
#
# Background
# ----------
# QwenVLWrapper._prepare_vl and _prepare_text did not apply the
# ChatML template (<|im_start|>...<|im_end|>) that model.chat()
# uses internally. All activations extracted via forward_vl /
# forward_text were in text-continuation mode rather than
# assistant-response mode. The fix uses make_context() to wrap
# every prompt in the chat template.
#
# Unlike the ShareGPT4V bug (VL-only), BOTH VL and TT activations
# are affected here — _prepare_text also lacked the template —
# so CatQA reference activations, safety directions, and all
# compositional directions must be recomputed.
#
# Vanilla and AdaShield-S eval responses are UNAFFECTED (they go
# through model.chat which already applies the template) and are
# preserved on disk.
#
# Phases:
#   B  Re-extract VL + TT activations + CatQA references (GPU).
#   C  Regenerate data/-side vanilla responses (GPU).
#   D  Diagnostic experiments (ShiftDC, behavioral, comp v2, mediation).
#   E  Eval framework — comp_safety_shift only (GPU).
#   F  Re-run probes after eval to pick up MSSBench refusal labels.
#
# Usage
# -----
#   bash helper_scripts/rebuild_qwen_vl_chat_artifacts.sh
#   bash helper_scripts/rebuild_qwen_vl_chat_artifacts.sh PHASES=B,C
#   bash helper_scripts/rebuild_qwen_vl_chat_artifacts.sh SKIP_EVAL=1
# ============================================================
set -e

MODEL="${MODEL:-Qwen/Qwen-VL-Chat}"
BENCHMARKS="${BENCHMARKS:-holisafe,mssbench,mm_safetybench,figstep}"
PHASES="${PHASES:-B,C,D,E,F}"
SKIP_EVAL="${SKIP_EVAL:-0}"
COMP_SOURCES="${COMP_SOURCES:-holisafe_tt,holisafe_vl,mssbench_tt,mssbench_vl}"
EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-mm_safetybench,figstep,mssbench}"
for arg in "$@"; do eval "$arg"; done

BENCHMARKS_LIST="${BENCHMARKS//,/ }"
PHASES_LIST="${PHASES//,/ }"
COMP_SOURCES_LIST="${COMP_SOURCES//,/ }"
EVAL_BENCHMARKS_LIST="${EVAL_BENCHMARKS//,/ }"

has_phase() { [[ " $PHASES_LIST " == *" $1 "* ]]; }

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIAGNOSTIC_ROOT="$PROJECT_ROOT/diagnostic_experiments"
RUN_SCRIPTS="$DIAGNOSTIC_ROOT/run_scripts"
cd "$PROJECT_ROOT"

MODEL_NAME=$(python -c "from src.model import _normalize_model_name; print(_normalize_model_name('$MODEL'))")

echo "======================================================"
echo " Rebuild Qwen-VL-Chat artifacts (chat-template fix)"
echo "   model       : $MODEL  (short=$MODEL_NAME)"
echo "   benchmarks  : $BENCHMARKS_LIST"
echo "   phases      : $PHASES_LIST"
echo "   skip eval   : $SKIP_EVAL"
echo "======================================================"

# ── Phase B: Extract VL + TT + CatQA refs ================================
if has_phase B; then
    echo ""
    echo "##### [B] Extract VL + TT activations + CatQA refs (GPU) #####"
    for B in $BENCHMARKS_LIST; do
        echo "--- VL extraction: benchmark=$B ---"
        python data_scripts/extract_vl.py \
            --model "$MODEL" --dataset "$B" --skip_extraction
        echo "--- TT extraction: benchmark=$B ---"
        python data_scripts/extract_tt.py \
            --model "$MODEL" --dataset "$B" --skip_extraction
    done
    # HoliSafe SUU/USU/UUU eval-only activations for probes.
    echo "--- HoliSafe SUU/USU/UUU eval VL+TT ---"
    python data_scripts/extract_vl.py \
        --model "$MODEL" --dataset holisafe \
        --holisafe_eval_only --holisafe_subsets SUU USU UUU
    python data_scripts/extract_tt.py \
        --model "$MODEL" --dataset holisafe \
        --holisafe_eval_only --holisafe_subsets SUU USU UUU
    # CatQA reference activations (text-only, but uses _prepare_text).
    echo "--- CatQA reference activations ---"
    python data_scripts/extract_ref_activations.py \
        --model "$MODEL" \
        --safe_ref catqa-harmless --unsafe_ref catqa-harmful
fi

# ── Phase C: Vanilla responses (data side) ================================
if has_phase C; then
    echo ""
    echo "##### [C] Regenerate vanilla VL responses (GPU) #####"
    python data_scripts/prepare_data.py \
        --models "$MODEL" \
        --benchmarks $BENCHMARKS_LIST \
        --phases responses
fi

# ── Phase D: Diagnostic experiments ======================================
if has_phase D; then
    echo ""
    echo "##### [D] Diagnostic experiments #####"

    echo "--- [D1/4] ShiftDC analysis ---"
    bash "$RUN_SCRIPTS/run_shiftdc.sh" MODEL="$MODEL"

    echo "--- [D2/4] Compositional safety v2 ---"
    bash "$RUN_SCRIPTS/run_compositional_safety_v2.sh" MODEL="$MODEL"

    echo "--- [D3/4] Behavioral ground truth ---"
    bash "$RUN_SCRIPTS/run_behavioral_ground_truth.sh" MODEL="$MODEL" \
        CLASSIFY_METHOD=keyword

    echo "--- [D4/4] Causal mediation ---"
    bash "$RUN_SCRIPTS/run_causal_mediation.sh" MODEL="$MODEL"
fi

# ── Phase E: Eval framework (comp_safety_shift only) =====================
if has_phase E && [ "$SKIP_EVAL" = "0" ]; then
    echo ""
    echo "##### [E] Eval framework (comp_safety_shift) #####"
    # vanilla + adashield_s already exist and --skip_if_exists preserves them.
    python evaluation/run_eval.py \
        --model "$MODEL" \
        --interventions vanilla comp_safety_shift adashield_s \
        --benchmarks $EVAL_BENCHMARKS_LIST \
        --comp_safety_sources $COMP_SOURCES_LIST \
        --skip_if_exists
elif [ "$SKIP_EVAL" = "1" ]; then
    echo ""
    echo "##### [E] SKIPPED (SKIP_EVAL=1) #####"
fi

# ── Phase F: Re-run probes with MSSBench refusal labels ===================
if has_phase F; then
    echo ""
    echo "##### [F] Re-run probes + plots #####"
    MSSB_VANILLA_JSON="$PROJECT_ROOT/evaluation/results/$MODEL_NAME/mssbench/vanilla/responses.json"
    if [ -f "$MSSB_VANILLA_JSON" ]; then
        python helper_scripts/build_mssbench_refusal_labels.py --model "$MODEL"
    fi
    python "$DIAGNOSTIC_ROOT/experiment_scripts/safety_probes.py" --model "$MODEL"
    python "$DIAGNOSTIC_ROOT/plotting_scripts/plot_probe_results.py" --model "$MODEL"
    python "$DIAGNOSTIC_ROOT/plotting_scripts/plot_compositional_eval.py" --model "$MODEL"
    python "$DIAGNOSTIC_ROOT/plotting_scripts/plot_behavioral_eval.py" --model "$MODEL"
fi

echo ""
echo "======================================================"
echo " Done. Verify:"
echo "   data/mssbench/$MODEL_NAME/activations/ (VL + TT)"
echo "   experiment_artifacts/$MODEL_NAME/"
echo "   diagnostic_experiments/$MODEL_NAME/"
echo "======================================================"
