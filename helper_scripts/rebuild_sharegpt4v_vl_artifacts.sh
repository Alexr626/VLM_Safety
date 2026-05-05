#!/bin/bash
# ============================================================
# Rebuild every ShareGPT4V VL-derived artifact after the
# `_prepare_vl_embeds` image-splicing bug fix.
# ============================================================
#
# Background
# ----------
# Until the fix in `ShareGPT4VWrapper._prepare_vl_embeds`
# (src/model.py), every VL forward pass for ShareGPT4V silently
# dropped the image — the literal `<image>` placeholder was tokenized
# as BPE subwords, so the post-tokenization mask `input_ids == 32000`
# was always all-False and the image-feature splice never executed.
# Anything derived from a ShareGPT4V VL forward pass (per-sample VL
# activations, the `*_vl` compositional directions, vanilla and
# defended response generations) is therefore stale.
#
# This script regenerates them from scratch. Text-only artifacts (TT/
# CT activations, the CatQA-derived semantic safety direction `s^l`,
# and the `*_tt` compositional directions) are untouched.
#
# Phases (each can be skipped individually):
#   B  Re-extract VL activations across the 4 benchmarks (GPU).
#   C  Regenerate vanilla VL responses on the data/-side (GPU; slow).
#   D  Diagnostic experiments (multi-step):
#        D1  ShiftDC analysis (CPU; reuses existing s^l).
#        D2  Compositional Safety v2 (CPU after MSSBench data prep).
#        D3  Behavioral ground truth (GPU: response gen + classify).
#        D4  Causal mediation analysis (GPU).
#   E  Eval framework (GPU; vanilla + comp_safety_shift{4 sources} +
#      adashield_s × 3 benchmarks).
#   F  Re-run safety probes after Phase E so the MSSBench
#      behavioral test sets pick up freshly-generated eval responses.
#
# Defaults run all phases. Override with PHASES=B,C or SKIP_EVAL=1.
#
# Usage
# -----
#   bash helper_scripts/rebuild_sharegpt4v_vl_artifacts.sh
#   bash helper_scripts/rebuild_sharegpt4v_vl_artifacts.sh PHASES=B,C
#   bash helper_scripts/rebuild_sharegpt4v_vl_artifacts.sh SKIP_EVAL=1
#   bash helper_scripts/rebuild_sharegpt4v_vl_artifacts.sh BENCHMARKS=mssbench
#
# Resume-safe: every step calls `--skip_if_exists` / `--skip_extraction`
# where supported, so re-running after an interruption only redoes
# what's missing.
# ============================================================
set -e

MODEL="${MODEL:-Lin-Chen/ShareGPT4V-7B}"
BENCHMARKS="${BENCHMARKS:-holisafe,mssbench,mm_safetybench,figstep}"
PHASES="${PHASES:-B,C,D,E,F}"
SKIP_EVAL="${SKIP_EVAL:-0}"
COMP_SOURCES="${COMP_SOURCES:-holisafe_tt,holisafe_vl,mssbench_tt,mssbench_vl}"
EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-mm_safetybench,figstep,mssbench}"
INTERVENTIONS="${INTERVENTIONS:-vanilla,comp_safety_shift,adashield_s}"
for arg in "$@"; do eval "$arg"; done

# Comma → space normalisation.
BENCHMARKS_LIST="${BENCHMARKS//,/ }"
PHASES_LIST="${PHASES//,/ }"
COMP_SOURCES_LIST="${COMP_SOURCES//,/ }"
EVAL_BENCHMARKS_LIST="${EVAL_BENCHMARKS//,/ }"
INTERVENTIONS_LIST="${INTERVENTIONS//,/ }"

has_phase() {
    [[ " $PHASES_LIST " == *" $1 "* ]]
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DIAGNOSTIC_ROOT="$PROJECT_ROOT/diagnostic_experiments"
RUN_SCRIPTS="$DIAGNOSTIC_ROOT/run_scripts"
cd "$PROJECT_ROOT"

MODEL_NAME=$(python -c "from src.model import _normalize_model_name; print(_normalize_model_name('$MODEL'))")

echo "======================================================"
echo " Rebuild ShareGPT4V VL artifacts"
echo "   model       : $MODEL  (short=$MODEL_NAME)"
echo "   benchmarks  : $BENCHMARKS_LIST"
echo "   phases      : $PHASES_LIST"
echo "   skip eval   : $SKIP_EVAL"
echo "======================================================"

# ── Phase B: VL extraction =================================================
if has_phase B; then
    echo ""
    echo "##### [B] Extract VL activations (GPU) #####"
    for B in $BENCHMARKS_LIST; do
        echo "--- VL extraction for benchmark=$B ---"
        python data_scripts/extract_vl.py \
            --model "$MODEL" --dataset "$B" --skip_extraction
    done
    # Compositional probes also need HoliSafe USU/SUU/UUU eval-only VL.
    echo "--- HoliSafe SUU/USU/UUU eval VL (for compositional probes) ---"
    python data_scripts/extract_vl.py \
        --model "$MODEL" --dataset holisafe \
        --holisafe_eval_only --holisafe_subsets SUU USU UUU
fi

# ── Phase C: vanilla responses (data/-side) ===============================
if has_phase C; then
    echo ""
    echo "##### [C] Regenerate vanilla VL responses (GPU) #####"
    python data_scripts/prepare_data.py \
        --models "$MODEL" \
        --benchmarks $BENCHMARKS_LIST \
        --phases responses
fi

# ── Phase D: diagnostic experiments =======================================
if has_phase D; then
    echo ""
    echo "##### [D] Diagnostic experiments #####"

    # D1: ShiftDC analysis. Reuses CatQA-derived s^l (text-only,
    # unaffected by the bug). The projection results JSON / plots are
    # rebuilt against the fresh VL activations from Phase B.
    echo "--- [D1/4] ShiftDC analysis ---"
    bash "$RUN_SCRIPTS/run_shiftdc.sh" MODEL="$MODEL"

    # D2: Compositional safety v2. Recomputes all 4 c^l (the _vl ones
    # are restored; the _tt ones already exist and are no-ops where
    # idempotent). Also runs the 5-probe × ~12-test cross-evaluation.
    echo "--- [D2/4] Compositional safety v2 ---"
    bash "$RUN_SCRIPTS/run_compositional_safety_v2.sh" MODEL="$MODEL" \
        SKIP_DATA_PREP=1   # we already ran extract_vl/tt in Phase B

    # D3: Behavioral ground truth (responses + refusal classification).
    echo "--- [D3/4] Behavioral ground truth ---"
    bash "$RUN_SCRIPTS/run_behavioral_ground_truth.sh" MODEL="$MODEL" \
        CLASSIFY_METHOD=keyword

    # D4: Causal mediation (top + bottom + all tiers, 33%).
    echo "--- [D4/4] Causal mediation ---"
    bash "$RUN_SCRIPTS/run_causal_mediation.sh" MODEL="$MODEL"
fi

# ── Phase E: eval framework ===============================================
if has_phase E && [ "$SKIP_EVAL" = "0" ]; then
    echo ""
    echo "##### [E] Eval framework #####"
    python evaluation/run_eval.py \
        --model "$MODEL" \
        --interventions $INTERVENTIONS_LIST \
        --benchmarks $EVAL_BENCHMARKS_LIST \
        --comp_safety_sources $COMP_SOURCES_LIST \
        --skip_if_exists
elif [ "$SKIP_EVAL" = "1" ]; then
    echo ""
    echo "##### [E] SKIPPED (SKIP_EVAL=1) #####"
fi

# ── Phase F: re-run probes to pick up MSSBench refusal labels =============
if has_phase F; then
    echo ""
    echo "##### [F] Re-run probes + plots #####"
    # Now that evaluation/results/.../mssbench/vanilla/responses.json
    # exists, build_mssbench_refusal_labels.py can populate the
    # behavioral test sets that safety_probes.py consumes.
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
echo " Done."
echo " Verify by checking that"
echo "   data/mssbench/sharegpt4v-7b/activations/ contains *_vl.npz"
echo "   experiment_artifacts/sharegpt4v-7b/compositional_safety/mssbench_vl/"
echo "   evaluation/results/sharegpt4v-7b/{benchmark}/{intervention}/"
echo "   diagnostic_experiments/sharegpt4v-7b/causal_mediation/outputs/"
echo "======================================================"
