#!/bin/bash
# =====================================================================
# Get MSSBench_VL compositional-safety eval results — teammate edition
# =====================================================================
# What this does, end-to-end, for ONE model:
#   1. (one-time) MSSBench dataset + caption + activation prep
#   2. Recompute the pairwise CatQA semantic safety direction s^l
#      (the existing safety_direction_vectors.npz on disk is the old
#       joint-PCA version and won't be reused — see note below)
#   3. Compute the MSSBench_VL pairwise compositional safety direction c^l
#   4. Run comp_safety_shift refusal eval on all three benchmarks with
#      --comp_safety_sources mssbench_vl. vanilla / adashield_s are
#      NOT regenerated; their existing full-dataset responses are kept.
#   5. Post-hoc filter existing vanilla / adashield_s / old comp_safety_shift
#      MSSBench responses to the 304 eval-split ids and write
#      asr_summary_eval.json next to each so the comparison table can
#      score every method on the same 304 samples.
#   6. Print the comparison table (default --mssbench_view=eval).
#
# Prerequisites
# -------------
#   - Pulled the latest VLM_Safety branch.
#   - conda env vlm_safety activated.
#   - ANTHROPIC_API_KEY in .env (or set CAPTION_PROVIDER=local to use the VLM
#     itself for captions; slower but no API key required).
#   - Existing vanilla/ and adashield_s/ MSSBench results on disk (the
#     post-hoc step needs their responses.json files to filter).
#
# Usage
# -----
#   bash evaluation/scripts/run_mssbench_vl_pipeline_for_teammate.sh \
#       MODEL=Qwen/Qwen-VL-Chat
#   bash evaluation/scripts/run_mssbench_vl_pipeline_for_teammate.sh \
#       MODEL=Lin-Chen/ShareGPT4V-7B
#
#   # OpenAI captions:
#   CAPTION_PROVIDER=openai bash ... MODEL=Qwen/Qwen-VL-Chat
#
#   # If you've ALREADY recomputed the pairwise s^l on this workstation,
#   # skip step 2:
#   SKIP_SHIFTDC=1 bash ... MODEL=Qwen/Qwen-VL-Chat
#
# About the pairwise s^l rerun
# ----------------------------
# The repo recently switched the CatQA semantic direction from joint PCA
# to pairwise PCA (per-pair difference matrix on minimal-edit pairs).
# safety_direction_vectors.npz on disk pre-dates that change, so step 2
# deletes the old artifact dir and re-runs vl_activation_shift.py to
# produce the new pairwise s^l plus the recipe_sanity.json side-output.
# This is a CPU operation downstream of the existing reference activation
# matrices — no captioning, no extraction work. ~30s per model.
# =====================================================================

set -euo pipefail

# ---- Args ----------------------------------------------------------------
MODEL="${MODEL:-}"
CAPTION_PROVIDER="${CAPTION_PROVIDER:-anthropic}"
SKIP_SHIFTDC="${SKIP_SHIFTDC:-0}"
for arg in "$@"; do eval "$arg"; done

if [ -z "$MODEL" ]; then
    echo "ERROR: MODEL not set. Example:"
    echo "  bash $0 MODEL=Qwen/Qwen-VL-Chat"
    exit 1
fi

# ---- Paths ---------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$EVAL_ROOT/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_NAME=$(python -c "from src.model import _normalize_model_name; print(_normalize_model_name('$MODEL'))")
ARTIFACTS_DIR="$PROJECT_ROOT/experiment_artifacts/$MODEL_NAME"

echo "============================================================"
echo " MSSBench_VL pipeline — model=$MODEL ($MODEL_NAME)"
echo " caption provider: $CAPTION_PROVIDER"
echo "============================================================"

# ---- 1. MSSBench data prep ----------------------------------------------
echo ""
echo "=== [1/5] MSSBench dataset + caption + activation prep ==="

# Dataset — idempotent; download_mssbench.py is a no-op when files exist.
if [ ! -f "$PROJECT_ROOT/data/mssbench/combined.json" ]; then
    python "$EVAL_ROOT/scripts/download_mssbench.py"
fi

# Train/eval split (75/25 record-level, stratified by Type, seed=42).
# Deterministic — produces the same 304 eval ids on every machine.
python -m src.dataset --mssbench_split

# Captions (image-stem dedup → ~600 API calls, not 1200).
python "$PROJECT_ROOT/data_scripts/generate_captions.py" \
    --dataset mssbench --provider "$CAPTION_PROVIDER" --skip_if_exists

# VL + TT activations for the model.
python "$PROJECT_ROOT/data_scripts/extract_vl.py" \
    --model "$MODEL" --dataset mssbench
python "$PROJECT_ROOT/data_scripts/extract_tt.py" \
    --model "$MODEL" --dataset mssbench

# ---- 2. Recompute pairwise CatQA s^l ------------------------------------
if [ "$SKIP_SHIFTDC" = "1" ]; then
    echo ""
    echo "=== [2/5] SKIPPED (SKIP_SHIFTDC=1) ==="
else
    echo ""
    echo "=== [2/5] Recomputing pairwise CatQA semantic safety direction s^l ==="
    # The on-disk safety_direction_vectors.npz pre-dates the joint→pairwise
    # PCA refactor. Delete and recompute so downstream artifacts use the new
    # pairwise s^l (and gain the joint-sanity side-file + recipe_sanity.json).
    if [ -d "$ARTIFACTS_DIR/vl_activation_shift" ]; then
        echo "  removing old $ARTIFACTS_DIR/vl_activation_shift/"
        rm -rf "$ARTIFACTS_DIR/vl_activation_shift"
    fi
    python "$PROJECT_ROOT/diagnostic_experiments/experiment_scripts/vl_activation_shift.py" \
        --model "$MODEL"
fi

# ---- 3. MSSBench_VL pairwise compositional direction --------------------
echo ""
echo "=== [3/5] Compute MSSBench_VL compositional safety direction c^l ==="
python "$PROJECT_ROOT/diagnostic_experiments/experiment_scripts/compositional_safety_direction.py" \
    --model "$MODEL" --source mssbench --representation vl

# ---- 4. Refusal eval with comp_safety_shift × mssbench_vl ---------------
echo ""
echo "=== [4/5] comp_safety_shift refusal eval (mssbench_vl) ==="
# vanilla and adashield_s are intentionally omitted so their existing
# responses are not regenerated. Their eval-only ASR is added in step 5.
# --mssbench_eval_only is True by default (built-in), so this run only
# generates 304 MSSBench responses. mm_safetybench and figstep run full.
python "$EVAL_ROOT/run_eval.py" \
    --model "$MODEL" \
    --interventions comp_safety_shift \
    --benchmarks mm_safetybench figstep mssbench \
    --comp_safety_sources mssbench_vl \
    --skip_if_exists

# ---- 5. Post-hoc eval-only ASR for vanilla / adashield_s ----------------
echo ""
echo "=== [5/5] Post-hoc: write asr_summary_eval.json for existing methods ==="
python "$EVAL_ROOT/scripts/recompute_mssbench_eval_asr.py" --model "$MODEL"

# ---- Print the comparison table -----------------------------------------
echo ""
echo "============================================================"
echo " Done. Comparison table (mssbench_view=eval):"
echo "============================================================"
python -c "
import sys
sys.path.insert(0, '$PROJECT_ROOT')
from evaluation.runners.eval_runner import print_comparison_table
print_comparison_table('$MODEL_NAME', '$EVAL_ROOT/results', mssbench_view='eval')
"

echo ""
echo "Other views:"
echo "  python -c \"import sys; sys.path.insert(0,'$PROJECT_ROOT'); \\"
echo "    from evaluation.runners.eval_runner import print_comparison_table; \\"
echo "    print_comparison_table('$MODEL_NAME', '$EVAL_ROOT/results', mssbench_view='both')\""
