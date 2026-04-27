#!/usr/bin/env bash
# ============================================================
# Full end-to-end pipeline for a FRESH workstation.
#
# Produces every artifact that isn't checked into git, then runs
# the evaluation pipeline across all 5 supported models.
#
# What this builds (in order):
#   1. HoliSafe-Bench metadata          (auto-download from HF on first use)
#   2. Per-model upstream artifacts     (run_shiftdc.sh)
#        - HoliSafe SSS+SSU VL/TT activations    (GPU)
#        - CatQA reference activations            (GPU)
#        - CatQA-derived safety direction         (CPU)
#   3. Per-model compositional safety direction   (CPU)
#   4. Three evaluation benchmark datasets
#        - MM-SafetyBench  (HF: PKU-Alignment/MM-SafetyBench)
#        - FigStep         (GitHub: ThuCCSLab/FigStep)
#        - MSSBench        (HF: kzhou35/mssbench)
#   5. Evaluation: 5 models × 3 interventions × 3 benchmarks   (GPU)
#   6. Per-model ASR comparison tables to stdout
#
# Resumability: every step has skip-if-exists semantics, so if you
# interrupt the run and restart it, work that's already on disk is
# not redone (per-sample activation .npz, per-sample response
# checkpoints, downloaded datasets).
# ============================================================
# Recommended invocation (overnight, log to disk, survives logout):
#
#   mkdir -p logs
#   nohup bash evaluation/scripts/run_full_pipeline.sh \
#       > logs/full_pipeline_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   echo $! > logs/full_pipeline.pid
#
#   # tail progress
#   tail -f logs/full_pipeline_*.log
#
# Quick smoke test (single model, 5 samples per benchmark):
#   MODELS="llava-hf/llava-1.5-7b-hf" LIMIT=5 \
#       bash evaluation/scripts/run_full_pipeline.sh
#
# Override knobs:
#   MODELS            space-separated HF ids (default: all 5)
#   LIMIT             cap eval samples per benchmark (default: no cap)
#   CAPTION_PROVIDER  anthropic | openai | local (default: anthropic;
#                     only used during step 2 when HoliSafe captions are
#                     missing — they are committed to the repo, so this
#                     usually does nothing)
#   SKIP_DIRECTIONS   1 to skip phase 1 entirely. Use when the
#                     experiment_artifacts/ direction vectors are already
#                     committed to the repo and present on this workstation
#                     — the evaluation pipeline (phase 3) only reads
#                     compositional_safety_direction_vectors.npz, so phase 1
#                     becomes redundant. Default: 0 (run phase 1).
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── Conda ───────────────────────────────────────────────────────────────────
if [[ "${CONDA_DEFAULT_ENV:-}" != "vlm_safety" ]]; then
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        echo "ERROR: Could not find conda. Activate the vlm_safety env manually." >&2
        exit 1
    fi
    conda activate vlm_safety
fi

mkdir -p logs

# ── Knobs ───────────────────────────────────────────────────────────────────
DEFAULT_MODELS="llava-hf/llava-1.5-7b-hf Lin-Chen/ShareGPT4V-7B Qwen/Qwen-VL-Chat Qwen/Qwen2-VL-7B Qwen/Qwen2-VL-7B-Instruct"
MODELS="${MODELS:-$DEFAULT_MODELS}"
LIMIT="${LIMIT:-}"
CAPTION_PROVIDER="${CAPTION_PROVIDER:-anthropic}"
SKIP_DIRECTIONS="${SKIP_DIRECTIONS:-0}"

LIMIT_FLAG=""
if [[ -n "$LIMIT" ]]; then
    LIMIT_FLAG="--limit $LIMIT"
fi

# ── Helpers ─────────────────────────────────────────────────────────────────
phase() {
    echo
    echo "============================================================"
    echo " $1   $(date)"
    echo "============================================================"
}
sec() { date +%s; }
elapsed() { local s=$(( $(sec) - $1 )); printf "%02d:%02d:%02d" $((s/3600)) $(((s%3600)/60)) $((s%60)); }

T_TOTAL=$(sec)
echo "============================================================"
echo " Full Pipeline   $(date)"
echo "============================================================"
echo "  models          : $MODELS"
echo "  python          : $(which python)"
echo "  env             : ${CONDA_DEFAULT_ENV:-<none>}"
echo "  caption provider: $CAPTION_PROVIDER (only used if captions missing)"
echo "  eval LIMIT      : ${LIMIT:-<none>}"
echo "  SKIP_DIRECTIONS : $SKIP_DIRECTIONS"

# ── Phase 0: HoliSafe metadata pre-flight (cheap; auto-downloads if needed) ─
phase "[0/4] HoliSafe metadata"
T0=$(sec)
python -c "from src.dataset import load_holisafe; load_holisafe()"
echo "  elapsed: $(elapsed $T0)"

# ── Phase 1: Per-model upstream artifacts + compositional direction ─────────
phase "[1/4] Per-model upstream artifacts + compositional safety direction"
T1=$(sec)
if [[ "$SKIP_DIRECTIONS" == "1" ]]; then
    echo "  SKIP_DIRECTIONS=1 — verifying committed direction vectors are present..."
    MISSING=()
    for M in $MODELS; do
        SHORT=$(python -c "from src.model import _normalize_model_name; print(_normalize_model_name('$M'))")
        NPZ="$PROJECT_ROOT/experiment_artifacts/$SHORT/compositional_safety/compositional_safety_direction_vectors.npz"
        if [[ -f "$NPZ" ]]; then
            echo "    ✓ $M  →  $NPZ"
        else
            echo "    ✗ $M  →  $NPZ (missing)"
            MISSING+=("$M")
        fi
    done
    if [[ ${#MISSING[@]} -gt 0 ]]; then
        echo
        echo "  ERROR: comp direction vectors missing for: ${MISSING[*]}"
        echo "  Either commit + pull experiment_artifacts/, or rerun without"
        echo "  SKIP_DIRECTIONS=1 to regenerate them on this workstation."
        exit 1
    fi
    echo "  Skipping phase 1."
else
    MODELS="$MODELS" CAPTION_PROVIDER="$CAPTION_PROVIDER" \
        bash "$PROJECT_ROOT/diagnostic_experiments/run_scripts/run_overnight_comp_directions.sh"
fi
echo "  elapsed: $(elapsed $T1)"

# ── Phase 2: Download evaluation benchmarks ─────────────────────────────────
phase "[2/4] Download evaluation benchmarks"
T2=$(sec)
python "$PROJECT_ROOT/evaluation/scripts/download_mm_safetybench.py"
python "$PROJECT_ROOT/evaluation/scripts/download_figstep.py"
python "$PROJECT_ROOT/evaluation/scripts/download_mssbench.py"
echo "  elapsed: $(elapsed $T2)"

# ── Phase 3: Run evaluation ─────────────────────────────────────────────────
phase "[3/4] Evaluation (per model: 3 interventions × 3 benchmarks)"
T3=$(sec)
declare -A EVAL_STATUS
for M in $MODELS; do
    echo
    echo "-------------------------------------------------------------"
    echo " Evaluating $M    $(date)"
    echo "-------------------------------------------------------------"
    if MODEL="$M" LIMIT="$LIMIT" bash "$PROJECT_ROOT/evaluation/scripts/run_eval.sh"; then
        EVAL_STATUS[$M]="ok"
    else
        echo "  [error] run_eval.sh failed for $M (continuing with other models)"
        EVAL_STATUS[$M]="failed"
    fi
done
echo "  elapsed: $(elapsed $T3)"

# ── Phase 4: Print final comparison tables ──────────────────────────────────
phase "[4/4] Comparison tables"
for M in $MODELS; do
    python - <<PY
import sys
sys.path.insert(0, "$PROJECT_ROOT")
from src.model import _normalize_model_name
from evaluation.runners import print_comparison_table
print_comparison_table(_normalize_model_name("$M"), "evaluation/results")
PY
done

echo
echo "============================================================"
echo " Summary  ($(date))"
echo "============================================================"
echo "  total elapsed: $(elapsed $T_TOTAL)"
for M in $MODELS; do
    printf "  %-35s %s\n" "$M" "${EVAL_STATUS[$M]:-not_run}"
done
echo "============================================================"
