#!/usr/bin/env bash
# ============================================================
# RTX 5090 box: VL activation extractions + vanilla model responses.
# Both run sequentially per (model, benchmark). Pure GPU work.
#
# Pair this with run_fill_artifacts_5080.sh on the other box.
# Captions are NOT needed by either step here (VL extraction is
# image+text → activations; vanilla responses are image+text →
# greedy generation). So this script is fully independent of
# the 5080 box's caption progress.
# ============================================================
# Usage:
#   bash data_scripts/run_fill_artifacts_5090.sh
#
#   # Overnight via nohup:
#   mkdir -p logs
#   nohup bash data_scripts/run_fill_artifacts_5090.sh \
#       > logs/fill_5090_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
#   # Skip a phase:
#   SKIP_VL=1        bash data_scripts/run_fill_artifacts_5090.sh
#   SKIP_RESPONSES=1 bash data_scripts/run_fill_artifacts_5090.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ "${CONDA_DEFAULT_ENV:-}" != "vlm_safety" ]]; then
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
    conda activate vlm_safety
fi

SKIP_VL="${SKIP_VL:-0}"
SKIP_RESPONSES="${SKIP_RESPONSES:-0}"

MODELS=(
    "llava-hf/llava-1.5-7b-hf"
    "Lin-Chen/ShareGPT4V-7B"
    "Qwen/Qwen-VL-Chat"
)
BENCHMARKS=(mm_safetybench figstep mssbench holisafe)

mkdir -p "$PROJECT_ROOT/logs"

sec() { date +%s; }
elapsed() { local s=$(( $(sec) - $1 )); printf "%02d:%02d:%02d" $((s/3600)) $(((s%3600)/60)) $((s%60)); }

T_TOTAL=$(sec)
echo "============================================================"
echo " 5090 box: VL activations + vanilla responses    $(date)"
echo "============================================================"
echo "  models:     ${MODELS[*]}"
echo "  benchmarks: ${BENCHMARKS[*]}"
echo "  SKIP_VL=$SKIP_VL  SKIP_RESPONSES=$SKIP_RESPONSES"

# ── VL activations (GPU) ───────────────────────────────────────────────────
if [[ "$SKIP_VL" != "1" ]]; then
    echo
    echo "============================================================"
    echo " VL Activations    $(date)"
    echo "============================================================"
    T_VL=$(sec)
    for BM in "${BENCHMARKS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            echo
            echo "--- VL: $MODEL × $BM ---"
            python "$PROJECT_ROOT/data_scripts/extract_vl.py" \
                --model "$MODEL" --dataset "$BM"
        done
    done
    echo "  VL activations elapsed: $(elapsed $T_VL)"
else
    echo "[SKIP] VL activations"
fi

# ── Vanilla responses (GPU) ────────────────────────────────────────────────
if [[ "$SKIP_RESPONSES" != "1" ]]; then
    echo
    echo "============================================================"
    echo " Vanilla Responses    $(date)"
    echo "============================================================"
    T3=$(sec)
    python "$PROJECT_ROOT/data_scripts/prepare_data.py" \
        --benchmarks "${BENCHMARKS[@]}" \
        --models "${MODELS[@]}" \
        --phases responses \
        --max_new_tokens 256
    echo "  Responses elapsed: $(elapsed $T3)"
else
    echo "[SKIP] Vanilla responses"
fi

echo
echo "============================================================"
echo " 5090 box done. Total elapsed: $(elapsed $T_TOTAL)   $(date)"
echo "============================================================"
