#!/usr/bin/env bash
# ============================================================
# RTX 5080 box: Captions + TT activation extractions.
# Captions run in background (CPU + Anthropic API);
# TT extraction runs once captions are complete (GPU).
#
# Pair this with run_fill_artifacts_5090.sh on the other box.
#
# Captions are committed to git, so when the 5090 box pulls,
# it gets them automatically. TT activations are gitignored
# but live under data/{benchmark}/{model}/activations/ and
# don't need to be on the 5090 (TT and VL extraction are
# independent — eval pipeline only reads VL+caption at runtime).
# ============================================================
# Usage:
#   bash data_scripts/run_fill_artifacts_5080.sh
#
#   # Overnight via nohup:
#   mkdir -p logs
#   nohup bash data_scripts/run_fill_artifacts_5080.sh \
#       > logs/fill_5080_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
#   # Skip a phase:
#   SKIP_CAPTIONS=1 bash data_scripts/run_fill_artifacts_5080.sh
#   SKIP_TT=1       bash data_scripts/run_fill_artifacts_5080.sh
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

SKIP_CAPTIONS="${SKIP_CAPTIONS:-0}"
SKIP_TT="${SKIP_TT:-0}"
CAPTION_PROVIDER="${CAPTION_PROVIDER:-anthropic}"
CAPTION_MODEL="${CAPTION_MODEL:-claude-sonnet-4-6}"

MODELS=(
    "llava-hf/llava-1.5-7b-hf"
    "Lin-Chen/ShareGPT4V-7B"
    "Qwen/Qwen-VL-Chat"
)
BENCHMARKS=(mm_safetybench figstep mssbench holisafe)

mkdir -p "$PROJECT_ROOT/logs"
CAPTION_LOG="$PROJECT_ROOT/logs/fill_captions_$$.log"

sec() { date +%s; }
elapsed() { local s=$(( $(sec) - $1 )); printf "%02d:%02d:%02d" $((s/3600)) $(((s%3600)/60)) $((s%60)); }

T_TOTAL=$(sec)
echo "============================================================"
echo " 5080 box: Captions + TT activations    $(date)"
echo "============================================================"
echo "  models:     ${MODELS[*]}"
echo "  benchmarks: ${BENCHMARKS[*]}"
echo "  caption:    $CAPTION_PROVIDER ($CAPTION_MODEL)"
echo "  SKIP_CAPTIONS=$SKIP_CAPTIONS  SKIP_TT=$SKIP_TT"

# ── Captions in background (CPU + API) ─────────────────────────────────────
CAPTION_PID=""
if [[ "$SKIP_CAPTIONS" != "1" ]]; then
    echo
    echo "Launching captions in background. Log: $CAPTION_LOG"
    (
        set -e
        T1=$(date +%s)
        for BM in "${BENCHMARKS[@]}"; do
            echo
            echo "--- Captioning: $BM  $(date) ---"
            python "$PROJECT_ROOT/data_scripts/generate_captions.py" \
                --dataset "$BM" \
                # --provider "$CAPTION_PROVIDER" \
                # --api_model "$CAPTION_MODEL"
        done
        elapsed_s=$(( $(date +%s) - T1 ))
        printf "\nCaptions done. Elapsed: %02d:%02d:%02d\n" \
            $((elapsed_s/3600)) $(((elapsed_s%3600)/60)) $((elapsed_s%60))
    ) > "$CAPTION_LOG" 2>&1 &
    CAPTION_PID=$!
    echo "Caption PID: $CAPTION_PID"
else
    echo "[SKIP] Captions"
fi

# ── Wait for captions before TT (TT needs captions) ─────────────────────────
if [[ -n "$CAPTION_PID" ]]; then
    echo
    echo "Waiting for captions (PID $CAPTION_PID) to finish..."
    if wait "$CAPTION_PID"; then
        echo "  Captions completed successfully."
    else
        echo "  [ERROR] Caption process failed. Check $CAPTION_LOG"
        echo "  TT extraction may fail for benchmarks missing captions."
    fi
    echo "  Caption log tail:"
    tail -5 "$CAPTION_LOG" | sed 's/^/    /'
fi

# ── TT activations (GPU) ───────────────────────────────────────────────────
if [[ "$SKIP_TT" != "1" ]]; then
    echo
    echo "============================================================"
    echo " TT Activations    $(date)"
    echo "============================================================"
    T_TT=$(sec)
    for BM in "${BENCHMARKS[@]}"; do
        for MODEL in "${MODELS[@]}"; do
            echo
            echo "--- TT: $MODEL × $BM ---"
            python "$PROJECT_ROOT/data_scripts/extract_tt.py" \
                --model "$MODEL" --dataset "$BM"
        done
    done
    echo "  TT activations elapsed: $(elapsed $T_TT)"
else
    echo "[SKIP] TT activations"
fi

echo
echo "============================================================"
echo " 5080 box done. Total elapsed: $(elapsed $T_TOTAL)   $(date)"
echo "============================================================"
echo
echo "Next: commit + push captions so the 5090 box can pull them."
echo "      Activations stay local (gitignored)."
