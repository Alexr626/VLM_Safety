#!/usr/bin/env bash
# ============================================================
# Teammate-runnable CompShift evaluation pipeline.
#
# Runs the SAME experiment Alex ran on his 5080 (CompShift on
# mm_safetybench + figstep + mssbench, with two direction sources:
# mssbench_vl and mssbench_tt) but parameterised by --model so each
# teammate can take one of the supported models.
#
# Phases:
#   0. Migrate any cached artifacts that still live under the OLD
#      directory layout to the current layout. The old layouts were:
#          data/{bench}/activations/{model}/sample_*_*.npz
#          data/{bench}/{model}/vanilla/responses.json
#      The current layouts are:
#          data/{bench}/{model}/activations/sample_*_*.npz
#          data/{bench}/{model}/responses/vanilla/responses.json
#      No data is regenerated if old caches are present — we just move them.
#
#   1. Captions for all 4 benchmarks (committed to git → typically present
#      after `git pull`). Local LLaVA-1.5-7B is the default captioner;
#      override with CAPTION_PROVIDER=anthropic CAPTION_MODEL=claude-sonnet-4-6.
#
#   2. TT activations for {chosen model} × {3 eval benchmarks}. Skip-if-exists.
#      VL activations are NOT generated — CompShift does live VL prefill at
#      eval time and doesn't read cached VL.
#
#   3. CompShift evaluation (mssbench_vl AND mssbench_tt direction sources)
#      against vanilla + adashield_s on 3 eval benchmarks.
#
# All phases are idempotent. Safe to interrupt and rerun.
# ============================================================
# Usage:
#   MODEL=Lin-Chen/ShareGPT4V-7B \
#       bash evaluation/scripts/run_compshift_teammate.sh
#
#   MODEL=Qwen/Qwen-VL-Chat \
#       bash evaluation/scripts/run_compshift_teammate.sh
#
#   # Overnight:
#   mkdir -p logs
#   MODEL=Lin-Chen/ShareGPT4V-7B \
#       nohup bash evaluation/scripts/run_compshift_teammate.sh \
#       > logs/compshift_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
#   # Skip a phase:
#   SKIP_MIGRATE=1 SKIP_CAPTIONS=1 SKIP_TT=1 \
#       MODEL=... bash evaluation/scripts/run_compshift_teammate.sh
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# ── Required arg ────────────────────────────────────────────────────────────
MODEL="${MODEL:-}"
if [[ -z "$MODEL" ]]; then
    echo "ERROR: MODEL env var required."
    echo "  e.g. MODEL=Lin-Chen/ShareGPT4V-7B  bash $0"
    echo "       MODEL=Qwen/Qwen-VL-Chat        bash $0"
    exit 1
fi

# ── Conda ───────────────────────────────────────────────────────────────────
if [[ "${CONDA_DEFAULT_ENV:-}" != "vlm_safety" ]]; then
    if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    fi
    conda activate vlm_safety
fi

# ── Knobs ───────────────────────────────────────────────────────────────────
SKIP_MIGRATE="${SKIP_MIGRATE:-0}"
SKIP_CAPTIONS="${SKIP_CAPTIONS:-0}"
SKIP_TT="${SKIP_TT:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

# Captioner backend. Default = local LLaVA-1.5-7B.
# Override with anthropic API:
#   CAPTION_PROVIDER=anthropic CAPTION_MODEL=claude-sonnet-4-6 bash ...
CAPTION_PROVIDER="${CAPTION_PROVIDER:-local}"
CAPTION_MODEL="${CAPTION_MODEL:-llava-hf/llava-1.5-7b-hf}"
CAPTION_BATCH_SIZE="${CAPTION_BATCH_SIZE:-1}"
# expandable_segments avoids fragmentation OOM on small (~16 GB) GPUs.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

# Benchmarks where we need cached TT activations (the 3 eval benchmarks).
TT_BENCHMARKS=(mm_safetybench figstep mssbench)
# All benchmarks where we want captions present (in case anyone runs
# diagnostic experiments later — captions are cheap if already in git).
CAPTION_BENCHMARKS=(holisafe mssbench mm_safetybench figstep)

# Resolve the model's short directory name (e.g. "sharegpt4v-7b").
MODEL_SHORT=$(python -c \
    "from src.model import _normalize_model_name; \
     print(_normalize_model_name('$MODEL'))")

# Cleanup trap: kill all child processes if Ctrl+C / SIGTERM / EXIT.
_cleanup() {
    local rc=$?
    pkill -P $$ 2>/dev/null || true
    exit $rc
}
trap _cleanup EXIT INT TERM

mkdir -p "$PROJECT_ROOT/logs"
sec() { date +%s; }
elapsed() { local s=$(( $(sec) - $1 )); printf "%02d:%02d:%02d" $((s/3600)) $(((s%3600)/60)) $((s%60)); }

T_TOTAL=$(sec)
echo "============================================================"
echo " CompShift teammate pipeline    $(date)"
echo "============================================================"
echo "  model:           $MODEL"
echo "  model short:     $MODEL_SHORT"
echo "  caption_provider: $CAPTION_PROVIDER ($CAPTION_MODEL)"
echo "  TT benchmarks:   ${TT_BENCHMARKS[*]}"
echo "  caption benchmarks: ${CAPTION_BENCHMARKS[*]}"
echo "  SKIP_MIGRATE=$SKIP_MIGRATE SKIP_CAPTIONS=$SKIP_CAPTIONS \
SKIP_TT=$SKIP_TT SKIP_EVAL=$SKIP_EVAL"

# ── Phase 0: Migrate old-layout caches ────────────────────────────────────
# OLD activations:  data/{bench_dir}/activations/{model_short}/sample_*.npz
# NEW activations:  data/{bench_dir}/{model_short}/activations/sample_*.npz
#
# OLD responses:    data/{bench_dir}/{model_short}/vanilla/responses.json
# NEW responses:    data/{bench_dir}/{model_short}/responses/vanilla/responses.json
if [[ "$SKIP_MIGRATE" != "1" ]]; then
    echo
    echo "=== [0/3] Migrate old-layout caches (no data regenerated) ==="
    BENCH_DIRS=(holisafe-bench mssbench mm-safetybench figstep)
    n_moved=0
    for BENCH_DIR in "${BENCH_DIRS[@]}"; do
        # ── Activations ─────────────────────────────────────────────────────
        OLD_ACT="data/$BENCH_DIR/activations/$MODEL_SHORT"
        NEW_ACT="data/$BENCH_DIR/$MODEL_SHORT/activations"
        if [[ -d "$OLD_ACT" ]]; then
            mkdir -p "$NEW_ACT"
            # Move every file individually; keep new files if both exist.
            n_files=0
            for f in "$OLD_ACT"/*; do
                [[ -e "$f" ]] || continue
                target="$NEW_ACT/$(basename "$f")"
                if [[ ! -e "$target" ]]; then
                    mv "$f" "$target"
                    n_files=$((n_files + 1))
                fi
            done
            # Remove the now-empty old dir
            rmdir "$OLD_ACT" 2>/dev/null || true
            # Try to remove the parent if empty too (e.g. data/{bench}/activations/)
            rmdir "data/$BENCH_DIR/activations" 2>/dev/null || true
            if [[ $n_files -gt 0 ]]; then
                echo "  $BENCH_DIR/activations:  moved $n_files files → $NEW_ACT"
                n_moved=$((n_moved + n_files))
            fi
        fi
        # ── Responses (vanilla only — old layout had no 'responses/' level) ─
        OLD_RESP="data/$BENCH_DIR/$MODEL_SHORT/vanilla/responses.json"
        NEW_RESP="data/$BENCH_DIR/$MODEL_SHORT/responses/vanilla/responses.json"
        if [[ -f "$OLD_RESP" && ! -f "$NEW_RESP" ]]; then
            mkdir -p "$(dirname "$NEW_RESP")"
            mv "$OLD_RESP" "$NEW_RESP"
            rmdir "data/$BENCH_DIR/$MODEL_SHORT/vanilla" 2>/dev/null || true
            echo "  $BENCH_DIR/responses:  moved $OLD_RESP → $NEW_RESP"
            n_moved=$((n_moved + 1))
        fi
    done
    echo "  Total files migrated: $n_moved"
else
    echo "[SKIP] Phase 0: migration"
fi

# ── Phase 1: Captions (if missing) ────────────────────────────────────────
if [[ "$SKIP_CAPTIONS" != "1" ]]; then
    echo
    echo "=== [1/3] Captions (skip-if-exists) ==="
    T1=$(sec)
    for BM in "${CAPTION_BENCHMARKS[@]}"; do
        CAP_FILE="data/captions/$BM.json"
        if [[ -f "$CAP_FILE" ]]; then
            n=$(python -c "import json; print(len(json.load(open('$CAP_FILE'))))")
            echo "  [$BM] $n captions already present at $CAP_FILE"
            continue
        fi
        echo "  [$BM] generating captions ($CAPTION_PROVIDER, $CAPTION_MODEL)..."
        if [[ "$CAPTION_PROVIDER" == "local" ]]; then
            python "$PROJECT_ROOT/data_scripts/generate_captions.py" \
                --dataset "$BM" \
                --provider local \
                --model "$CAPTION_MODEL" \
                --batch_size "$CAPTION_BATCH_SIZE"
        else
            python "$PROJECT_ROOT/data_scripts/generate_captions.py" \
                --dataset "$BM" \
                --provider "$CAPTION_PROVIDER" \
                --api_model "$CAPTION_MODEL"
        fi
    done
    echo "  Phase 1 elapsed: $(elapsed $T1)"
else
    echo "[SKIP] Phase 1: captions"
fi

# ── Phase 2: TT activations for the chosen model ──────────────────────────
if [[ "$SKIP_TT" != "1" ]]; then
    echo
    echo "=== [2/3] TT activations for $MODEL_SHORT (skip-if-exists) ==="
    T2=$(sec)
    for BM in "${TT_BENCHMARKS[@]}"; do
        echo
        echo "  --- TT: $MODEL_SHORT × $BM ---"
        python "$PROJECT_ROOT/data_scripts/extract_tt.py" \
            --model "$MODEL" --dataset "$BM"
    done
    echo "  Phase 2 elapsed: $(elapsed $T2)"
else
    echo "[SKIP] Phase 2: TT activations"
fi

# ── Phase 3: CompShift eval (two direction sources) ───────────────────────
if [[ "$SKIP_EVAL" != "1" ]]; then
    echo
    echo "=== [3/3] CompShift evaluation: vanilla + adashield_s + comp_safety_shift ==="
    echo "  direction sources: mssbench_vl, mssbench_tt"
    T3=$(sec)

    # Clear out stale CompShift results (from earlier, pre-fix runs) for ALL
    # supported models. Forces --skip_if_exists to regenerate them with the
    # corrected intervention math. Vanilla / adashield_s results are kept
    # since they don't depend on the CompShift fix.
    echo "  Removing stale comp_safety_* result directories..."
    rm -rf evaluation/results/qwen-vl-chat/mm_safetybench/comp_safety*
    rm -rf evaluation/results/qwen-vl-chat/figstep/comp_safety*
    rm -rf evaluation/results/qwen-vl-chat/mssbench/comp_safety*

    rm -rf evaluation/results/llava-1.5-7b-hf/mm_safetybench/comp_safety*
    rm -rf evaluation/results/llava-1.5-7b-hf/figstep/comp_safety*
    rm -rf evaluation/results/llava-1.5-7b-hf/mssbench/comp_safety*

    rm -rf evaluation/results/sharegpt4v-7b/mm_safetybench/comp_safety*
    rm -rf evaluation/results/sharegpt4v-7b/figstep/comp_safety*
    rm -rf evaluation/results/sharegpt4v-7b/mssbench/comp_safety*

    python "$PROJECT_ROOT/evaluation/run_eval.py" \
        --model "$MODEL" \
        --interventions comp_safety_shift vanilla adashield_s \
        --benchmarks mssbench mm_safetybench figstep \
        --comp_safety_sources mssbench_vl mssbench_tt \
        --skip_if_exists
    echo "  Phase 3 elapsed: $(elapsed $T3)"
else
    echo "[SKIP] Phase 3: evaluation"
fi

echo
echo "============================================================"
echo " Done. Total elapsed: $(elapsed $T_TOTAL)    $(date)"
echo "============================================================"
echo
echo "Results: evaluation/results/$MODEL_SHORT/{benchmark}/{intervention}/"
echo "  - vanilla/"
echo "  - adashield_s/"
echo "  - comp_safety_shift_mssbench_vl/"
echo "  - comp_safety_shift_mssbench_tt/"
