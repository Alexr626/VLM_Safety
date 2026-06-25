#!/usr/bin/env bash
# Experiment 1 — CHAIR + AMBER reproduction grid (fixed-beta methods).
#
# The run_vti_pope_repro / beta_grid analog, extended to CHAIR (generative) and
# AMBER-discriminative, over the shared beta grid. Runs the three VTI methods
# that do NOT collapse at canonical beta:
#   - vti_textual_additive_mlp
#   - vti_textual_additive_layer
#   - vti_textual_uniform_rotation_mlp
# (uniform_rotation_LAYER is excluded here — it is Experiment 2.)
#
# Beta is the OUTERMOST loop: every model/benchmark is run at one beta before
# moving on, so a complete, paper-comparable slice lands after each beta. The
# no_intervention baseline is beta-independent and computed once per
# model/benchmark up front (skipped on resume via --skip_if_exists).
#
# Both benchmarks score the PINNED subsets (identical ids across all cells):
#   CHAIR : data/chair/pinned_chair_500.json  (500 COCO val2014 images)
#   AMBER : data/amber/pinned_amber_disc_450.json (150 ea existence/attr/relation)
# CHAIR uses the frozen cap (64, Step 0) + the verbatim VTI prompt. AMBER uses
# --amber_task discriminative. Greedy decoding + Policy-A native resolution are
# the wrapper defaults (unchanged here => frozen across conditions).
#
# Results per cell: {OUTPUT_DIR}/{RUN_DATE}/{model_short}/{chair|amber}/{iv}__b{beta}/
#
# Env knobs: LIMIT(unused-subset-pinned) BETAS MODELS IVS RUN_DATE OUTPUT_DIR CUDA_VISIBLE_DEVICES
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash evaluation/chair_amber_diagnostics/run_scripts/run_exp1_repro_grid.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

export HF_HOME="${HF_HOME:-/data/romanus/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation/results}"
RUN_DATE="${RUN_DATE:-$(date +%Y-%m-%d)}"
# Lead with 0.4 (paper's primary CHAIR setting) so comparable numbers come first.
BETAS="${BETAS:-0.4 0.1 0.2 0.3 0.5 0.6 0.7 0.8 0.9}"
CHAIR_PROMPT="${CHAIR_PROMPT:-Please Describe this image in detail.}"
CHAIR_CAP="${CHAIR_CAP:-64}"

CHAIR_SUBSET="data/chair/pinned_chair_500.json"
AMBER_SUBSET="data/amber/pinned_amber_disc_450.json"

read -r -a MODELS <<< "${MODELS:-llava-hf/llava-1.5-7b-hf Qwen/Qwen2.5-VL-7B-Instruct}"
read -r -a IVS <<< "${IVS:-vti_textual_additive_mlp vti_textual_additive_layer vti_textual_uniform_rotation_mlp}"
read -r -a BETA_ARR <<< "$BETAS"

# ── Guard: pinned subsets exist (draw once if missing) ────────────────────────
if [[ ! -f "$CHAIR_SUBSET" || ! -f "$AMBER_SUBSET" ]]; then
  echo "[prep] pinned subsets missing; drawing them now (seed=1234) ..."
  python evaluation/chair_amber_diagnostics/draw_subsets.py --seed 1234 \
    || { echo "[prep] FAILED to draw subsets; aborting."; exit 1; }
fi

# CHAIR eval for one model at an optional beta (no beta => baseline).
run_chair() {
  local model="$1"; local beta="${2:-}"
  local extra=(); [[ -n "$beta" ]] && extra=(--beta "$beta")
  local ivs=(); [[ -n "$beta" ]] && ivs=("${IVS[@]}") || ivs=(no_intervention)
  python evaluation/run_eval.py \
    --model "$model" --benchmarks chair --interventions "${ivs[@]}" \
    --subset_ids_file "$CHAIR_SUBSET" \
    --chair_prompt "$CHAIR_PROMPT" --chair_max_new_tokens "$CHAIR_CAP" \
    --output_dir "$OUTPUT_DIR" --run_date "$RUN_DATE" --skip_if_exists "${extra[@]}" \
    || echo "[warn] CHAIR run failed: $model beta=${beta:-baseline} (continuing)"
}

# AMBER discriminative eval for one model at an optional beta.
run_amber() {
  local model="$1"; local beta="${2:-}"
  local extra=(); [[ -n "$beta" ]] && extra=(--beta "$beta")
  local ivs=(); [[ -n "$beta" ]] && ivs=("${IVS[@]}") || ivs=(no_intervention)
  python evaluation/run_eval.py \
    --model "$model" --benchmarks amber --amber_task discriminative \
    --interventions "${ivs[@]}" --subset_ids_file "$AMBER_SUBSET" \
    --output_dir "$OUTPUT_DIR" --run_date "$RUN_DATE" --skip_if_exists "${extra[@]}" \
    || echo "[warn] AMBER run failed: $model beta=${beta:-baseline} (continuing)"
}

print_tables() {
  for MODEL in "${MODELS[@]}"; do
    python - "$MODEL" "$OUTPUT_DIR" "$RUN_DATE" <<'PY' || true
import sys
from evaluation.runners import print_comparison_table
from src.model import _normalize_model_name
print_comparison_table(_normalize_model_name(sys.argv[1]), sys.argv[2], run_date=sys.argv[3])
PY
  done
}

echo "=== Experiment 1 — CHAIR + AMBER reproduction grid (beta-outermost) ==="
echo "  models : ${MODELS[*]}"
echo "  ivs    : ${IVS[*]}"
echo "  betas  : ${BETAS}"
echo "  chair  : cap=$CHAIR_CAP  prompt=\"$CHAIR_PROMPT\"  subset=$CHAIR_SUBSET"
echo "  amber  : discriminative  subset=$AMBER_SUBSET"
echo "  date   : $RUN_DATE   output: $OUTPUT_DIR   gpu: $CUDA_VISIBLE_DEVICES"

# ── Phase 0: baselines once (no beta) ─────────────────────────────────────────
echo ""
echo "########## baselines (no_intervention) ##########"
for MODEL in "${MODELS[@]}"; do
  echo "--- $MODEL | CHAIR baseline ---";  run_chair "$MODEL"
  echo "--- $MODEL | AMBER baseline ---";  run_amber "$MODEL"
done

# ── Phase 1: beta grid (beta outermost) ───────────────────────────────────────
for BETA in "${BETA_ARR[@]}"; do
  echo ""
  echo "############ beta=$BETA ############"
  for MODEL in "${MODELS[@]}"; do
    echo "--- $MODEL | CHAIR | beta=$BETA ---";  run_chair "$MODEL" "$BETA"
    echo "--- $MODEL | AMBER | beta=$BETA ---";  run_amber "$MODEL" "$BETA"
  done
  echo "=== beta=$BETA complete; tables so far: ==="
  print_tables
done

echo ""
echo "=== Experiment 1 complete ==="
