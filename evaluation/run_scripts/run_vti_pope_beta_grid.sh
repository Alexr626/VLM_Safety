#!/usr/bin/env bash
# Deliverable A (grid) — VTI textual-arm POPE reproduction over a beta grid.
#
# Sweeps the textual steering coefficient beta for the VTI textual interventions
# vs a single no_intervention baseline, on the pinned POPE subset across the four
# target models. Beta is the OUTERMOST loop variable: all models/splits are run
# for one beta before moving to the next, so a full, paper-comparable slice lands
# on disk after each beta. The default grid leads with beta=0.4 (the paper's VTI
# textual value) so comparable numbers appear first.
#
# Results per grid point: {OUTPUT_DIR}/{RUN_DATE}/{model}/pope_{split}/{iv}__b{beta}/
# The no_intervention baseline (no beta) is computed once up front.
#
# Single 48 GB A6000 assumed; one model in memory at a time (the model is
# reloaded per beta — accepted cost of beta-outermost ordering).
#
# Env knobs: LIMIT (samples/split), BETAS, MODELS, IVS, RUN_DATE, OUTPUT_DIR.
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash evaluation/run_scripts/run_vti_pope_beta_grid.sh
#   LIMIT=3000 bash evaluation/run_scripts/run_vti_pope_beta_grid.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export HF_HOME="${HF_HOME:-/data/romanus/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LIMIT="${LIMIT:-200}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation/results}"
RUN_DATE="${RUN_DATE:-$(date +%Y-%m-%d)}"
# Lead with 0.4 (paper VTI textual value) so comparable numbers come out first.
BETAS="${BETAS:-0.4 0.1 0.2 0.3 0.5 0.6 0.7 0.8 0.9 1.0}"

read -r -a MODELS <<< "${MODELS:-llava-hf/llava-1.5-7b-hf Qwen/Qwen-VL-Chat Qwen/Qwen2-VL-7B-Instruct Qwen/Qwen2.5-VL-7B-Instruct}"
read -r -a IVS <<< "${IVS:-vti_textual_additive_mlp vti_textual_additive_layer vti_textual_uniform_rotation_mlp}"
read -r -a BETA_ARR <<< "$BETAS"
SPLITS=(random popular adversarial)

# ── Guard: confirm the per-split subset still matches the pinned manifest ──────
echo "[guard] verifying pinned POPE eval ids ..."
python - <<'PY' || { echo "[guard] FAILED — pinned ids drifted; aborting."; exit 1; }
import json, sys
from src.dataset import combined_json_path
data = json.load(open(combined_json_path('pope')))
pin = json.load(open('data/pope/pinned_eval_ids.json'))
for split, ids in pin['splits'].items():
    ents = [e for e in data if e.get('category')==split or e.get('task')==split]
    cur = [e['id'] for e in ents[:pin['limit_per_split']]]
    if cur != ids:
        print(f"  MISMATCH on split {split}", file=sys.stderr); sys.exit(1)
print("  ok: subset matches pinned manifest")
PY

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

echo "=== VTI POPE beta-grid reproduction (beta-outermost) ==="
echo "  models : ${MODELS[*]}"
echo "  ivs    : ${IVS[*]}"
echo "  betas  : ${BETAS}"
echo "  limit  : $LIMIT   output: $OUTPUT_DIR   date: $RUN_DATE   gpu: $CUDA_VISIBLE_DEVICES"

# ── Phase 0: baselines once (no beta) ─────────────────────────────────────────
echo ""
echo "########## baselines (no_intervention) ##########"
for MODEL in "${MODELS[@]}"; do
  for split in "${SPLITS[@]}"; do
    echo "--- $MODEL | POPE $split | no_intervention ---"
    python evaluation/run_eval.py \
      --model "$MODEL" --benchmarks pope --interventions no_intervention \
      --pope_split "$split" --limit "$LIMIT" \
      --output_dir "$OUTPUT_DIR" --run_date "$RUN_DATE" --skip_if_exists \
      || echo "[warn] baseline failed for $MODEL / $split (continuing)"
  done
done

# ── Phase 1: beta grid (beta outermost) ───────────────────────────────────────
for BETA in "${BETA_ARR[@]}"; do
  echo ""
  echo "############ beta=$BETA ############"
  for MODEL in "${MODELS[@]}"; do
    for split in "${SPLITS[@]}"; do
      echo "--- $MODEL | POPE $split | beta=$BETA ---"
      python evaluation/run_eval.py \
        --model "$MODEL" --benchmarks pope --interventions "${IVS[@]}" \
        --pope_split "$split" --limit "$LIMIT" --beta "$BETA" \
        --output_dir "$OUTPUT_DIR" --run_date "$RUN_DATE" --skip_if_exists \
        || echo "[warn] run_eval failed for $MODEL / $split / beta=$BETA (continuing)"
    done
  done
  echo "=== beta=$BETA complete across all models; tables so far: ==="
  print_tables
done

echo ""
echo "=== beta-grid reproduction complete ==="
