#!/usr/bin/env bash
# Deliverable A — VTI textual-arm reproduction on POPE.
#
# Runs the two defensible "VTI" reproductions vs the no_intervention baseline,
# on the pinned 200-per-split POPE subset, across the four target models:
#   - vti_textual_additive_mlp     (paper's described method, MLP site)
#   - vti_textual_additive_layer   (paper's described method, residual site)
#   - vti_textual_uniform_rotation_mlp (the authors' RELEASED code: renorm@MLP)
# no_intervention is included so Qwen2.5-VL (no baseline yet) gets one; existing
# baselines are skipped via --skip_if_exists.
#
# Single 48 GB A6000 assumed; one model in memory at a time (sequential).
# Qwen-VL-Chat direction extraction needs the full model on one device — on a
# dedicated 48 GB card device_map=auto will NOT offload, so this is satisfied.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash evaluation/run_scripts/run_vti_pope_repro.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export HF_HOME="${HF_HOME:-/data/romanus/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
LIMIT="${LIMIT:-200}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation/results}"
# Pin one run-date for the whole sweep so all models/splits land in the same
# evaluation/results/<date>/ tree and same-day restarts resume (skip_if_exists).
RUN_DATE="${RUN_DATE:-$(date +%Y-%m-%d)}"

MODELS=(
  "llava-hf/llava-1.5-7b-hf"
  "Qwen/Qwen-VL-Chat"
  "Qwen/Qwen2-VL-7B-Instruct"
  "Qwen/Qwen2.5-VL-7B-Instruct"
)
IVS=(
  no_intervention
  vti_textual_additive_mlp
  vti_textual_additive_layer
  vti_textual_uniform_rotation_mlp
)

# ── Guard: confirm the 200/split subset still matches the pinned manifest ──────
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
print("  ok: 200/split matches pinned manifest")
PY

echo "=== VTI POPE reproduction ==="
echo "  models : ${MODELS[*]}"
echo "  ivs    : ${IVS[*]}"
echo "  limit  : $LIMIT   output: $OUTPUT_DIR   gpu: $CUDA_VISIBLE_DEVICES"
echo "  date   : $RUN_DATE   (results -> $OUTPUT_DIR/$RUN_DATE/<model>/pope_<split>/<iv>)"

for MODEL in "${MODELS[@]}"; do
  echo ""
  echo "############ $MODEL ############"
  for split in random popular adversarial; do
    echo "--- $MODEL | POPE $split ---"
    python evaluation/run_eval.py \
      --model "$MODEL" \
      --benchmarks pope \
      --interventions "${IVS[@]}" \
      --pope_split "$split" \
      --limit "$LIMIT" \
      --output_dir "$OUTPUT_DIR" \
      --run_date "$RUN_DATE" \
      --skip_if_exists \
      || echo "[warn] run_eval failed for $MODEL / $split (continuing)"
  done
  python - "$MODEL" "$OUTPUT_DIR" "$RUN_DATE" <<'PY' || true
import sys
from evaluation.runners import print_comparison_table
from src.model import _normalize_model_name
print_comparison_table(_normalize_model_name(sys.argv[1]), sys.argv[2], run_date=sys.argv[3])
PY
done

echo ""
echo "=== reproduction run complete ==="
