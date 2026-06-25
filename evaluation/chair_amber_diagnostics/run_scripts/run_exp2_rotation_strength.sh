#!/usr/bin/env bash
# Experiment 2 — CHAIR + AMBER rotation-strength sweep (layer site, swept beta).
#
# The vti_rotation_strength analog on the new benchmarks: sweeps beta for the
# layer-site uniform_rotation variant (the residual-stream site that collapses
# to empty generations at high beta on POPE) over the pinned CHAIR-500 /
# AMBER-450 subsets, for both models. Per (model, benchmark) it writes a single
# metrics-vs-beta sweep JSON with the empty/identical/changed collapse split,
# §C empty-caption handling (CHAIR), AMBER by_qtype + decision-flip accounting,
# and decode-only / skip-position-0 mitigation probes at the strongest beta.
#
# The collapse band is NOT hard-capped: the full grid is swept and the per-beta
# empty-fraction / n_unparsed reveal the usable band per (model, benchmark).
#
# Results: evaluation/results/{RUN_DATE}/{model_short}/{benchmark}_rotation_strength/
#            sweep_uniform_rotation_layer_n{N}.json
#
# Env knobs: BETAS MODELS BENCHMARKS RUN_DATE CUDA_VISIBLE_DEVICES MAX_NEW_TOKENS
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash evaluation/chair_amber_diagnostics/run_scripts/run_exp2_rotation_strength.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

export HF_HOME="${HF_HOME:-/data/romanus/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_DATE="${RUN_DATE:-$(date +%Y-%m-%d)}"
BETAS="${BETAS:-0.6 0.5 0.45 0.4 0.35 0.3 0.25 0.2 0.1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
CHAIR_PROMPT="${CHAIR_PROMPT:-Please Describe this image in detail.}"
# Optional Qwen2/2.5-VL visual-token cap (pixels). Unset => native resolution
# (some CHAIR/AMBER images then OOM; the driver drops those per-sample).
MAX_PIXELS="${MAX_PIXELS:-}"

CHAIR_SUBSET="data/chair/pinned_chair_500.json"
AMBER_SUBSET="data/amber/pinned_amber_disc_450.json"

read -r -a MODELS <<< "${MODELS:-llava-hf/llava-1.5-7b-hf Qwen/Qwen2.5-VL-7B-Instruct}"
read -r -a BENCHMARKS <<< "${BENCHMARKS:-chair amber}"
read -r -a BETA_ARR <<< "$BETAS"

# ── Guard: pinned subsets exist (draw once if missing) ────────────────────────
if [[ ! -f "$CHAIR_SUBSET" || ! -f "$AMBER_SUBSET" ]]; then
  echo "[prep] pinned subsets missing; drawing them now (seed=1234) ..."
  python evaluation/chair_amber_diagnostics/draw_subsets.py --seed 1234 \
    || { echo "[prep] FAILED to draw subsets; aborting."; exit 1; }
fi

echo "=== Experiment 2 — CHAIR + AMBER rotation-strength sweep (layer) ==="
echo "  models     : ${MODELS[*]}"
echo "  benchmarks : ${BENCHMARKS[*]}"
echo "  betas      : ${BETAS}"
echo "  max_pixels : ${MAX_PIXELS:-native (uncapped)}"
echo "  date       : $RUN_DATE   gpu: $CUDA_VISIBLE_DEVICES"

for MODEL in "${MODELS[@]}"; do
  for BENCH in "${BENCHMARKS[@]}"; do
    echo ""
    echo "--- $MODEL | $BENCH | uniform_rotation @ layer ---"
    PIX_ARG=(); [[ -n "$MAX_PIXELS" ]] && PIX_ARG=(--max_pixels "$MAX_PIXELS")
    python evaluation/chair_amber_diagnostics/rotation_strength_chair_amber.py \
      --model "$MODEL" \
      --benchmark "$BENCH" \
      --variant uniform_rotation \
      --betas "${BETA_ARR[@]}" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --chair_prompt "$CHAIR_PROMPT" \
      --run_date "$RUN_DATE" \
      "${PIX_ARG[@]}" \
      --skip_if_exists \
      || echo "[warn] sweep failed for $MODEL / $BENCH (continuing)"
  done
done

echo ""
echo "=== Experiment 2 complete ==="
