#!/usr/bin/env bash
# Step 0 — CHAIR max-new-tokens provenance check (run FIRST, gates the cap).
#
# Generates captions for 20 fixed COCO val2014 images on LLaVA-1.5
# no_intervention at caps 64 and 512 (verbatim VTI prompt), scores both with the
# CHAIR scorer, and writes a JSON + console table. Compare against the VTI paper
# LLaVA-1.5 baseline (CHAIR_S 51.0 / CHAIR_I 15.2 / Recall 75.2, max=512). Default
# cap is 64 regardless; only flag if 512 is dramatically closer and 64 far off.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash evaluation/chair_amber_diagnostics/run_scripts/run_step0_chair_cap.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export HF_HOME="${HF_HOME:-/data/romanus/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MODEL="${MODEL:-llava-hf/llava-1.5-7b-hf}"
RUN_DATE="${RUN_DATE:-$(date +%Y-%m-%d)}"

python evaluation/chair_amber_diagnostics/step0_chair_token_cap.py \
  --model "$MODEL" --num_images "${NUM_IMAGES:-20}" --caps ${CAPS:-64 512} \
  --seed "${SEED:-1234}" --run_date "$RUN_DATE"
