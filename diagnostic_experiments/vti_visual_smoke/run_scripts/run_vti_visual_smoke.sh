#!/usr/bin/env bash
# VTI visual-arm smoke (LLaVA-1.5). See vti_visual_discrepancies_and_smoke_plan.md
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export HF_HOME="${HF_HOME:-/data/romanus/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_DATE="${RUN_DATE:-$(date +%Y-%m-%d)}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation/results}"

echo "=== verify ViT layout ==="
python diagnostic_experiments/vti_visual_smoke/run_smoke.py --stage verify

echo "=== Stage 0: textual gate positive control ==="
python diagnostic_experiments/vti_visual_smoke/run_smoke.py \
  --stage 0 --run_date "$RUN_DATE" --output_dir "$OUTPUT_DIR" --skip_if_exists

echo "=== Stage 1: visual shakedown (paper + code cells) ==="
python diagnostic_experiments/vti_visual_smoke/run_smoke.py \
  --stage 1 --run_date "$RUN_DATE" --output_dir "$OUTPUT_DIR" --skip_if_exists

echo "=== Stage 1b: remaining visual variants ==="
python diagnostic_experiments/vti_visual_smoke/run_smoke.py \
  --stage 1b --run_date "$RUN_DATE" --output_dir "$OUTPUT_DIR" --skip_if_exists

echo "=== Gate summary ==="
python diagnostic_experiments/vti_visual_smoke/run_smoke.py \
  --stage analyze --run_date "$RUN_DATE" --output_dir "$OUTPUT_DIR"

echo "=== visual smoke driver complete ==="
