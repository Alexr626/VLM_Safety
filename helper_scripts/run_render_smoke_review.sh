#!/usr/bin/env bash
# Render self-contained HTML galleries for a VTI visual-smoke run.
#
# Output:
#   evaluation/results/{RUN_DATE}/_samples/{model}/{benchmark}/
#     {model}_{benchmark}_smoke_review.html
#
# Usage:
#   RUN_DATE=2026-07-02 bash helper_scripts/run_render_smoke_review.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_DATE="${RUN_DATE:-2026-07-02}"
SAMPLE_RUN_DATE="${SAMPLE_RUN_DATE:-2026-06-22}"
python helper_scripts/render_smoke_review.py \
  --run_date "$RUN_DATE" \
  --sample-run-date "$SAMPLE_RUN_DATE" \
  --output_dir "${OUTPUT_DIR:-evaluation/results}"
