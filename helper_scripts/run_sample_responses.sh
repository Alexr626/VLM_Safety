#!/usr/bin/env bash
# Extract qualitative response samples + HTML review galleries for a run_date.
#
# Output: evaluation/results/{RUN_DATE}/_samples/{model}/{benchmark}/
#   {model}_{benchmark}_response_samples.md / .json
#   {model}_{benchmark}_review.html
#
# Usage:
#   RUN_DATE=2026-06-22 bash helper_scripts/run_sample_responses.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_DATE="${RUN_DATE:-$(date +%Y-%m-%d)}"
python helper_scripts/sample_responses.py \
  --run_date "$RUN_DATE" --output_dir "${OUTPUT_DIR:-evaluation/results}"
