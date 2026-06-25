#!/usr/bin/env bash
# Consolidate Exp1 + Exp2 results for a run_date into one markdown report
# (evaluation/results/{RUN_DATE}/_diagnostic_summary_chair_amber.md) and append a
# pointer to RESEARCH_LOG.md. Run after Experiment 1 and 2 (handles partial runs).
#
# Usage:
#   RUN_DATE=2026-06-22 bash evaluation/chair_amber_diagnostics/run_scripts/run_report.sh
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
RUN_DATE="${RUN_DATE:-$(date +%Y-%m-%d)}"
python evaluation/chair_amber_diagnostics/make_diagnostic_summary.py \
  --run_date "$RUN_DATE" --output_dir "${OUTPUT_DIR:-evaluation/results}"
