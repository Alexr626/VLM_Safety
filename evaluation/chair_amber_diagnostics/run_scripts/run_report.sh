#!/usr/bin/env bash
# Consolidate CHAIR + AMBER diagnostic results for a run_date into markdown + JSON.
#
# Reads whatever Exp1 grid cells and Exp2 sweep JSONs exist on disk for RUN_DATE
# and writes sibling reports (facts only — interpretation is the analyst's job):
#   evaluation/results/{RUN_DATE}/_diagnostic_summary_chair_amber.md
#   evaluation/results/{RUN_DATE}/_diagnostic_summary_chair_amber.json
#
# Also appends a pointer entry to RESEARCH_LOG.md unless NO_LOG=1.
# Robust to partial runs (missing cells are omitted from the report).
#
# Run after Experiment 1 (run_exp1_repro_grid.sh) and Experiment 2
# (run_exp2_rotation_strength.sh) have produced results for the same RUN_DATE.
#
# Env knobs: RUN_DATE OUTPUT_DIR NO_LOG
# Usage:
#   RUN_DATE=2026-06-22 bash evaluation/chair_amber_diagnostics/run_scripts/run_report.sh
#   RUN_DATE=2026-06-22 NO_LOG=1 bash evaluation/chair_amber_diagnostics/run_scripts/run_report.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

RUN_DATE="${RUN_DATE:-$(date +%Y-%m-%d)}"
OUTPUT_DIR="${OUTPUT_DIR:-evaluation/results}"

ARGS=(--run_date "$RUN_DATE" --output_dir "$OUTPUT_DIR")
if [[ "${NO_LOG:-0}" == "1" ]]; then
  ARGS+=(--no_log)
fi

echo "=== CHAIR+AMBER diagnostic report ==="
echo "  run_date : $RUN_DATE"
echo "  output   : $OUTPUT_DIR"
echo ""

python evaluation/chair_amber_diagnostics/make_diagnostic_summary.py "${ARGS[@]}"
