#!/usr/bin/env bash
# Draw + pin the shared CHAIR-500 / AMBER-450 subsets (run ONCE before Exp1/Exp2).
# Deterministic for a fixed seed; both experiments read the pinned files. The
# Exp1/Exp2 scripts auto-run this if the pins are missing, so this is optional.
#
# Usage:
#   bash evaluation/chair_amber_diagnostics/run_scripts/run_prep_subsets.sh
#   SEED=1234 FORCE=1 bash .../run_prep_subsets.sh   # redraw/overwrite
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
FORCE_FLAG=""; [[ "${FORCE:-0}" == "1" ]] && FORCE_FLAG="--force"
python evaluation/chair_amber_diagnostics/draw_subsets.py \
  --seed "${SEED:-1234}" $FORCE_FLAG
