#!/bin/bash
# ============================================================
# Run Augmented Diagnostic Experiments (Phases 0-3)
# ============================================================
# Usage:
#   bash run_all_new_experiments.sh
#   MODEL="OpenGVLab/InternVL2-8B" bash run_all_new_experiments.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ">>> Phase 0: Data Prep"
bash "$SCRIPT_DIR/run_data_prep.sh" "$@"

echo ">>> Phase 1: Behavioral Ground Truth"
bash "$SCRIPT_DIR/run_behavioral_ground_truth.sh" "$@"

echo ">>> Phase 2: Compositional Safety"
bash "$SCRIPT_DIR/run_compositional_safety.sh" "$@"

echo ">>> Phase 3: Augmented Baseline"
bash "$SCRIPT_DIR/run_augmented_diagnostics.sh" "$@"

echo "Done."
