#!/bin/bash
# ============================================================
# Run All New Augmented Diagnostic Experiments
# ============================================================
# Top-level convenience script that runs everything in order:
#   1. Data preparation (GPU): cohesive text + CT extraction
#   2. Behavioral ground truth (GPU): response generation + classification
#   3. Augmented diagnostics (CPU): baseline projections, combinatorial
#      direction, safety probes, and all plots
#
# Prerequisites:
#   - run_shiftdc.sh already completed (base pipeline)
#
# Usage:
#   bash diagnostic_experiments/llava-1.5-7b-hf/run_all_new_experiments.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "######################################################"
echo "#  Running All New Augmented Diagnostic Experiments   #"
echo "######################################################"

# ── Phase 0: Data Preparation (GPU) ─────────────────────────
echo ""
echo ">>> Phase 0: Data Preparation"
bash "$SCRIPT_DIR/run_data_prep.sh" "$@"

# ── Phase 1: Behavioral Ground Truth (GPU) ───────────────────
echo ""
echo ">>> Phase 1: Behavioral Ground Truth"
bash "$SCRIPT_DIR/run_behavioral_ground_truth.sh" "$@"

# ── Phase 2: Augmented Baseline (CPU) ────────────────────────
echo ""
echo ">>> Phase 2: Augmented Baseline"
bash "$SCRIPT_DIR/run_augmented_diagnostics.sh" "$@"

# ── Phase 3: Combinatorial Safety (CPU) ──────────────────────
echo ""
echo ">>> Phase 3: Combinatorial Safety"
bash "$SCRIPT_DIR/run_combinatorial_safety.sh" "$@"

echo ""
echo "######################################################"
echo "#  All experiments complete!                          #"
echo "#                                                     #"
echo "#  Results:                                           #"
echo "#    behavioral_ground_truth/outputs/results/         #"
echo "#    augmented_baseline/outputs/results/              #"
echo "#    combinatorial_safety/outputs/results/            #"
echo "######################################################"
