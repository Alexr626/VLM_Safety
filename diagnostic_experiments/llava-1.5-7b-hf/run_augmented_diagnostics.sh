#!/bin/bash
# ============================================================
# Augmented Baseline Diagnostic Experiment (CPU-only Analysis)
# ============================================================
# Runs Experiment 1 (augmented baseline projections) and its
# plotting script.
#
# Prerequisites:
#   - run_shiftdc.sh completed
#   - run_data_prep.sh completed (CT activations)
#   - run_behavioral_ground_truth.sh completed (optional but
#     recommended — enables conditional behavioral analysis)
#
# GPU required: No (all CPU-based analysis)
#
# Usage:
#   bash diagnostic_experiments/llava-1.5-7b-hf/run_augmented_diagnostics.sh
# ============================================================
set -e

# ── Resolve directories ─────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARTIFACTS_DIR="$PROJECT_ROOT/experiment_artifacts/llava-1.5-7b-hf"

echo ""
echo "======================================================"
echo " Augmented Baseline Diagnostic Experiment"
echo "======================================================"

# ── Verify prerequisites ────────────────────────────────────
if [ ! -f "$ARTIFACTS_DIR/vl_activation_shift/safety_direction_vectors.npz" ]; then
    echo "ERROR: Safety direction vectors not found."
    echo "Run the ShiftDC pipeline first:"
    echo "  bash diagnostic_experiments/llava-1.5-7b-hf/shift_dc/run_shiftdc.sh"
    exit 1
fi

# Check for behavioral labels (optional)
BG_LABELS="$SCRIPT_DIR/behavioral_ground_truth/outputs/results/holisafe_refusal_labels.json"
if [ -f "$BG_LABELS" ]; then
    echo "  Behavioral labels found — conditional analysis enabled."
else
    echo "  Behavioral labels not found — conditional analysis will be skipped."
    echo "  Run run_behavioral_ground_truth.sh first for full analysis."
fi
echo ""

# ── Experiment 1: Augmented Baseline Projections ─────────────
echo "=== [1/1] Augmented baseline projections ==="
python "$SCRIPT_DIR/augmented_baseline/experiment_scripts/augmented_baseline_projections.py"
echo "  Plotting..."
python "$SCRIPT_DIR/augmented_baseline/plotting_scripts/plot_augmented_baseline.py"

echo ""
echo "======================================================"
echo " Augmented baseline diagnostic experiment complete."
echo " Results in:"
echo "   $SCRIPT_DIR/augmented_baseline/outputs/results/"
echo "======================================================"
