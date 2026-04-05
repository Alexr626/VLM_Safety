#!/bin/bash
# ============================================================
# Combinatorial Safety Diagnostic Experiment (CPU-only Analysis)
# ============================================================
# Runs Experiment 2:
#   - Combinatorial safety direction (SSU vs SSS PCA on TT)
#   - Safety probes (CatQA-trained vs SSU/SSS-trained)
# plus their plotting scripts.
#
# Prerequisites:
#   - run_shiftdc.sh completed
#   - run_data_prep.sh completed (CT activations)
#   - run_behavioral_ground_truth.sh completed (optional but
#     recommended — enables SSU behavioral probe evaluation)
#
# GPU required: No (all CPU-based analysis)
#
# Usage:
#   bash diagnostic_experiments/llava-1.5-7b-hf/run_combinatorial_safety.sh
# ============================================================
set -e

# ── Resolve directories ─────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARTIFACTS_DIR="$PROJECT_ROOT/experiment_artifacts/llava-1.5-7b-hf"

echo ""
echo "======================================================"
echo " Combinatorial Safety Diagnostic Experiment"
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
    echo "  Behavioral labels found — SSU behavioral probe evaluation enabled."
else
    echo "  Behavioral labels not found — SSU behavioral probe evaluation will be skipped."
    echo "  Run run_behavioral_ground_truth.sh first for full analysis."
fi
echo ""

# ── Experiment 2A: Combinatorial Safety Direction ────────────
echo "=== [1/2] Combinatorial safety direction ==="
python "$SCRIPT_DIR/combinatorial_safety/experiment_scripts/combinatorial_direction.py"
echo "  Plotting..."
python "$SCRIPT_DIR/combinatorial_safety/plotting_scripts/plot_direction_comparison.py"

# ── Experiment 2B: Safety Probes ─────────────────────────────
echo ""
echo "=== [2/2] Safety probes ==="
python "$SCRIPT_DIR/combinatorial_safety/experiment_scripts/safety_probes.py"
echo "  Plotting..."
python "$SCRIPT_DIR/combinatorial_safety/plotting_scripts/plot_probe_results.py"

echo ""
echo "======================================================"
echo " Combinatorial safety diagnostic experiment complete."
echo " Results in:"
echo "   $SCRIPT_DIR/combinatorial_safety/outputs/results/"
echo "======================================================"
