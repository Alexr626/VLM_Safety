#!/bin/bash
# ============================================================
# Subspace Analysis Experiments & Plotting
# ============================================================
# Runs all subspace analysis experiments and their corresponding
# plots after the diagnostic ShiftDC pipeline has completed.
#
# Prerequisites:
#   - diagnostic_experiments/.../shift_dc/ pipeline has completed
#   - shift_dc/outputs/activations/ exists (VL + TT)
#   - shift_dc/outputs/results/sample_metadata.json exists
#   - shift_dc/outputs/artifacts/safety_direction_vectors.npz exists
#   - data/reference/ exists (safe + unsafe activation matrices)
#
# Usage:
#   bash run_followup.sh
# ============================================================
set -e

# ── Resolve directories ─────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="$SCRIPT_DIR/llava-1.5-7b-hf"

echo ""
echo "======================================================"
echo " Subspace Analysis Experiments & Plotting"
echo "   model dir: $MODEL_DIR"
echo "======================================================"

# ── Verify prerequisites ────────────────────────────────────
ARTIFACTS_DIR="$PROJECT_ROOT/experiment_artifacts/llava-1.5-7b-hf"
if [ ! -f "$ARTIFACTS_DIR/vl_activation_shift/safety_direction_vectors.npz" ]; then
    echo "ERROR: Shared artifacts not found at $ARTIFACTS_DIR/vl_activation_shift/"
    echo "Run the diagnostic pipeline first:"
    echo "  bash diagnostic_experiments/llava-1.5-7b-hf/shift_dc/run_shiftdc.sh"
    exit 1
fi

# ── Experiment A: Effective rank ────────────────────────────
echo ""
echo "=== [1/4] Experiment A: Effective rank ==="
python "$MODEL_DIR/effective_rank/experiment_scripts/experiment_a_effective_rank.py"
echo "  Plotting..."
python "$MODEL_DIR/effective_rank/plotting_scripts/plot_effective_rank.py"

# ── Experiment B: Safety decomposition ──────────────────────
echo ""
echo "=== [2/4] Experiment B: Safety decomposition ==="
python "$MODEL_DIR/safety_decomposition/experiment_scripts/experiment_b_safety_decomposition.py"
echo "  Plotting..."
python "$MODEL_DIR/safety_decomposition/plotting_scripts/plot_safety_decomposition.py"

# ── Experiment C: Subspace overlap ──────────────────────────
echo ""
echo "=== [3/4] Experiment C: Subspace overlap ==="
python "$MODEL_DIR/subspace_overlap/experiment_scripts/experiment_c_subspace_overlap.py"
echo "  Plotting..."
python "$MODEL_DIR/subspace_overlap/plotting_scripts/plot_subspace_overlap.py"

# ── Experiment D: Category analysis ─────────────────────────
echo ""
echo "=== [4/4] Experiment D: Per-category analysis ==="
python "$MODEL_DIR/category_analysis/experiment_scripts/experiment_d_category_analysis.py"
echo "  Plotting..."
python "$MODEL_DIR/category_analysis/plotting_scripts/plot_category_analysis.py"

echo ""
echo "======================================================"
echo " All experiments complete. Results in:"
echo "   $MODEL_DIR/effective_rank/outputs/results/"
echo "   $MODEL_DIR/safety_decomposition/outputs/results/"
echo "   $MODEL_DIR/subspace_overlap/outputs/results/"
echo "   $MODEL_DIR/category_analysis/outputs/results/"
echo "======================================================"
