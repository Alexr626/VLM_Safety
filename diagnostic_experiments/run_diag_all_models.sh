#!/bin/bash
# ============================================================
# Run full diagnostic pipeline on every model under investigation
# ============================================================
# Runs ALL phases from run_all_diagnostics.sh for each of:
#   - llava-hf/llava-1.5-7b-hf
#   - Qwen/Qwen2-VL-7B
#   - Qwen/Qwen2-VL-7B-Instruct
#   - OpenGVLab/InternVL2-8B
#   - OpenGVLab/InternVL2_5-8B-MPO
#
# Assumes a GPU with enough VRAM for all phases (including Phase 2
# generation) on every model — use run_diag_16gb.sh / run_diag_24gb.sh
# if you need to split work across cards.
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Helper: run all phases ────────────────────────────────────
run_all_phases() {
    local MODEL="$1"
    echo ""
    echo "============================================================"
    echo "  Running ALL phases for: $MODEL"
    echo "============================================================"
    MODEL="$MODEL" bash "$SCRIPT_DIR/run_all_diagnostics.sh"
}

# ── LLaVA 1.5 ────────────────────────────────────────────────
run_all_phases "llava-hf/llava-1.5-7b-hf"

# ── Qwen 2 VL (base + Instruct, for direct comparison) ──────
run_all_phases "Qwen/Qwen2-VL-7B"
run_all_phases "Qwen/Qwen2-VL-7B-Instruct"

# ── InternVL2 ────────────────────────────────────────────────
run_all_phases "OpenGVLab/InternVL2-8B"

# ── InternVL2.5 MPO ──────────────────────────────────────────
run_all_phases "OpenGVLab/InternVL2_5-8B-MPO"

echo ""
echo "============================================================"
echo "  All-model diagnostic run complete."
echo "============================================================"
