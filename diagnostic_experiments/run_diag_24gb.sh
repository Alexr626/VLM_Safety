#!/bin/bash
# ============================================================
# Diagnostic experiments requiring >=24 GB VRAM (e.g. RTX 5090)
# ============================================================
# Runs:
#   - Phase 2 ONLY for Qwen2-VL-7B (base) and Qwen2-VL-7B-Instruct
#     (generation needs more VRAM than the 5080 can handle without
#     CPU offloading)
#   - ALL phases for InternVL2-8B and InternVL2.5-8B-MPO
#     (bf16 weights alone fill ~16 GB, so even extraction is slow
#     on a 16 GB card)
#
# Prerequisite: Phases 0, 1, 3, 4 for Qwen should already be
# complete — run run_diag_16gb.sh first.
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Helper: run Phase 2 only ─────────────────────────────────
run_phase2_only() {
    local MODEL="$1"
    echo ""
    echo "============================================================"
    echo "  Running Phase 2 only for: $MODEL"
    echo "============================================================"

    echo "=== Phase 2: Behavioral ground truth ==="
    python "$SCRIPT_DIR/experiment_scripts/generate_responses.py" --model "$MODEL" --skip_if_exists
    python "$SCRIPT_DIR/experiment_scripts/classify_responses.py" --model "$MODEL" \
        --method llm_twoaxis --provider anthropic
    python "$SCRIPT_DIR/experiment_scripts/catqa_behavioral_baseline.py" --model "$MODEL" \
        --method llm_twoaxis --provider anthropic --skip_if_exists
    python "$SCRIPT_DIR/plotting_scripts/plot_behavioral_ground_truth.py" --model "$MODEL"

    echo "=== Done (Phase 2): $MODEL ==="
}

# ── Helper: run all phases ────────────────────────────────────
run_all_phases() {
    local MODEL="$1"
    echo ""
    echo "============================================================"
    echo "  Running ALL phases for: $MODEL"
    echo "============================================================"
    MODEL="$MODEL" bash "$SCRIPT_DIR/run_all_diagnostics.sh"
}

# ── Qwen 2 VL (base + Instruct) — Phase 2 only ──────────────
run_phase2_only "Qwen/Qwen2-VL-7B"
run_phase2_only "Qwen/Qwen2-VL-7B-Instruct"

# ── InternVL2 — all phases ───────────────────────────────────
run_all_phases "OpenGVLab/InternVL2-8B"

# ── InternVL2.5 MPO — all phases ─────────────────────────────
run_all_phases "OpenGVLab/InternVL2_5-8B-MPO"

echo ""
echo "============================================================"
echo "  24 GB+ run complete."
echo "============================================================"
